"""EasyColab 服务端模块: telemetry_service."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import base64
import json
import os
import shutil
import socket
import threading
import time
import urllib.parse
import urllib.request
import uuid

from . import runtime_state as rs
from .runtime_state import torch
from .comfyui_service import _poll_local_comfyui_runtime_quiet, sync_comfyui_to_drive_core
from .runtime_state import COMFYUI_STATE, DRIVE_MOUNTED, GPU_LOCK, MODEL_DOWNLOAD_STATE, TOYBOX_STATE, _ACTIVE_API_STREAMS_LOCK, _TELEMETRY_LOCK, _get_system_ram_gb, torch, touch_activity
from .toybox_service import _poll_local_toybox_runtime_quiet, stop_toybox_core

_LAST_JUPYTER_HEARTBEAT_TS: float = 0.0

_JUPYTER_HB_WAKE_EVENT = threading.Event()

_STOP_JUPYTER_KEEPALIVE = False

def _build_ws_masked_text_frame(text: str) -> bytes:
    """构建标准 RFC 6455 客户端掩码文本帧 (opcode=0x1, FIN=1)，零第三方依赖连接本地 Jupyter WebSocket"""
    payload = text.encode("utf-8")
    length = len(payload)
    mask_key = os.urandom(4)
    header = bytearray([0x81])
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(length.to_bytes(2, "big"))
    else:
        header.append(0x80 | 127)
        header.extend(length.to_bytes(8, "big"))
    header.extend(mask_key)
    masked = bytearray(length)
    for i in range(length):
        masked[i] = payload[i] ^ mask_key[i & 3]
    return bytes(header) + bytes(masked)

def _build_ws_masked_pong_frame(payload: bytes = b"") -> bytes:
    mask_key = os.urandom(4)
    length = min(len(payload), 125)
    header = bytearray([0x8A, 0x80 | length])
    header.extend(mask_key)
    masked = bytes(payload[i] ^ mask_key[i & 3] for i in range(length))
    return bytes(header) + masked

def _ensure_local_colab_kernel_id() -> Optional[str]:
    """查询或启动本地 127.0.0.1:8080 上的 Jupyter 内核并返回 kernel_id"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Colab-Client-Agent": "vscode",
    }
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/api/kernels", headers=headers)
        with opener.open(req, timeout=3.0) as resp:
            kernels = json.loads(resp.read().decode("utf-8", errors="replace"))
        if isinstance(kernels, list) and kernels:
            for k in kernels:
                kid = k.get("id")
                if kid:
                    return str(kid)
    except Exception:
        return None

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8080/api/kernels",
            data=b'{"name":"python3"}',
            headers=headers,
            method="POST",
        )
        with opener.open(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        kid = data.get("id")
        if kid:
            return str(kid)
    except Exception:
        pass
    return None

def _open_local_jupyter_ws_channel(kernel_id: str, session_id: str) -> Optional[socket.socket]:
    """建立到 ws://127.0.0.1:8080/api/kernels/<id>/channels 的常驻 WebSocket 连接，使内核 connections 始终 >= 1"""
    sock = None
    try:
        sock = socket.create_connection(("127.0.0.1", 8080), timeout=5.0)
        ws_key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = f"/api/kernels/{kernel_id}/channels?session_id={session_id}"
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:8080\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {ws_key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"X-Colab-Client-Agent: vscode\r\n"
            f"Origin: http://127.0.0.1:8080\r\n\r\n"
        )
        sock.sendall(handshake.encode("ascii"))
        buf = b""
        while b"\r\n\r\n" not in buf and len(buf) < 8192:
            chunk = sock.recv(1024)
            if not chunk:
                break
            buf += chunk
        status_line = buf.split(b"\r\n", 1)[0]
        if b" 101 " not in status_line:
            sock.close()
            return None
        sock.settimeout(0.2)
        return sock
    except Exception:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        return None

def _recv_ws_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("WebSocket EOF mid-frame")
        buf.extend(chunk)
    return bytes(buf)

def _recv_ws_frame(sock: socket.socket, poll_timeout: float = 0.2) -> Optional[Tuple[int, bytes]]:
    """
    严格按 RFC 6455 帧边界解析单帧，返回 (opcode, payload)。
    首个字节到达前若超时无数据返回 None；对端断开连接则抛出 ConnectionError。
    杜绝直接用 sock.recv(4096)[0] & 0x0F 将分片 JSON 载荷字节 ('h'/'x'/'8'/'(') 误判为 Close 帧。
    """
    prev_timeout = sock.gettimeout()
    try:
        sock.settimeout(poll_timeout)
        try:
            first_two = sock.recv(2)
        except socket.timeout:
            return None
        if not first_two:
            raise ConnectionError("WebSocket closed by peer")
        sock.settimeout(2.5)
        if len(first_two) == 1:
            first_two += _recv_ws_exact(sock, 1)
        b0, b1 = first_two[0], first_two[1]
        opcode = b0 & 0x0F
        masked = (b1 & 0x80) != 0
        payload_len = b1 & 0x7F
        if payload_len == 126:
            payload_len = int.from_bytes(_recv_ws_exact(sock, 2), "big")
        elif payload_len == 127:
            payload_len = int.from_bytes(_recv_ws_exact(sock, 8), "big")
        if payload_len > 32 * 1024 * 1024:
            raise ConnectionError(f"Oversized WebSocket frame: {payload_len}")
        mask_key = _recv_ws_exact(sock, 4) if masked else b""
        payload = _recv_ws_exact(sock, payload_len) if payload_len > 0 else b""
        if masked and payload:
            payload = bytes(payload[i] ^ mask_key[i & 3] for i in range(payload_len))
        return opcode, payload
    finally:
        try:
            sock.settimeout(prev_timeout)
        except Exception:
            pass

def _drain_local_ws_frames(sock: socket.socket, max_frames: int = 24, poll_timeout: float = 0.15) -> bool:
    """按完整 RFC 6455 帧排空本地 Jupyter WebSocket 缓冲区：收到 Ping 回显原 payload Pong，收到 Close 返回 False"""
    try:
        for idx in range(max_frames):
            frame = _recv_ws_frame(sock, poll_timeout=poll_timeout if idx == 0 else 0.05)
            if frame is None:
                break
            opcode, payload = frame
            if opcode == 0x08:
                return False
            if opcode == 0x09:
                sock.sendall(_build_ws_masked_pong_frame(payload))
        return True
    except Exception:
        return False

def _send_jupyter_ws_execute_heartbeat(sock: socket.socket, session_id: str) -> bool:
    """在已连接的本地 Jupyter WebSocket 通道上发送轻量 execute_request，真实刷新内核 last_activity 与 execution_state"""
    msg = {
        "channel": "shell",
        "header": {
            "msg_id": uuid.uuid4().hex,
            "username": "easycolab",
            "session": session_id,
            "date": datetime.utcnow().isoformat() + "Z",
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": "_easycolab_hb = 1",
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": False,
        },
    }
    try:
        frame = _build_ws_masked_text_frame(json.dumps(msg))
        sock.sendall(frame)
        return _drain_local_ws_frames(sock, max_frames=32, poll_timeout=0.25)
    except Exception:
        return False

def _local_jupyter_ws_keepalive_loop():
    """
    后台守护线程：只要当前处于活跃保活窗口内，始终保持一条连接到本地 Colab Jupyter Server
    (ws://127.0.0.1:8080/api/kernels/<id>/channels) 的常驻 WebSocket，确保内核 connections >= 1，
    并每 30 秒发送一次 execute_request 脉冲刷新 last_activity，彻底杜绝无网页模式下被 Colab 误判闲置回收。
    """
    session_id = uuid.uuid4().hex
    ws_sock: Optional[socket.socket] = None
    active_kid: Optional[str] = None
    last_exec_ts = 0.0

    while not _STOP_JUPYTER_KEEPALIVE:
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for ep in ("http://127.0.0.1:8080/api/kernels", "http://127.0.0.1:8080/api/sessions", "http://127.0.0.1:8080/api/status"):
                try:
                    req = urllib.request.Request(ep, headers={"X-Colab-Client-Agent": "vscode"})
                    with opener.open(req, timeout=2.0) as resp:
                        resp.read(256)
                except Exception:
                    pass

            if ws_sock is None:
                active_kid = _ensure_local_colab_kernel_id()
                if active_kid:
                    ws_sock = _open_local_jupyter_ws_channel(active_kid, session_id)
                    last_exec_ts = 0.0

            now = time.time()
            if ws_sock is not None and (now - last_exec_ts >= 30.0 or _JUPYTER_HB_WAKE_EVENT.is_set()):
                if _send_jupyter_ws_execute_heartbeat(ws_sock, session_id):
                    last_exec_ts = now
                else:
                    try:
                        ws_sock.close()
                    except Exception:
                        pass
                    ws_sock = None
            elif ws_sock is not None:
                # 非发送周期也按 RFC 6455 帧边界排空控制帧，保持连接健康
                if not _drain_local_ws_frames(ws_sock, max_frames=16, poll_timeout=0.15):
                    try:
                        ws_sock.close()
                    except Exception:
                        pass
                    ws_sock = None
        except Exception:
            pass

        _JUPYTER_HB_WAKE_EVENT.wait(timeout=5.0)
        _JUPYTER_HB_WAKE_EVENT.clear()

    if ws_sock is not None:
        try:
            ws_sock.close()
        except Exception:
            pass

threading.Thread(target=_local_jupyter_ws_keepalive_loop, daemon=True).start()

def _ping_local_colab_jupyter(force: bool = False):
    """触发本地常驻 Jupyter WebSocket 保活线程立即执行一次脉冲"""
    global _LAST_JUPYTER_HEARTBEAT_TS
    now = time.time()
    if not force and (now - _LAST_JUPYTER_HEARTBEAT_TS) < 25.0:
        return
    _LAST_JUPYTER_HEARTBEAT_TS = now
    _JUPYTER_HB_WAKE_EVENT.set()

class _ActiveApiStreamGuard:
    """标记当前正在进行的流式/长耗时网关请求，避免长思维链生成期间被闲置看门狗误判"""

    def __enter__(self):
        # (shared state on runtime_state)
        with _ACTIVE_API_STREAMS_LOCK:
            rs._ACTIVE_API_STREAMS += 1
        touch_activity()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # (shared state on runtime_state)
        with _ACTIVE_API_STREAMS_LOCK:
            rs._ACTIVE_API_STREAMS = max(0, rs._ACTIVE_API_STREAMS - 1)
        touch_activity()
        return False

def estimate_cu_burn_rate(gpu_name: str) -> float:
    """根据 Colab GPU 型号估算每小时消耗的计算单元 (CU/hr)"""
    name_up = (gpu_name or "").upper()
    if "H100" in name_up:
        return 11.78
    elif "PRO 6000" in name_up or "BLACKWELL" in name_up or "G4" in name_up:
        return 10.05
    elif "A100" in name_up:
        return 5.37
    elif "L4" in name_up:
        return 1.71
    elif "T4" in name_up:
        return 1.17
    elif "V100" in name_up:
        return 3.80
    return 1.50

def _idle_watchdog_loop():
    """借鉴 comfy-colab-kit 的 stopper.py：当超过设定闲置时间无任务时自动释放 Colab 虚拟机"""
    global _STOP_JUPYTER_KEEPALIVE
    while True:
        time.sleep(15)
        # 检查客户端桥接层 (cmd_keepalive) 写入的保活时间戳文件
        try:
            if os.path.exists("/tmp/.easycolab_client_hb"):
                hb_mtime = os.path.getmtime("/tmp/.easycolab_client_hb")
                if hb_mtime > rs.LAST_ACTIVE_TIME:
                    touch_activity()
        except Exception:
            pass

        # 若正在下载、加载模型、生成视频/LLM推理或 ComfyUI 正在运行/安装/执行节点队列，持续刷新活跃时间并向本地 Jupyter 发送保活脉冲
        if (
            MODEL_DOWNLOAD_STATE["is_downloading"]
            or MODEL_DOWNLOAD_STATE["is_loading_vram"]
            or GPU_LOCK.locked()
            or rs._ACTIVE_API_STREAMS > 0
            or COMFYUI_STATE.get("is_running")
            or COMFYUI_STATE["is_starting"]
            or COMFYUI_STATE["is_downloading_weight"]
            or COMFYUI_STATE["queue_running"] > 0
            or COMFYUI_STATE["queue_pending"] > 0
            or TOYBOX_STATE.get("is_running")
            or TOYBOX_STATE.get("is_starting")
            or TOYBOX_STATE.get("is_preparing_assets")
        ):
            touch_activity()
            continue

        if rs.IDLE_TIMEOUT_MINUTES <= 0:
            _ping_local_colab_jupyter(force=False)
            continue

        idle_sec = time.time() - rs.LAST_ACTIVE_TIME
        if idle_sec >= rs.IDLE_TIMEOUT_MINUTES * 60:
            _STOP_JUPYTER_KEEPALIVE = True
            _JUPYTER_HB_WAKE_EVENT.set()
            print(f"🛑 [自动省额度看门狗] 已连续闲置 {rs.IDLE_TIMEOUT_MINUTES} 分钟，正在自动释放 Colab 虚拟机...")
            disconnect_colab_runtime()
            break
        else:
            # 尚在用户设定的活跃保活窗口内，轻触本地 Jupyter 防止 Colab 提前强杀
            _ping_local_colab_jupyter(force=False)

threading.Thread(target=_idle_watchdog_loop, daemon=True).start()

_CACHED_DRIVE_METRICS: Dict[str, Any] = {
    "ts": 0.0,
    "drive_total_gb": 0.0,
    "drive_used_gb": 0.0,
    "drive_free_gb": 0.0,
}

_CACHED_LIVE_TELEMETRY: Dict[str, Any] = {}

def _get_storage_metrics_dict(refresh_drive: bool = False) -> Dict[str, Any]:
    """采集 Colab 实例系统内存 (RAM)、本地磁盘、/dev/shm 内存加速盘及 Google Drive 云盘指标 (单位 GB)"""
    ram_total_gb, ram_free_gb = _get_system_ram_gb()
    ram_used_gb = round(max(0.0, ram_total_gb - ram_free_gb), 1) if ram_total_gb > 0 else 0.0

    disk_path = "/content" if os.path.exists("/content") else os.getcwd()
    disk_total_gb, disk_used_gb, disk_free_gb = 0.0, 0.0, 0.0
    try:
        du = shutil.disk_usage(disk_path)
        disk_total_gb = round(du.total / (1024 ** 3), 1)
        disk_free_gb = round(du.free / (1024 ** 3), 1)
        disk_used_gb = round(max(0.0, (du.total - du.free) / (1024 ** 3)), 1)
    except Exception:
        pass

    shm_total_gb, shm_used_gb, shm_free_gb = 0.0, 0.0, 0.0
    if os.path.exists("/dev/shm"):
        try:
            su = shutil.disk_usage("/dev/shm")
            shm_total_gb = round(su.total / (1024 ** 3), 1)
            shm_free_gb = round(su.free / (1024 ** 3), 1)
            shm_used_gb = round(max(0.0, (su.total - su.free) / (1024 ** 3)), 1)
        except Exception:
            pass

    now = time.time()
    if DRIVE_MOUNTED and (refresh_drive or (now - _CACHED_DRIVE_METRICS["ts"]) >= 15.0):
        if os.path.exists("/content/drive/MyDrive"):
            try:
                gu = shutil.disk_usage("/content/drive/MyDrive")
                _CACHED_DRIVE_METRICS["drive_total_gb"] = round(gu.total / (1024 ** 3), 1)
                _CACHED_DRIVE_METRICS["drive_free_gb"] = round(gu.free / (1024 ** 3), 1)
                _CACHED_DRIVE_METRICS["drive_used_gb"] = round(max(0.0, (gu.total - gu.free) / (1024 ** 3)), 1)
                _CACHED_DRIVE_METRICS["ts"] = now
            except Exception:
                pass

    return {
        "ram_total_gb": ram_total_gb,
        "ram_used_gb": ram_used_gb,
        "ram_free_gb": ram_free_gb,
        "disk_total_gb": disk_total_gb,
        "disk_used_gb": disk_used_gb,
        "disk_free_gb": disk_free_gb,
        "shm_total_gb": shm_total_gb,
        "shm_used_gb": shm_used_gb,
        "shm_free_gb": shm_free_gb,
        "drive_total_gb": _CACHED_DRIVE_METRICS["drive_total_gb"],
        "drive_used_gb": _CACHED_DRIVE_METRICS["drive_used_gb"],
        "drive_free_gb": _CACHED_DRIVE_METRICS["drive_free_gb"],
    }

def _sample_live_hardware_telemetry_once() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        gpu_name = "CPU Mode (未检测到 GPU)"
        cuda_avail = False
        free_gb, total_gb, used_gb = 0.0, 0.0, 0.0
        cu_rate = 0.10
    else:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_avail = True
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        free_gb = round(free_mem / (1024 ** 3), 2)
        total_gb = round(total_mem / (1024 ** 3), 2)
        used_gb = round(total_gb - free_gb, 2)
        cu_rate = estimate_cu_burn_rate(gpu_name)

    snap = {
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "vram_free_gb": free_gb,
        "vram_total_gb": total_gb,
        "vram_used_gb": used_gb,
        "cu_burn_rate_per_hour": cu_rate,
    }
    snap.update(_get_storage_metrics_dict(refresh_drive=False))
    if "_poll_local_comfyui_runtime_quiet" in globals():
        _poll_local_comfyui_runtime_quiet()
    if "_poll_local_toybox_runtime_quiet" in globals():
        _poll_local_toybox_runtime_quiet()
    with _TELEMETRY_LOCK:
        _CACHED_LIVE_TELEMETRY.clear()
        _CACHED_LIVE_TELEMETRY.update(snap)
    return snap

def _telemetry_sampler_loop():
    """独立后台线程以 2s 周期采样 GPU 显存/内存/磁盘指标，保证 HTTP 轮询非阻塞秒回"""
    while True:
        try:
            _sample_live_hardware_telemetry_once()
        except Exception:
            pass
        time.sleep(2.0)

threading.Thread(target=_telemetry_sampler_loop, daemon=True).start()

def get_live_telemetry_snapshot() -> Dict[str, Any]:
    with _TELEMETRY_LOCK:
        if _CACHED_LIVE_TELEMETRY:
            return dict(_CACHED_LIVE_TELEMETRY)
    return _sample_live_hardware_telemetry_once()

def disconnect_colab_runtime() -> str:
    try:
        if TOYBOX_STATE.get("is_running"):
            rs.resolve_callable("stop_toybox_core", stop_toybox_core)(stop_comfyui_too=False)
        if os.path.isdir("/content/ComfyUI") or COMFYUI_STATE.get("is_running"):
            rs.resolve_callable("sync_comfyui_to_drive_core", sync_comfyui_to_drive_core)(quiet=True)
    except Exception:
        pass
    try:
        from google.colab import runtime
        runtime.unassign()
        return "🛑 已向 Colab 发送释放运行时指令，实例即将关闭并停止扣除计算单元。"
    except ImportError:
        return "ℹ️ 当前不在 Google Colab 环境中，无需释放云端运行时。"
    except Exception as e:
        return f"⚠️ 释放运行时失败: {e}"

rs._JUPYTER_PING_HOOK = _ping_local_colab_jupyter
