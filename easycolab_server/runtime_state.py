"""EasyColab 服务端模块: runtime_state."""

from PIL import Image
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import base64
import io
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

main_mod = sys.modules.get("__main__")
_DEPLOY_PASSWORD = getattr(main_mod, "_DEPLOY_PASSWORD", "") if main_mod else os.environ.pop("EASYCOLAB_ADMIN_PASSWORD", "")
try:
    from runtime_auth import (
        authenticated_ingress,
        COMFY_BACKEND,
        COMFY_INGRESS,
        GRADIO_BACKEND,
        GRADIO_INGRESS,
        TOYBOX_BACKEND,
        TOYBOX_INGRESS,
    )
except ImportError:
    authenticated_ingress = None
    GRADIO_INGRESS = 7860
    GRADIO_BACKEND = 27860
    COMFY_INGRESS = 8188
    COMFY_BACKEND = 28188
    TOYBOX_INGRESS = 8190
    TOYBOX_BACKEND = 28190

if main_mod and getattr(main_mod, "__file__", "").endswith("app.py") and not _DEPLOY_PASSWORD and (authenticated_ingress is None or not sys.stdin.isatty()):
    GRADIO_BACKEND = GRADIO_INGRESS
    COMFY_BACKEND = COMFY_INGRESS
    TOYBOX_BACKEND = TOYBOX_INGRESS

_IS_LOCAL_PREVIEW = (
    os.environ.get("EASYCOLAB_LOCAL_PREVIEW", "").strip() in ("1", "true", "yes")
    or "--local" in sys.argv
)

def resolve_callable(name: str, default_fn):
    """兼容单元测试在 app 模块上直接 mock.patch.object(app, ...) 的解析代理。"""
    app_mod = sys.modules.get("app") or sys.modules.get("__main__")
    if app_mod is not None and app_mod is not sys.modules.get(__name__):
        cand = getattr(app_mod, name, None)
        if callable(cand) and cand is not default_fn:
            return cand
    return default_fn

# ==============================================================================
# 实时终端进度与 tqdm 捕获器 (将 Colab 网页端的实时下载进度同步给 macOS App)
# ==============================================================================
class TerminalProgressMirror:
    def __init__(self, orig_stream, buffer_deque: deque, active_bars: dict, lock: threading.Lock):
        self.orig_stream = orig_stream
        self.buffer_deque = buffer_deque
        self.active_bars = active_bars
        self.lock = lock
        self._tls = threading.local()
        self._ansi_re = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    def write(self, data: str):
        try:
            self.orig_stream.write(data)
        except Exception:
            pass
        if not data:
            return
        clean = self._ansi_re.sub("", data)
        partial = getattr(self._tls, "partial", "")
        with self.lock:
            for ch in clean:
                if ch in ("\r", "\n"):
                    line = partial.strip()
                    if line:
                        # 识别类似 "model-00001-of-00014.safetensors: 22%..." 或无前缀采样/VAE进度条 "54%|#####4 | 19/35 [08:31<07:11...]"
                        if "%" in line and ("it/s" in line or "B/s" in line or "step" in line or "|" in line):
                            pct_idx = line.find("%")
                            colon_idx = line.find(":")
                            if 0 < colon_idx < pct_idx:
                                key = line[:colon_idx].strip()
                                self.active_bars[key] = line
                            else:
                                stage_lbl = CURRENT_STAGE_BAR_LABEL
                                if stage_lbl:
                                    # 若在 VAE 解码等子阶段内部触发 tqdm，标注阶段名避免子阶段 100% 被误认为整条视频已生成完毕
                                    if line.startswith("100%"):
                                        self.active_bars["sampling"] = f"{stage_lbl}: 子阶段已完成，正在继续后续合成封装..."
                                    else:
                                        self.active_bars["sampling"] = f"{stage_lbl}: {line}"
                                else:
                                    self.active_bars["sampling"] = line
                        else:
                            self.buffer_deque.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
                    partial = ""
                else:
                    partial += ch
        self._tls.partial = partial

    def flush(self):
        try:
            self.orig_stream.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.orig_stream, "isatty", lambda: False)()

CONSOLE_LOCK = threading.Lock()

CONSOLE_LINES: deque = deque(maxlen=120)

ACTIVE_TQDM_BARS: Dict[str, str] = {}

CURRENT_STAGE_BAR_LABEL: Optional[str] = None

if not isinstance(sys.stdout, TerminalProgressMirror):
    sys.stdout = TerminalProgressMirror(sys.__stdout__, CONSOLE_LINES, ACTIVE_TQDM_BARS, CONSOLE_LOCK)

if not isinstance(sys.stderr, TerminalProgressMirror):
    sys.stderr = TerminalProgressMirror(sys.__stderr__, CONSOLE_LINES, ACTIVE_TQDM_BARS, CONSOLE_LOCK)

def get_live_console_Lines() -> List[str]:
    with CONSOLE_LOCK:
        bars = list(ACTIVE_TQDM_BARS.values())[-10:]
        logs = list(CONSOLE_LINES)[-60:]
        return logs + (["─── 实时下载/采样进度流 ───"] + bars if bars else [])

# ==============================================================================
# 自动补齐音视频依赖 (防止出现 No module named 'av' 等错误)
# ==============================================================================
def ensure_runtime_dependencies():
    missing = []
    for pkg, mod in [
        ("av", "av"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("sentencepiece", "sentencepiece"),
        ("tiktoken", "tiktoken"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"正在自动安装缺失的运行期依赖: {', '.join(missing)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("缺失依赖自动安装完成。")

if not _IS_LOCAL_PREVIEW:
    ensure_runtime_dependencies()

try:
    import torch
except ImportError:
    if not _IS_LOCAL_PREVIEW:
        raise
    import types as _types

    class _StubCuda:
        OutOfMemoryError = RuntimeError
        @staticmethod
        def is_available() -> bool:
            return False
        @staticmethod
        def mem_get_info(_dev=None):
            return (0, 0)
        @staticmethod
        def empty_cache():
            pass

    torch = _types.SimpleNamespace(
        cuda=_StubCuda(),
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
        Tensor=object,
        nn=_types.SimpleNamespace(Module=object, ModuleList=list),
        randint=lambda low, high, size: _types.SimpleNamespace(item=lambda: 42),
        Generator=lambda device="cpu": _types.SimpleNamespace(manual_seed=lambda s: None),
    )
    sys.modules["torch"] = torch


# ==============================================================================
# 全局状态、算力计费账本与 /dev/shm 内存盘加速 (借鉴 comfy-colab-kit)
# ==============================================================================
BOOT_TIME = time.time()

LAST_ACTIVE_TIME = time.time()

try:
    IDLE_TIMEOUT_MINUTES = int(os.environ.get("EASYCOLAB_IDLE_TIMEOUT_MINUTES", "30").strip())
except Exception:
    IDLE_TIMEOUT_MINUTES = 30  # 默认闲置 30 分钟自动释放 Colab 虚拟机，可在 Mac 端调节

PIPELINE = None

LLM_BUNDLE = None  # {"backend": "transformers"|"gguf", ...}

LOADED_RUNTIMES: Dict[str, Dict[str, Any]] = {}  # 多模型显存共驻表: model_id -> runtime dict

SELECTED_MODEL_ID = "minimax_h3"

LOADED_MODEL_ID = None

CURRENT_MODEL_MODE = None

QWEN_CONTEXT_LENGTH = 32768  # 用户可自定义上下文长度 (4K ~ 256K = 262144 tokens)

def _parse_env_api_tokens() -> List[str]:
    """从环境变量读取预置 OpenAI/Anthropic 推理 API Token（支持逗号/空格分隔多密钥）。"""
    tokens: List[str] = []
    for env_name in ("EASYCOLAB_API_TOKENS", "EASYCOLAB_API_KEY", "OPENAI_API_KEY"):
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            continue
        for part in re.split(r"[\s,;]+", raw):
            tok = part.strip()
            if tok and tok not in tokens:
                tokens.append(tok)
    return tokens

_ENV_API_TOKENS: List[str] = _parse_env_api_tokens()

_PERSISTED_API_KEY_FILE = "/content/.easycolab_api_key"
INSTANCE_NAME: str = os.environ.get("EASYCOLAB_INSTANCE_NAME", "").strip()
BUNDLE_HASH: str = os.environ.get("EASYCOLAB_BUNDLE_HASH", "").strip()


def _load_or_create_initial_api_key() -> str:
    if _ENV_API_TOKENS:
        return _ENV_API_TOKENS[0]
    if not _IS_LOCAL_PREVIEW and os.path.isdir("/content"):
        try:
            if os.path.isfile(_PERSISTED_API_KEY_FILE):
                with open(_PERSISTED_API_KEY_FILE, "r", encoding="utf-8") as kf:
                    saved = kf.read().strip()
                if saved:
                    return saved
        except Exception:
            pass
    generated = f"sk-colab-{secrets.token_urlsafe(24)}"
    if not _IS_LOCAL_PREVIEW and os.path.isdir("/content"):
        try:
            fd = os.open(_PERSISTED_API_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as kf:
                kf.write(generated)
        except Exception:
            pass
    return generated


INITIAL_API_KEY = _load_or_create_initial_api_key()

API_AUTH_KEY = INITIAL_API_KEY

EXTRA_API_TOKENS: List[str] = list(_ENV_API_TOKENS[1:])

API_KEY_LOCK = threading.Lock()

def _get_valid_api_tokens() -> List[str]:
    """返回当前所有有效的 API Token 列表（线程安全去重）。"""
    with API_KEY_LOCK:
        candidates = [API_AUTH_KEY] + list(EXTRA_API_TOKENS)
    valid: List[str] = []
    for t in candidates:
        s = (t or "").strip()
        if s and s not in valid:
            valid.append(s)
    return valid

API_REQUEST_COUNT = 0

GPU_LOCK = threading.Lock()

_ACTIVE_API_STREAMS = 0

_ACTIVE_API_STREAMS_LOCK = threading.Lock()

ABORT_GENERATION_EVENT = threading.Event()

JOBS: Dict[str, Dict[str, Any]] = {}

_JOB_CANCEL_EVENTS: Dict[str, threading.Event] = {}

MODEL_DOWNLOAD_STATE: Dict[str, Any] = {
    "is_downloading": False,
    "is_loading_vram": False,
    "progress": 0.0,
    "message": "等待检查模型状态...",
}

def _get_system_ram_gb() -> Tuple[float, float]:
    """返回 (系统总内存 GB, 当前可用内存 GB)"""
    try:
        mem_total, mem_avail = 0.0, 0.0
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) / (1024 ** 2)
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1]) / (1024 ** 2)
        return round(mem_total, 1), round(mem_avail, 1)
    except Exception:
        return 0.0, 0.0

def setup_colab_ramdisk() -> Optional[str]:
    """
    借鉴 comfy-colab-kit/packages/colab/setup.py：
    在 Colab 大内存实例 (G4 176GB / A100 83GB / L4 53GB RAM) 上自动扩容 /dev/shm (tmpfs 内存盘，读写 2.4GB/s+)，
    为适配容量的模型提供零延迟 mmap 预热加速。
    """
    if not os.path.exists("/dev/shm"):
        return None
    total_ram_gb, avail_ram_gb = _get_system_ram_gb()
    if total_ram_gb >= 48.0:
        # 避免 DiffSynth DiskMap 默认每 2GB (10^9 参数) 反复关闭并重开全部 14 个 safetensors 分片句柄
        os.environ.setdefault("DIFFSYNTH_DISK_MAP_BUFFER_SIZE", str(80 * 10**9))
    if total_ram_gb < 40.0 or avail_ram_gb < 25.0:
        return None
    try:
        cur_shm_total_gb = shutil.disk_usage("/dev/shm").total / (1024 ** 3)
        target_gb = min(88, max(30, int(total_ram_gb * 0.50)))
        if cur_shm_total_gb + 1.0 < target_gb:
            subprocess.run(
                ["mount", "-o", f"remount,size={target_gb}G", "/dev/shm"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        shm_stat = shutil.disk_usage("/dev/shm")
        shm_free_gb = shm_stat.free / (1024 ** 3)
        if shm_free_gb >= 22.5:
            ram_root = "/dev/shm/EasyColab_Models"
            os.makedirs(ram_root, exist_ok=True)
            print(f"⚡ [RAMDisk 内存盘加速已开启] /dev/shm 可用空间 {shm_free_gb:.1f} GB (2.4GB/s 零延迟读写)")
            return ram_root
    except Exception:
        pass
    return None

RAM_ROOT_DIR = setup_colab_ramdisk()

# 如果挂载了 Google Drive，优先将模型持久化保存在云盘 EasyColab_Models（并自动兼容读取历史 Aevix_Models / MiniMax_H3_Models 目录）
DRIVE_MOUNTED = os.path.exists("/content/drive/MyDrive")

def _has_rclone_drive_remote() -> bool:
    rc_path = os.path.expanduser("~/.config/rclone/rclone.conf")
    if not os.path.exists(rc_path) or not shutil.which("rclone"):
        return False
    try:
        with open(rc_path, "r", encoding="utf-8", errors="ignore") as f:
            return "[easycolab_gdrive]" in f.read()
    except Exception:
        return False

COMFYUI_LOCK = threading.Lock()

COMFYUI_STATE: Dict[str, Any] = {
    "is_running": False,
    "is_starting": False,
    "is_syncing_drive": False,
    "is_downloading_weight": False,
    "installing_pack_id": "",
    "progress": 0.0,
    "status_message": "尚未启动（支持一键拉起 ComfyUI + Google Drive 自动软链持久化）",
    "public_url": "",
    "local_port": COMFY_BACKEND,
    "queue_running": 0,
    "queue_pending": 0,
    "persistence_mode": "drive_symlink" if DRIVE_MOUNTED else ("rclone_drive_sync" if _has_rclone_drive_remote() else "local_only"),
    "persistence_label": "Google Drive 实时软链挂载" if DRIVE_MOUNTED else ("Google Drive 云盘自动持久化" if _has_rclone_drive_remote() else "实例本地高速存储"),
    "drive_persisted": bool(DRIVE_MOUNTED or _has_rclone_drive_remote()),
    "custom_nodes_count": 0,
    "models_count": 0,
    "workflows_count": 0,
    "last_sync_ts": 0.0,
    "last_history_count": 0,
    "comfy_proc": None,
    "tunnel_proc": None,
}

TOYBOX_LOCK = threading.Lock()

TOYBOX_STATE: Dict[str, Any] = {
    "is_running": False,
    "is_starting": False,
    "is_preparing_assets": False,
    "progress": 0.0,
    "status_message": "尚未启动（复用底层 ComfyUI 引擎 + 独立带密码公网隧道）",
    "public_url": "",
    "local_port": TOYBOX_BACKEND,
    "ingress_port": TOYBOX_INGRESS,
    "toybox_proc": None,
    "tunnel_proc": None,
}

MODEL_BASE_ROOT = (
    "/content/drive/MyDrive/EasyColab_Models"
    if DRIVE_MOUNTED
    else os.path.join(os.getcwd(), "models")
)

LEGACY_DRIVE_MODEL_ROOT = "/content/drive/MyDrive/MiniMax_H3_Models"

LEGACY_DRIVE_MODEL_ROOTS = [
    "/content/drive/MyDrive/Aevix_Models",
    "/content/drive/MyDrive/MiniMax_H3_Models",
]

os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = MODEL_BASE_ROOT

# ==============================================================================
# 借鉴 comfy-colab-kit 的 8 线程 HTTP Range 并行极速下载器 (400~700 MB/s)
# ==============================================================================
class _StripAuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    """重定向到 HuggingFace CDN (CloudFront/Xet) 时剥离 Authorization 头，防止 400/403 拒绝"""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None and "Authorization" in new_req.unredirected_hdrs:
            del new_req.unredirected_hdrs["Authorization"]
        if new_req is not None and "Authorization" in new_req.headers:
            del new_req.headers["Authorization"]
        return new_req

_JUPYTER_PING_HOOK = None

def touch_activity():
    global LAST_ACTIVE_TIME
    LAST_ACTIVE_TIME = time.time()
    if _JUPYTER_PING_HOOK is not None:
        _JUPYTER_PING_HOOK(force=False)

def _safe_shm_available_bytes(reserve_ram_gb: float = 34.0) -> int:
    """
    计算 /dev/shm 内存盘真正可安全写入的字节数：
    Colab 容器中 /dev/shm (tmpfs) 与进程匿名内存共享同一个 Cgroup 物理内存上限（如 G4 176GB）。
    若 MiniMax-H3 已锁入 ~128GB 系统内存热备，shutil.disk_usage('/dev/shm').free 仍会虚报 ~88GB 空闲，
    此时继续向 /dev/shm 写入大权重会直接击穿容器 Cgroup 触发 OOM 强杀。
    因此必须同时受限于 /dev/shm 配额剩余空间与 /proc/meminfo MemAvailable 安全余量。
    """
    if not os.path.exists("/dev/shm"):
        return 0
    try:
        shm_free = int(shutil.disk_usage("/dev/shm").free)
    except Exception:
        return 0
    total_ram_gb, avail_ram_gb = _get_system_ram_gb()
    if total_ram_gb > 0:
        eff_reserve = reserve_ram_gb
        try:
            if any(
                ("内存热备" in str(rt.get("mode_label", "")) or "ComponentsManager" in str(rt.get("mode_label", "")))
                for rt in LOADED_RUNTIMES.values()
            ):
                eff_reserve = max(eff_reserve, 40.0)
        except Exception:
            pass
        ram_budget = max(0, int((avail_ram_gb - eff_reserve) * (1024 ** 3)))
        return min(shm_free, ram_budget)
    return shm_free

_TELEMETRY_LOCK = threading.Lock()

def _make_noproxy_opener():
    """本地回环请求 (127.0.0.1 llama-server) 强制绕过环境变量代理"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))

def decode_base64_image(b64_str: Optional[str]) -> Optional[Image.Image]:
    if not b64_str:
        return None
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")

def _load_pil_from_url_or_b64(src: str) -> Optional[Image.Image]:
    if not src:
        return None
    src = src.strip()
    if src.startswith("http://") or src.startswith("https://"):
        req = urllib.request.Request(src, headers={"User-Agent": "EasyColab-API-Gateway/2.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    return decode_base64_image(src)


DEFAULT_OUTPUT_DIR = (
    "/content/drive/MyDrive/EasyColab_Outputs"
    if DRIVE_MOUNTED
    else os.path.join(os.getcwd(), "outputs")
)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "models"), exist_ok=True)
