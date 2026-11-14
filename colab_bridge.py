#!/usr/bin/env python3
"""
Local Headless Google Colab Bridge for EasyColab macOS App.
Powered by google-colab-cli (inspired by FireTable/comfy-colab-kit).
Supports:
  - auth_status : Check OAuth login state, real CU balance, hourly burn rate, and active VMs
  - auth_url    : Generate Google OAuth2 authorization URL for 1-click browser login
  - auth_code   : Exchange Google authorization code and save ~/.config/colab-cli/token.json
  - wake        : Provision/reuse Colab GPU VM (G4/H100/A100/L4/T4), deploy app.py, and return gradio.live URL
  - keepalive   : Refresh Colab runtime proxy & kernel activity while active
  - stop_all    : Unassign and terminate all active Colab VMs on the account (verify 0 CU/hr burn)
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import uuid
import signal
import hashlib
import argparse
import tempfile
import threading
import contextlib
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from importlib import resources

import requests
import websocket
from websocket import ABNF
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


def _ensure_utf8_stdio():
    """确保 Windows / macOS 子进程管道输出始终采用 UTF-8 编码，避免 emoji 在 GBK 代码页下抛出 UnicodeEncodeError"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_ensure_utf8_stdio()


def _sync_macos_system_proxies_to_env():
    """
    macOS / Linux 原生 GUI 应用启动子进程时默认可能不包含终端的 HTTP_PROXY / HTTPS_PROXY 环境变量，
    而 websocket-client (Jupyter 内核 WebSocket 通道) 仅从 os.environ 读取代理配置。
    此处自动读取 macOS / Windows / Linux 系统级代理 (scutil --proxy / Internet Settings / gsettings / urllib.request.getproxies) 并注入环境变量。
    """
    try:
        proxies = urllib.request.getproxies()
        for k in ("http", "https"):
            if k in proxies and proxies[k]:
                os.environ.setdefault(f"{k}_proxy", proxies[k])
                os.environ.setdefault(f"{k.upper()}_PROXY", proxies[k])
        if sys.platform.startswith("linux") and not (os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")):
            import subprocess
            mode = subprocess.run(
                ["gsettings", "get", "org.gnome.system.proxy", "mode"],
                capture_output=True, text=True, timeout=2
            ).stdout.strip().strip("'")
            if mode == "manual":
                for k in ("http", "https"):
                    host = subprocess.run(
                        ["gsettings", "get", f"org.gnome.system.proxy.{k}", "host"],
                        capture_output=True, text=True, timeout=2
                    ).stdout.strip().strip("'")
                    port_raw = subprocess.run(
                        ["gsettings", "get", f"org.gnome.system.proxy.{k}", "port"],
                        capture_output=True, text=True, timeout=2
                    ).stdout.strip()
                    port = port_raw.split()[-1].strip("'") if port_raw else ""
                    if host and port and port.isdigit() and int(port) > 0:
                        proxy_url = f"http://{host}:{port}"
                        os.environ.setdefault(f"{k}_proxy", proxy_url)
                        os.environ.setdefault(f"{k.upper()}_PROXY", proxy_url)
    except Exception:
        pass


_sync_macos_system_proxies_to_env()
os.environ.setdefault("COLAB_CLI_REQUEST_TIMEOUT", "75")

from colab_cli.auth import PUBLIC_SCOPES, TOKEN_CONFIG_PATH, REMOTE_REDIRECT_URI
from colab_cli.common import state
from colab_cli.client import (
    Accelerator,
    Variant,
    Shape,
    resolve_assign_shape,
    PostAssignmentResponse,
    ColabRequestError,
    TooManyAssignmentsError,
)
from colab_cli.utils import get_status_code
from colab_cli.state import SessionState
from colab_cli.runtime import ColabRuntime
from colab_cli.contents import ContentsClient


def _load_client_config() -> dict:
    cfg_path = os.path.expanduser("~/.colab-cli-oauth-config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    res = resources.files("colab_cli").joinpath("oauth_config.json")
    return json.loads(res.read_text(encoding="utf-8"))


def _has_valid_token() -> bool:
    if not os.path.exists(TOKEN_CONFIG_PATH):
        return False
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH, PUBLIC_SCOPES)
        if creds and creds.valid:
            return True
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
            return True
    except Exception:
        return False
    return False


PKCE_VERIFIER_PATH = os.path.expanduser("~/.config/colab-cli/pkce_verifier.txt")
ACCOUNT_PROFILE_CACHE_PATH = os.path.expanduser("~/.config/colab-cli/account_profile.json")


def _instance_registry_path() -> str:
    return os.path.join(os.path.dirname(TOKEN_CONFIG_PATH), "easycolab_instances.json")


def _load_instance_registry() -> dict[str, dict]:
    path = _instance_registry_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def _save_instance_registry(reg: dict[str, dict]) -> None:
    path = _instance_registry_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def _upsert_instance_registry(
    endpoint: str,
    *,
    name: str = "",
    accelerator: str = "",
    server_url: str = "",
    bundle_hash: str = "",
    managed: bool | None = None,
) -> None:
    if not endpoint:
        return
    reg = _load_instance_registry()
    cur = dict(reg.get(endpoint) or {})
    cur["endpoint"] = endpoint
    if name:
        cur["name"] = name
    if accelerator:
        cur["accelerator"] = accelerator
    if server_url:
        cur["server_url"] = server_url
    if bundle_hash:
        cur["bundle_hash"] = bundle_hash
    if managed is not None:
        cur["managed"] = bool(managed)
    elif "managed" not in cur:
        cur["managed"] = True
    cur["updated_at"] = time.time()
    reg[endpoint] = cur
    _save_instance_registry(reg)


def _remove_instance_registry(endpoint: str) -> None:
    if not endpoint:
        return
    reg = _load_instance_registry()
    if endpoint in reg:
        reg.pop(endpoint, None)
        _save_instance_registry(reg)


def _prune_instance_registry(active_endpoints: set[str]) -> dict[str, dict]:
    reg = _load_instance_registry()
    cleaned = {ep: meta for ep, meta in reg.items() if ep in active_endpoints}
    if len(cleaned) != len(reg):
        _save_instance_registry(cleaned)
    return cleaned


def _default_session_name(
    accel_value: str,
    endpoint: str,
    local_sessions: dict[str, SessionState],
    requested_name: str = "",
    registry: dict[str, dict] | None = None,
) -> str:
    req = re.sub(r"[^a-zA-Z0-9._-]+", "-", (requested_name or "").strip()).strip("-")
    if req:
        owner = local_sessions.get(req)
        if owner is None or owner.endpoint == endpoint:
            return req

    for s in local_sessions.values():
        if getattr(s, "endpoint", None) == endpoint and getattr(s, "name", None):
            return s.name

    if registry and endpoint in registry and registry[endpoint].get("name"):
        reg_name = str(registry[endpoint]["name"])
        owner = local_sessions.get(reg_name)
        if owner is None or owner.endpoint == endpoint:
            return reg_name

    short_ep = re.sub(r"[^a-zA-Z0-9]+", "", endpoint or "")[-6:].lower() or uuid.uuid4().hex[:6]
    base = f"easycolab-{(accel_value or 'gpu').lower()}-{short_ep}"
    candidate = base
    suffix = 2
    while candidate in local_sessions and local_sessions[candidate].endpoint != endpoint:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _is_managed_endpoint(
    endpoint: str,
    local_sessions: dict[str, SessionState],
    registry: dict[str, dict],
) -> bool:
    if not endpoint:
        return False
    if endpoint in registry:
        return bool(registry[endpoint].get("managed", True))
    return any(getattr(s, "endpoint", None) == endpoint for s in local_sessions.values())


def _compute_server_bundle_hash(app_py_path: str, srv_bundle: dict[str, str]) -> str:
    h = hashlib.sha256()
    for label, p in (
        ("app.py", app_py_path),
        ("runtime_auth.py", os.path.join(os.path.dirname(os.path.abspath(app_py_path)), "runtime_auth.py")),
    ):
        h.update(label.encode("utf-8"))
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except Exception:
            pass
    for fname in sorted(srv_bundle.keys()):
        h.update(f"easycolab_server/{fname}".encode("utf-8"))
        h.update(srv_bundle[fname].encode("utf-8"))
    return h.hexdigest()[:24]


def _clear_local_session_store():
    """切换或退出 Google 账号时，清理本地缓存的旧账号 Colab 会话记录"""
    try:
        existing = state.store.list()
        if isinstance(existing, dict):
            for name in list(existing.keys()):
                state.store.remove(name)
    except Exception:
        pass
    try:
        reg_path = _instance_registry_path()
        if os.path.exists(reg_path):
            os.remove(reg_path)
    except Exception:
        pass
    state._client = None
    state._sessions = None


def _get_account_email() -> str:
    """获取当前绑定 Google 账号邮箱（按 refresh_token 指纹本地缓存，避免重复请求）"""
    if not os.path.exists(TOKEN_CONFIG_PATH):
        return ""
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH, PUBLIC_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        tok_fp = (creds.refresh_token or creds.token or "")[-20:]
        if tok_fp and os.path.exists(ACCOUNT_PROFILE_CACHE_PATH):
            try:
                with open(ACCOUNT_PROFILE_CACHE_PATH, "r", encoding="utf-8") as cf:
                    cached = json.load(cf)
                if cached.get("fp") == tok_fp and cached.get("email"):
                    return str(cached["email"])
            except Exception:
                pass
        if creds and creds.token:
            r = requests.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {creds.token}"},
                timeout=5,
            )
            if r.status_code == 200:
                email = str(r.json().get("email") or "").strip()
                if email and tok_fp:
                    os.makedirs(os.path.dirname(ACCOUNT_PROFILE_CACHE_PATH), exist_ok=True)
                    with open(ACCOUNT_PROFILE_CACHE_PATH, "w", encoding="utf-8") as cf:
                        json.dump({"fp": tok_fp, "email": email}, cf)
                return email
    except Exception:
        pass
    return ""


def cmd_auth_url():
    client_config = _load_client_config()
    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    # 显式传入 select_account consent，强制弹出 Google 多账号选择器以支持自由切换账号
    auth_url, _ = flow.authorization_url(prompt="select_account consent", token_usage="remote")
    os.makedirs(os.path.dirname(PKCE_VERIFIER_PATH), exist_ok=True)
    if getattr(flow, "code_verifier", None):
        with open(PKCE_VERIFIER_PATH, "w", encoding="utf-8") as f:
            f.write(flow.code_verifier)
    print(json.dumps({"ok": True, "auth_url": auth_url}))


def cmd_auth_code(code: str):
    try:
        client_config = _load_client_config()
        flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
        flow.redirect_uri = REMOTE_REDIRECT_URI
        if os.path.exists(PKCE_VERIFIER_PATH):
            with open(PKCE_VERIFIER_PATH, "r", encoding="utf-8") as f:
                flow.code_verifier = f.read().strip()
        flow.fetch_token(code=code.strip())
        creds = flow.credentials
        _clear_local_session_store()
        for p in (ACCOUNT_PROFILE_CACHE_PATH, PKCE_VERIFIER_PATH):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        os.makedirs(os.path.dirname(TOKEN_CONFIG_PATH), exist_ok=True)
        with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        return cmd_auth_status()
    except Exception as e:
        print(json.dumps({"ok": False, "logged_in": False, "error": str(e)}))


def cmd_auth_logout():
    _clear_local_session_store()
    for p in (TOKEN_CONFIG_PATH, PKCE_VERIFIER_PATH, ACCOUNT_PROFILE_CACHE_PATH):
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    print(json.dumps({"ok": True, "logged_in": False, "email": ""}))


@contextlib.contextmanager
def _quiet_colab_cli():
    """将 colab_cli 内部 typer.echo (如 Pruned stale local session) 重定向至 stderr，保持 stdout 纯净 JSON 输出"""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _safe_sync_sessions():
    for attempt in range(3):
        try:
            with _quiet_colab_cli():
                # 重置 colab_cli 进程内缓存，确保每次都重新读取 sessions.json 并对比云端最新分配
                state._sessions = None
                return state.sync_sessions()
        except Exception:
            if attempt == 2:
                try:
                    with _quiet_colab_cli():
                        listed = state.store.list()
                except Exception:
                    listed = []
                if isinstance(listed, dict):
                    cached = dict(listed)
                else:
                    cached = {s.name: s for s in listed if getattr(s, "name", None)}
                return cached, []
            time.sleep(1.2)


def _build_sessions_payload(local_sessions: dict, assignments: list, registry: dict[str, dict] | None = None) -> list[dict]:
    reg = registry if registry is not None else _load_instance_registry()
    name_by_ep = {getattr(s, "endpoint", ""): s.name for s in local_sessions.values() if getattr(s, "endpoint", None)}
    sess_list: list[dict] = []
    for a in assignments:
        ep = getattr(a, "endpoint", "") or ""
        meta = reg.get(ep) or {}
        accel_val = a.accelerator.value if hasattr(a.accelerator, "value") else str(a.accelerator)
        variant_name = a.variant.name if hasattr(a.variant, "name") else str(a.variant)
        shape_name = a.machine_shape.name if hasattr(a.machine_shape, "name") else str(a.machine_shape)
        sess_list.append({
            "name": name_by_ep.get(ep) or str(meta.get("name") or f"colab-{accel_val.lower()}"),
            "endpoint": ep,
            "accelerator": accel_val,
            "variant": variant_name,
            "shape": shape_name,
            "managed": _is_managed_endpoint(ep, local_sessions, reg),
            "server_url": str(meta.get("server_url") or ""),
            "bundle_hash": str(meta.get("bundle_hash") or ""),
        })
    return sess_list


def cmd_auth_status():
    if not _has_valid_token():
        print(json.dumps({"ok": True, "logged_in": False}))
        return

    try:
        info = state.client.get_consumption_user_info()
        local_sessions, assignments = _safe_sync_sessions()
        active_eps = {getattr(a, "endpoint", "") for a in assignments if getattr(a, "endpoint", None)}
        # 仅在云端活跃列表与账号计数器一致或拿到非空列表时修剪本地过期登记
        registry = (
            _prune_instance_registry(active_eps)
            if (assignments or int(getattr(info, "assignments_count", 0) or 0) == 0)
            else _load_instance_registry()
        )
        sess_list = _build_sessions_payload(local_sessions, assignments, registry=registry)

        email = _get_account_email()
        print(json.dumps({
            "ok": True,
            "logged_in": True,
            "email": email,
            "paid_balance": round(float(info.paid_compute_units_balance), 2),
            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
            "active_assignments": int(info.assignments_count),
            "sessions": sess_list,
        }))
    except Exception as e:
        print(json.dumps({
            "ok": False,
            "logged_in": True,
            "email": _get_account_email(),
            "error": str(e),
        }))


DRIVE_RCLONE_CONF_PATH = os.path.expanduser(
    os.environ.get("EASYCOLAB_DRIVE_RCLONE_CONF", "~/.config/easycolab/drive_rclone.conf")
)
_DRIVE_RCLONE_KEYS = ("type", "scope", "token", "client_id", "client_secret", "root_folder_id")


def _main_account_refresh_token() -> str:
    try:
        with open(TOKEN_CONFIG_PATH, "r", encoding="utf-8") as f:
            return str(json.load(f).get("refresh_token") or "")
    except Exception:
        return ""


def _build_rclone_drive_conf() -> tuple[str, str]:
    """
    生成上传到云端 VM 的 rclone Google Drive 配置，返回 (配置文本, 说明)。

    VM 上的第三方节点和依赖包都能读到这份配置，所以绝不使用 Colab 主账号的
    OAuth 令牌（它带 cloud-platform 权限）。只接受用户为独立 Drive 账号单独
    授权、权限为 drive.file 的 rclone 配置，例如：
      rclone config create easycolab_side drive scope=drive.file \
          --config ~/.config/easycolab/drive_rclone.conf
    """
    path = DRIVE_RCLONE_CONF_PATH
    if not os.path.exists(path):
        return "", f"未找到独立 Drive 配置 {path}，跳过 Drive 持久化（主账号令牌不会上传）"
    import configparser

    parser = configparser.ConfigParser(interpolation=None)
    try:
        with open(path, "r", encoding="utf-8") as f:
            parser.read_file(f)
    except Exception as e:
        return "", f"独立 Drive 配置无法解析，跳过 Drive 持久化: {e}"
    sections = parser.sections()
    if len(sections) != 1:
        return "", "独立 Drive 配置必须只包含一个 remote，跳过 Drive 持久化"
    sec = parser[sections[0]]
    if sec.get("type", "").strip() != "drive":
        return "", "独立 Drive 配置的 type 必须为 drive，跳过 Drive 持久化"
    if sec.get("scope", "").strip() != "drive.file":
        return "", "独立 Drive 配置的 scope 必须为 drive.file，跳过 Drive 持久化"
    try:
        token = json.loads(sec.get("token", ""))
    except Exception:
        return "", "独立 Drive 配置缺少有效的 token，跳过 Drive 持久化"
    refresh = str(token.get("refresh_token") or "")
    if not refresh:
        return "", "独立 Drive 配置缺少 refresh_token，跳过 Drive 持久化"
    if refresh == _main_account_refresh_token():
        return "", "独立 Drive 配置与 Colab 主账号使用同一令牌，已拒绝上传"
    lines = ["[easycolab_gdrive]"]
    for key in _DRIVE_RCLONE_KEYS:
        value = sec.get(key, "").strip()
        if value and "\n" not in value:
            lines.append(f"{key} = {value}")
    lines.append("")
    return "\n".join(lines), "Drive 持久化使用独立 Drive 账号（drive.file 权限）"


def _unassign_endpoints_and_wait_quota_release(endpoints: list[str], wait_s: float = 8.0) -> int:
    """
    去重释放指定云端 endpoint 列表，并轮询 list_assignments 等待 Google 配额账本真正注销这些实例。
    避免前脚刚调 unassign、后脚立刻调 assign 时因 Google 后端秒级延迟导致 G4/H100 配额仍显示占用而报 503/412。
    """
    unique_eps = []
    seen = set()
    for ep in endpoints:
        if ep and ep not in seen:
            seen.add(ep)
            unique_eps.append(ep)
    if not unique_eps:
        return 0

    stopped_count = 0
    for ep in unique_eps:
        try:
            state.client.unassign(ep)
            stopped_count += 1
        except Exception:
            pass

    if wait_s > 0:
        deadline = time.time() + wait_s
        target_set = set(unique_eps)
        while time.time() < deadline:
            try:
                remaining = {a.endpoint for a in state.client.list_assignments()}
                if not (remaining & target_set):
                    break
            except Exception:
                break
            time.sleep(1.5)
    return stopped_count


def _graceful_sync_and_stop_session(s: SessionState) -> None:
    try:
        with _quiet_colab_cli():
            rt = ColabRuntime(s.url, s.token, kernel_id=s.kernel_id)
            try:
                rt.execute_code(
                    "import urllib.request\n"
                    "try:\n"
                    "    urllib.request.urlopen(urllib.request.Request("
                    "'http://127.0.0.1:27860/minimax_api/comfyui/sync_drive', "
                    "data=b'{}', headers={'Content-Type':'application/json'}, method='POST'), timeout=18)\n"
                    "except Exception:\n"
                    "    pass\n",
                    allow_stdin=False,
                    timeout=22,
                )
            except Exception:
                pass
            rt.stop(shutdown_kernel=True)
    except Exception:
        pass


def cmd_stop_instance(endpoint: str = "", name: str = ""):
    if not _has_valid_token():
        print(json.dumps({"ok": False, "error": "尚未登录 Google Colab CLI"}))
        return

    try:
        local_sessions, assignments = _safe_sync_sessions()
        target_ep = (endpoint or "").strip()
        target_name = (name or "").strip()

        matched_sessions: list[SessionState] = []
        if target_ep:
            for s in local_sessions.values():
                if getattr(s, "endpoint", None) == target_ep:
                    matched_sessions.append(s)
        elif target_name:
            if target_name in local_sessions:
                matched_sessions.append(local_sessions[target_name])
                target_ep = getattr(local_sessions[target_name], "endpoint", "") or ""
            else:
                reg = _load_instance_registry()
                for ep, meta in reg.items():
                    if meta.get("name") == target_name:
                        target_ep = ep
                        break

        matched_assignment = next(
            (a for a in assignments if getattr(a, "endpoint", None) == target_ep),
            None,
        )

        if not target_ep:
            print(json.dumps({"ok": False, "error": "未指定或未找到要释放的实例 endpoint/name"}))
            return

        if not matched_sessions and matched_assignment is not None:
            matched_sessions.append(
                _session_from_listed_assignment(
                    matched_assignment,
                    session_name=_default_session_name(
                        matched_assignment.accelerator.value,
                        target_ep,
                        local_sessions,
                    ),
                )
            )

        for s in matched_sessions:
            if getattr(s, "url", None) and getattr(s, "token", None):
                _graceful_sync_and_stop_session(s)
            try:
                with _quiet_colab_cli():
                    state.store.remove(s.name)
            except Exception:
                pass

        stopped_count = _unassign_endpoints_and_wait_quota_release([target_ep], wait_s=8.0)
        _remove_instance_registry(target_ep)

        info = state.client.get_consumption_user_info()
        new_local, new_assignments = _safe_sync_sessions()
        sess_list = _build_sessions_payload(new_local, new_assignments)
        print(json.dumps({
            "ok": True,
            "released_endpoint": target_ep,
            "stopped_count": stopped_count,
            "paid_balance": round(float(info.paid_compute_units_balance), 2),
            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
            "active_assignments": int(info.assignments_count),
            "sessions": sess_list,
            "message": f"🛑 已释放实例 {target_ep}，账号当前运行中: {int(info.assignments_count)} 台 ({info.consumption_rate_hourly:.2f} CU/小时)",
        }))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


def cmd_probe_instance(endpoint: str):
    if not _has_valid_token():
        print(json.dumps({"ok": False, "error": "尚未登录 Google Colab CLI"}))
        return

    try:
        ep = (endpoint or "").strip()
        if not ep:
            print(json.dumps({"ok": False, "error": "缺少 endpoint 参数"}))
            return
        local_sessions, assignments = _safe_sync_sessions()
        target_a = next((a for a in assignments if getattr(a, "endpoint", None) == ep), None)
        if target_a is None:
            print(json.dumps({"ok": False, "error": f"云端未找到实例 {ep}"}))
            return

        reg = _load_instance_registry()
        existing_local = next((s for s in local_sessions.values() if getattr(s, "endpoint", None) == ep), None)
        s_name = _default_session_name(target_a.accelerator.value, ep, local_sessions, registry=reg)
        sess = _session_from_listed_assignment(target_a, session_name=s_name, existing=existing_local)
        if not _wait_for_runtime_ready(sess, timeout_s=15):
            print(json.dumps({"ok": False, "endpoint": ep, "error": "实例内核尚未就绪或已失联"}))
            return

        probe_code = r"""
import json, os, re, urllib.request
res = {"managed": False, "running": False, "server_url": "", "bundle_hash": "", "instance_name": ""}
for p in ("/content/.easycolab_managed.json", "/content/.easycolab_runtime.json"):
    if os.path.isfile(p):
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
            if d.get("managed_by") == "easycolab":
                res["managed"] = True
            if d.get("instance_name"):
                res["instance_name"] = str(d.get("instance_name"))
            if d.get("bundle_hash"):
                res["bundle_hash"] = str(d.get("bundle_hash"))
            if d.get("share_url"):
                res["server_url"] = str(d.get("share_url"))
            pid = int(d.get("pid") or 0)
            if pid > 0 and os.path.exists(f"/proc/{pid}"):
                res["running"] = True
        except Exception:
            pass
if not res["server_url"] and os.path.isfile("/content/easycolab_server.log"):
    try:
        txt = open("/content/easycolab_server.log", "r", errors="replace").read()
        m = re.findall(r"^EasyColab URL: (https://[a-zA-Z0-9-]+\.gradio\.live)$", txt, re.M)
        if m:
            res["server_url"] = m[-1]
    except Exception:
        pass
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    opener.open("http://127.0.0.1:7860/", timeout=1.5).close()
    res["running"] = True
except urllib.error.HTTPError as he:
    if he.code in (200, 401):
        res["running"] = True
except Exception:
    pass
print("PROBE_JSON:" + json.dumps(res, ensure_ascii=False))
"""
        rt = ColabRuntime(sess.url, sess.token, kernel_id=sess.kernel_id)
        try:
            outs = rt.execute_code(probe_code, allow_stdin=False, timeout=25)
        finally:
            try:
                rt.stop()
            except Exception:
                pass

        probed = {}
        for out in outs:
            for line in (out.get("text") or "").splitlines():
                if line.startswith("PROBE_JSON:"):
                    probed = json.loads(line[len("PROBE_JSON:"):])

        if probed.get("managed") or probed.get("running") or probed.get("server_url"):
            with _quiet_colab_cli():
                state.store.add(sess)
            _upsert_instance_registry(
                ep,
                name=probed.get("instance_name") or s_name,
                accelerator=target_a.accelerator.value,
                server_url=probed.get("server_url") or "",
                bundle_hash=probed.get("bundle_hash") or "",
                managed=bool(probed.get("managed", True)),
            )

        print(json.dumps({
            "ok": True,
            "endpoint": ep,
            "name": probed.get("instance_name") or s_name,
            "accelerator": target_a.accelerator.value,
            "managed": bool(probed.get("managed")),
            "running": bool(probed.get("running")),
            "server_url": str(probed.get("server_url") or ""),
            "bundle_hash": str(probed.get("bundle_hash") or ""),
        }))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


def cmd_stop_all(managed_only: bool = False):
    if not _has_valid_token():
        print(json.dumps({"ok": False, "error": "尚未登录 Google Colab CLI"}))
        return

    try:
        local_sessions, assignments = _safe_sync_sessions()
        registry = _load_instance_registry()
        endpoints_to_release: list[str] = []
        for s in list(local_sessions.values()):
            ep = getattr(s, "endpoint", None) or ""
            if managed_only and ep and not _is_managed_endpoint(ep, local_sessions, registry):
                continue
            _graceful_sync_and_stop_session(s)
            if ep:
                endpoints_to_release.append(ep)
            with _quiet_colab_cli():
                state.store.remove(s.name)

        for a in assignments:
            ep = getattr(a, "endpoint", None) or ""
            if not ep:
                continue
            if managed_only and not _is_managed_endpoint(ep, local_sessions, registry):
                continue
            endpoints_to_release.append(ep)

        stopped_count = _unassign_endpoints_and_wait_quota_release(endpoints_to_release, wait_s=8.0)
        for ep in endpoints_to_release:
            _remove_instance_registry(ep)

        info = state.client.get_consumption_user_info()
        print(json.dumps({
            "ok": True,
            "stopped_count": stopped_count,
            "paid_balance": round(float(info.paid_compute_units_balance), 2),
            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
            "active_assignments": int(info.assignments_count),
            "message": f"🛑 已彻底释放{'托管' if managed_only else '全部'}云端实例！当前计费速率: {info.consumption_rate_hourly:.2f} CU/小时",
        }))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


def _wait_for_runtime_ready(session: SessionState, timeout_s: int = 45) -> bool:
    """
    等待新分配的 Colab 容器内部 Jupyter Server (/api/kernels) 完成启动。
    规避容器刚启动前几秒出现的 SSLEOFError / ConnectionReset / 502 / 503。
    若检测到 404 (实例如已被销毁)，立即返回 False 以便重新分配新实例。
    """
    base_url = session.url.rstrip("/")
    probe_url = f"{base_url}/api/kernels"
    headers = {
        "Accept": "application/json",
        "X-Colab-Client-Agent": "colab-cli",
        "X-Colab-Runtime-Proxy-Token": session.token,
        "Authorization": f"Bearer {session.token}",
    }
    params = {"authuser": "0", "colab-runtime-proxy-token": session.token}
    deadline = time.time() + timeout_s
    not_found_count = 0

    while time.time() < deadline:
        try:
            r = requests.get(probe_url, headers=headers, params=params, timeout=8)
            if r.status_code == 200:
                return True
            if r.status_code in (404, 410):
                not_found_count += 1
                if not_found_count >= 2:
                    return False
        except requests.exceptions.RequestException:
            pass
        time.sleep(2.0)
    return False


GPU_ACCEL_MAP = {
    "h100": Accelerator.H100,
    "g4": Accelerator.G4,
    "a100": Accelerator.A100,
    "l4": Accelerator.L4,
    "t4": Accelerator.T4,
}


def _resolve_gpu_accel(gpu_type: str) -> Accelerator:
    return GPU_ACCEL_MAP.get((gpu_type or "g4").strip().lower(), Accelerator.G4)


def _gpu_display_label(accel: Accelerator) -> str:
    if accel == Accelerator.G4:
        return "G4 (RTX PRO 6000 · 96GB)"
    return accel.value


def _candidate_shapes_for_accel(accel: Accelerator) -> list:
    """
    为各型号 GPU 提供主备机器内存规格 (Shape) 候选序列：
    - G4 / H100：首选不传 shape (默认大内存标准池，避免部分区域对 shape=hm 报 400)，若遇 503 库存不足自动尝试 HIGH_RAM 池；
    - A100 / T4：首选 HIGH_RAM 大内存池，若遇 400/503 自动降级尝试标准内存池；
    - L4：官方固定单规格 (HIGH_MEM_ONLY_ACCELERATORS)，仅使用 None。
    """
    if accel == Accelerator.L4:
        return [None]
    if accel in (Accelerator.G4, Accelerator.H100):
        return [None, Shape.HIGH_RAM]
    primary = resolve_assign_shape(accel, high_mem=True)
    return [primary, None] if primary is not None else [None]


def _session_from_listed_assignment(a, session_name: str = "easycolab", existing: SessionState | None = None) -> SessionState:
    return SessionState(
        name=existing.name if existing and getattr(existing, "name", None) else session_name,
        token=a.runtime_proxy_info.token,
        url=a.runtime_proxy_info.url,
        endpoint=a.endpoint,
        token_expires_at=a.runtime_proxy_info.expires_at(),
        kernel_id=getattr(existing, "kernel_id", None) if existing and existing.endpoint == a.endpoint else None,
        session_id=(
            getattr(existing, "session_id", None)
            if existing and existing.endpoint == a.endpoint and getattr(existing, "session_id", None)
            else uuid.uuid4().hex
        ),
        variant=Variant.GPU.value,
        accelerator=a.accelerator.value,
        machine_shape=Shape.HIGH_RAM.name if a.machine_shape == Shape.HIGH_RAM else Shape.STANDARD.name,
    )


def _try_adopt_matching_cloud_assignment(
    accel: Accelerator,
    session_name: str = "easycolab",
    exclude_endpoints: set[str] | None = None,
) -> SessionState | None:
    """
    当 POST /tun/m/assign 出现网关 504 / ReadTimeout / 503 重试时，立即核对云端真实 list_assignments()：
    若 Google 后端实际上已在后台完成该型号机器分配（且不在已排除的现有实例集合内），直接纳管返回，杜绝产生孤儿机器占满名额。
    """
    excluded = exclude_endpoints or set()
    try:
        assignments = state.client.list_assignments()
    except Exception:
        return None
    for a in assignments:
        ep = getattr(a, "endpoint", "") or ""
        if ep in excluded:
            continue
        if a.accelerator == accel:
            s = _session_from_listed_assignment(a, session_name=session_name)
            with _quiet_colab_cli():
                state.store.add(s)
            _upsert_instance_registry(ep, name=s.name, accelerator=accel.value, managed=True)
            return s
    return None


@contextlib.contextmanager
def _single_shot_colab_assign():
    """禁用 colab_cli 内部静默阻塞 70 秒的英文重试循环，交由 EasyColab 透明重试并实时对账云端分配"""
    prev_retries = os.environ.get("COLAB_CLI_ASSIGN_RETRIES")
    os.environ["COLAB_CLI_ASSIGN_RETRIES"] = "0"
    try:
        yield
    finally:
        if prev_retries is None:
            os.environ.pop("COLAB_CLI_ASSIGN_RETRIES", None)
        else:
            os.environ["COLAB_CLI_ASSIGN_RETRIES"] = prev_retries


def _allocate_fresh_session(
    session_name: str,
    gpu_type: str,
    allow_reclaim_on_412: bool = False,
    exclude_endpoints: set[str] | None = None,
    managed_endpoints: set[str] | None = None,
) -> SessionState:
    accel = _resolve_gpu_accel(gpu_type)
    gpu_desc = _gpu_display_label(accel)
    shape_candidates = list(_candidate_shapes_for_accel(accel))
    rejected_400_shapes = set()
    nbh_by_shape = {sh: uuid.uuid4() for sh in shape_candidates}
    excluded = set(exclude_endpoints or ())

    print(f"WAKE_LOG: 🚀 正在向 Google Colab 申请分配 {gpu_desc} 实例...", flush=True)

    max_attempts = 3
    retry_delays = [6.0, 10.0]
    attempt = 0
    shape_idx = 0
    cleaned_412_orphans = False
    last_err_tag = "HTTP 503"
    res = None
    chosen_shape = shape_candidates[0]

    while attempt < max_attempts:
        valid_shapes = [sh for sh in shape_candidates if sh not in rejected_400_shapes] or [None]
        chosen_shape = valid_shapes[shape_idx % len(valid_shapes)]
        nbh = nbh_by_shape.setdefault(chosen_shape, uuid.uuid4())

        try:
            with _quiet_colab_cli(), _single_shot_colab_assign():
                res = state.client.assign(nbh, variant=Variant.GPU, accelerator=accel, shape=chosen_shape)
            break
        except TooManyAssignmentsError as quota_err:
            adopted = _try_adopt_matching_cloud_assignment(
                accel, session_name=session_name, exclude_endpoints=excluded
            )
            if adopted is not None:
                print(f"WAKE_LOG: ✅ 检测到云端已在后台完成 {accel.value} 实例分配，直接接管！", flush=True)
                return adopted
            try:
                all_active = [a.endpoint for a in state.client.list_assignments() if getattr(a, "endpoint", None)]
            except Exception:
                all_active = []
            if allow_reclaim_on_412 and not cleaned_412_orphans:
                cleaned_412_orphans = True
                reclaim_eps = (
                    [ep for ep in all_active if ep in managed_endpoints]
                    if managed_endpoints is not None
                    else all_active
                )
                if reclaim_eps:
                    print("WAKE_LOG: 🧹 检测到托管旧实例占用并发名额，正在自动释放后重试...", flush=True)
                    _unassign_endpoints_and_wait_quota_release(reclaim_eps, wait_s=8.0)
                    for ep in reclaim_eps:
                        _remove_instance_registry(ep)
                        excluded.discard(ep)
                    continue
            active_hint = f"（当前账号已有 {len(all_active)} 台运行中）" if all_active else ""
            raise RuntimeError(
                f"Google Colab 拒绝分配 {gpu_desc} (HTTP 412)：当前账号并发实例数已达上限{active_hint}，"
                f"为防止误杀其他运行中实例或浏览器 Notebook，请先在实例列表释放闲置实例后再新建，或切换机型重试。"
            ) from quota_err
        except (ColabRequestError, requests.exceptions.RequestException) as req_err:
            status = get_status_code(req_err) if isinstance(req_err, ColabRequestError) else None

            # 1) 无论报 504/503/ReadTimeout，第一时间检查云端是否其实已经分配成功（排除已存在的旧实例）
            adopted = _try_adopt_matching_cloud_assignment(
                accel, session_name=session_name, exclude_endpoints=excluded
            )
            if adopted is not None:
                print(f"WAKE_LOG: ✅ 检测到云端已在后台完成 {accel.value} 实例分配，直接接管！", flush=True)
                return adopted

            # 2) 若当前机器规格报 400 Bad Request（如部分池不支持 shape=hm），立即切换候选规格且不消耗重试次数
            if status == 400 and chosen_shape is not None and None not in rejected_400_shapes:
                rejected_400_shapes.add(chosen_shape)
                shape_idx = 0
                continue
            if status == 400:
                raise RuntimeError(
                    f"Google Colab 后端拒绝分配 {gpu_desc} (HTTP 400)：当前账号订阅等级可能暂不支持该机型或可用计算单元不足，"
                    f"建议切换为 A100 / L4 / T4 实例。"
                ) from req_err

            retryable = isinstance(req_err, requests.exceptions.RequestException) or status in (429, 500, 502, 503, 504)
            if not retryable:
                raise

            last_err_tag = f"HTTP {status}" if status else type(req_err).__name__
            attempt += 1
            shape_idx += 1
            if attempt >= max_attempts:
                break

            delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
            print(
                f"WAKE_LOG: ⏳ Google Colab 当前区域 {gpu_desc} 算力池繁忙 ({last_err_tag})，"
                f"{delay:.0f} 秒后进行第 {attempt + 1}/{max_attempts} 次申请...",
                flush=True,
            )
            time.sleep(delay)
            adopted_after_wait = _try_adopt_matching_cloud_assignment(
                accel, session_name=session_name, exclude_endpoints=excluded
            )
            if adopted_after_wait is not None:
                print(f"WAKE_LOG: ✅ 检测到云端已完成 {accel.value} 实例分配，直接接管！", flush=True)
                return adopted_after_wait

    if res is None:
        raise RuntimeError(
            f"Google Colab 云端当前时段 {gpu_desc} 算力池库存不足 ({last_err_tag})，连续 {max_attempts} 次申请均未分到机器"
            f"（属于 Colab 官方高配机型时段性缺货，非本地客户端故障）。建议稍等几分钟再试，或在左侧临时切换为 A100 / L4 实例。"
        )

    if isinstance(res, PostAssignmentResponse):
        token = res.runtime_proxy_info.token
        url = res.runtime_proxy_info.url
        endpoint = res.endpoint
        exp = res.runtime_proxy_info.expires_at()
    else:
        token = res.runtime_proxy_info.token if hasattr(res, "runtime_proxy_info") else getattr(res, "runtime_proxy_token", "")
        url = res.runtime_proxy_info.url if hasattr(res, "runtime_proxy_info") else ""
        endpoint = res.endpoint
        exp = res.runtime_proxy_info.expires_at() if hasattr(res, "runtime_proxy_info") else None

    s = SessionState(
        name=session_name,
        token=token,
        url=url,
        endpoint=endpoint,
        token_expires_at=exp,
        session_id=uuid.uuid4().hex,
        variant=Variant.GPU.value,
        accelerator=accel.value,
        machine_shape=Shape.HIGH_RAM.name if chosen_shape == Shape.HIGH_RAM else Shape.STANDARD.name,
    )
    with _quiet_colab_cli():
        state.store.add(s)
    _upsert_instance_registry(endpoint, name=session_name, accelerator=accel.value, managed=True)
    print(f"WAKE_LOG: ✅ 成功分配 {accel.value} 云端虚拟机 ({endpoint[:14]})！正在等待容器内核就绪...", flush=True)
    return s


def cmd_wake(
    gpu_type: str,
    app_py_path: str,
    endpoint: str | None = None,
    new_instance: bool = False,
    instance_name: str | None = None,
    replace_existing: bool = False,
    force_redeploy: bool = False,
):
    """
    无浏览器一键唤醒 Colab GPU 并自动部署 EasyColab 多模型服务端（支持多实例共存与安全覆盖热重载）：
    1. 多实例调度：
       - 若指定 --endpoint：精确重部署/接管该云端实例，绝不影响账号下其他运行中实例；
       - 若指定 --new：强制申请一台全新 GPU 实例并存，不复用也不释放现有实例；
       - 默认模式：优先复用同型号健康托管实例；不误杀其他型号或用户手动开启的浏览器 Notebook（除非显式 --replace-existing）。
    2. 安全覆盖与热重载：
       - 先比对代码包哈希与管理口令指纹，未变更且服务健康时跳过重启，保留显存与运行中任务；
       - 确需重启时，先将代码上传至暂存文件，优雅停止旧 app.py（兼容 Notebook 相对路径与桥接绝对路径）并等待 7860/8188/8190 端口释放后，再原子替换上线。
    """
    if not _has_valid_token():
        print("WAKE_ERROR: 尚未授权登录 Google 账号，请先点击「🔑 绑定账号」", flush=True)
        return

    import getpass
    password = os.environ.pop("EASYCOLAB_ADMIN_PASSWORD", "")
    initial_api_tokens = ",".join(
        tok
        for tok in (
            os.environ.pop("EASYCOLAB_API_TOKENS", "").strip(),
            os.environ.pop("EASYCOLAB_API_KEY", "").strip(),
        )
        if tok
    )
    raw_idle_min = os.environ.get("EASYCOLAB_IDLE_TIMEOUT_MINUTES", "").strip()
    try:
        initial_idle_minutes = int(raw_idle_min) if raw_idle_min != "" else None
    except ValueError:
        initial_idle_minutes = None
    if not password and sys.stdin.isatty():
        password = getpass.getpass("Management password (24+ URL-safe characters): ")
    if not password:
        print("WAKE_ERROR: Set a management password before deploying", flush=True)
        return

    try:
        requested_accel = _resolve_gpu_accel(gpu_type)
        print("WAKE_LOG: 🔍 正在检查账号下已有的 Colab 实例...", flush=True)
        local_sessions, assignments = _safe_sync_sessions()
        registry = _load_instance_registry()
        ep_to_local = {
            s.endpoint: s for s in local_sessions.values() if getattr(s, "endpoint", None)
        }
        pre_existing_eps = {getattr(a, "endpoint", "") for a in assignments if getattr(a, "endpoint", None)}
        managed_eps = {
            ep for ep in pre_existing_eps if _is_managed_endpoint(ep, local_sessions, registry)
        }

        target_session: SessionState | None = None
        reused_existing_vm = False
        req_ep = (endpoint or "").strip()

        if req_ep:
            matched_a = next((a for a in assignments if getattr(a, "endpoint", None) == req_ep), None)
            if matched_a is None:
                print(f"WAKE_ERROR: 未在云端找到指定实例 ({req_ep})，可能已释放或过期", flush=True)
                return
            existing_local = ep_to_local.get(req_ep)
            s_name = _default_session_name(
                matched_a.accelerator.value,
                req_ep,
                local_sessions,
                requested_name=instance_name or "",
                registry=registry,
            )
            candidate = _session_from_listed_assignment(matched_a, session_name=s_name, existing=existing_local)
            print(
                f"WAKE_LOG: ♻️ 正在检查指定实例 [{candidate.name}] ({matched_a.accelerator.value}) 连通性...",
                flush=True,
            )
            if not _wait_for_runtime_ready(candidate, timeout_s=20):
                print(f"WAKE_ERROR: 指定实例 ({req_ep}) 已失效或无法连接内核", flush=True)
                return
            target_session = candidate
            reused_existing_vm = True
            with _quiet_colab_cli():
                state.store.add(target_session)
            _upsert_instance_registry(
                target_session.endpoint,
                name=target_session.name,
                accelerator=matched_a.accelerator.value,
                managed=True,
            )
        elif new_instance:
            s_name = _default_session_name(
                requested_accel.value,
                "",
                local_sessions,
                requested_name=instance_name or "",
                registry=registry,
            )
            target_session = _allocate_fresh_session(
                s_name,
                gpu_type,
                allow_reclaim_on_412=replace_existing,
                exclude_endpoints=pre_existing_eps,
                managed_endpoints=managed_eps,
            )
            if not _wait_for_runtime_ready(target_session, timeout_s=50):
                print("WAKE_LOG: ⏳ 容器仍在初始化，继续尝试连接内核...", flush=True)
        else:
            matching_assignments = [a for a in assignments if a.accelerator == requested_accel]
            # 优先复用由 EasyColab 托管的同型号实例，其次才考虑账号下其他同型号实例
            matching_assignments.sort(
                key=lambda a: (0 if getattr(a, "endpoint", "") in managed_eps else 1)
            )
            dead_matching_eps: list[str] = []

            for a in matching_assignments:
                existing_local = ep_to_local.get(a.endpoint)
                s_name = _default_session_name(
                    a.accelerator.value,
                    a.endpoint,
                    local_sessions,
                    requested_name=instance_name or "",
                    registry=registry,
                )
                candidate = _session_from_listed_assignment(a, session_name=s_name, existing=existing_local)
                print(
                    f"WAKE_LOG: ♻️ 正在检查已有实例 [{candidate.name}] ({a.accelerator.value}) 连通性...",
                    flush=True,
                )
                if _wait_for_runtime_ready(candidate, timeout_s=18):
                    target_session = candidate
                    reused_existing_vm = True
                    with _quiet_colab_cli():
                        state.store.add(target_session)
                    _upsert_instance_registry(
                        target_session.endpoint,
                        name=target_session.name,
                        accelerator=a.accelerator.value,
                        managed=True,
                    )
                    break
                else:
                    print(f"WAKE_LOG: ⚠️ 旧实例 ({a.endpoint[:18]}...) 已失效或正在销毁...", flush=True)
                    if a.endpoint in managed_eps:
                        dead_matching_eps.append(a.endpoint)

            if target_session is not None:
                extra_eps = list(dead_matching_eps)
                if replace_existing:
                    # 仅在显式开启 replace_existing 时才释放其余 EasyColab 托管实例，绝不误杀用户浏览器 Notebook
                    for ep in managed_eps:
                        if ep != target_session.endpoint and ep not in extra_eps:
                            extra_eps.append(ep)
                if extra_eps:
                    for ep in extra_eps:
                        for s in list(local_sessions.values()):
                            if getattr(s, "endpoint", None) == ep and s.name != target_session.name:
                                with _quiet_colab_cli():
                                    state.prune_session(s.name)
                        _remove_instance_registry(ep)
                    _unassign_endpoints_and_wait_quota_release(extra_eps, wait_s=5.0)
            else:
                release_eps = list(dead_matching_eps)
                if replace_existing:
                    for ep in managed_eps:
                        if ep not in release_eps:
                            release_eps.append(ep)
                if release_eps:
                    for ep in release_eps:
                        for s in list(local_sessions.values()):
                            if getattr(s, "endpoint", None) == ep:
                                with _quiet_colab_cli():
                                    state.prune_session(s.name)
                        _remove_instance_registry(ep)
                    print(
                        f"WAKE_LOG: 🔄 正在释放旧托管实例并准备启动 [{requested_accel.value}]...",
                        flush=True,
                    )
                    _unassign_endpoints_and_wait_quota_release(release_eps, wait_s=8.0)

                s_name = _default_session_name(
                    requested_accel.value,
                    "",
                    local_sessions,
                    requested_name=instance_name or "",
                    registry=registry,
                )
                target_session = _allocate_fresh_session(
                    s_name,
                    gpu_type,
                    allow_reclaim_on_412=replace_existing,
                    exclude_endpoints=pre_existing_eps - set(release_eps),
                    managed_endpoints=managed_eps - set(release_eps),
                )
                if not _wait_for_runtime_ready(target_session, timeout_s=50):
                    print("WAKE_LOG: ⏳ 容器仍在初始化，继续尝试连接内核...", flush=True)

        print("WAKE_LOG: 正在校验与暂存服务端代码 (app.py + easycolab_server/*)...", flush=True)
        contents = ContentsClient(target_session)
        app_base_dir = os.path.dirname(os.path.abspath(app_py_path))
        srv_pkg_dir = os.path.join(app_base_dir, "easycolab_server")
        srv_bundle: Dict[str, str] = {}
        if os.path.isdir(srv_pkg_dir):
            for fname in sorted(os.listdir(srv_pkg_dir)):
                if fname.endswith(".py") and os.path.isfile(os.path.join(srv_pkg_dir, fname)):
                    with open(os.path.join(srv_pkg_dir, fname), "r", encoding="utf-8") as mf:
                        srv_bundle[fname] = mf.read()

        bundle_hash = _compute_server_bundle_hash(app_py_path, srv_bundle)
        pw_sha256 = hashlib.sha256(password.encode("utf-8")).hexdigest()

        for up_attempt in range(4):
            try:
                contents.upload(app_py_path, "content/.easycolab_stage_app.py")
                contents.upload(os.path.join(app_base_dir, "runtime_auth.py"), "content/.easycolab_stage_runtime_auth.py")
                if srv_bundle:
                    tmp_bnd_fd, tmp_bnd_path = tempfile.mkstemp(prefix="easycolab_srv_", suffix=".json")
                    try:
                        with os.fdopen(tmp_bnd_fd, "w", encoding="utf-8") as bf:
                            json.dump(srv_bundle, bf, ensure_ascii=False)
                        contents.upload(tmp_bnd_path, "content/.easycolab_stage_server_bundle.json")
                    finally:
                        if os.path.exists(tmp_bnd_path):
                            try:
                                os.remove(tmp_bnd_path)
                            except Exception:
                                pass
                break
            except Exception as up_err:
                if up_attempt == 3:
                    raise up_err
                time.sleep(2.5)
        rclone_conf_text, drive_note = _build_rclone_drive_conf()
        print(f"WAKE_LOG: {drive_note}", flush=True)
        if rclone_conf_text:
            tmp_rc_fd, tmp_rc_path = tempfile.mkstemp(prefix="easycolab_rc_", suffix=".conf")
            try:
                with os.fdopen(tmp_rc_fd, "w", encoding="utf-8") as rcf:
                    rcf.write(rclone_conf_text)
                os.chmod(tmp_rc_path, 0o600)
                contents.upload(tmp_rc_path, "content/.easycolab_stage_rclone.conf")
            except Exception:
                pass
            finally:
                if os.path.exists(tmp_rc_path):
                    try:
                        os.remove(tmp_rc_path)
                    except Exception:
                        pass

        print("WAKE_LOG: 正在云端配置 Google Drive 持久化、多模型引擎并启动隧道 (约需 40~90 秒)...", flush=True)
        boot_code = r"""
import os, sys, time, re, json, signal, socket, shutil, subprocess, urllib.request, urllib.error

with open(ADMIN_STAGE, encoding="ascii") as stream:
    admin_password = stream.read()
os.unlink(ADMIN_STAGE)

api_tokens = ""
if API_TOKEN_STAGE and os.path.exists(API_TOKEN_STAGE):
    with open(API_TOKEN_STAGE, encoding="utf-8") as stream:
        api_tokens = stream.read().strip()
    os.unlink(API_TOKEN_STAGE)

def _cleanup_staged_files():
    for _sp in (
        "/content/.easycolab_stage_app.py",
        "/content/.easycolab_stage_runtime_auth.py",
        "/content/.easycolab_stage_server_bundle.json",
        "/content/.easycolab_stage_rclone.conf",
    ):
        if os.path.exists(_sp):
            try:
                os.remove(_sp)
            except Exception:
                pass

def _port_open(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.35)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        try:
            s.close()
        except Exception:
            pass

def _http_ingress_alive() -> bool:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        opener.open("http://127.0.0.1:7860/", timeout=1.5).close()
        return True
    except urllib.error.HTTPError as he:
        return he.code in (200, 401)
    except Exception:
        return False

# 0. 热复用判定：若代码指纹与密码指纹一致、且当前服务在线健康，则跳过重杀，保留显存与运行中任务
_rt_meta_file = "/content/.easycolab_runtime.json"
_managed_marker_file = "/content/.easycolab_managed.json"
if (not FORCE_REDEPLOY) and os.path.isfile(_rt_meta_file):
    try:
        _rt = json.load(open(_rt_meta_file, "r", encoding="utf-8"))
        _old_pid = int(_rt.get("pid") or 0)
        _old_url = str(_rt.get("share_url") or "").strip()
        if (
            _rt.get("bundle_hash") == BUNDLE_HASH
            and _rt.get("password_sha256") == PASSWORD_SHA256
            and _old_pid > 0
            and os.path.exists(f"/proc/{_old_pid}")
            and _old_url.startswith("https://")
            and _http_ingress_alive()
        ):
            _cleanup_staged_files()
            with open(_managed_marker_file, "w", encoding="utf-8") as _mf:
                json.dump({"managed_by": "easycolab", "instance_name": INSTANCE_NAME, "updated_at": time.time()}, _mf, ensure_ascii=False)
            print("WAKE_LOG: ⚡ 服务端代码与密码未变更且实例在线，已无缝热复用当前实例（保留显存与任务）！", flush=True)
            print("WAKE_SKIPPED_RELOAD:1", flush=True)
            print("WAKE_URL:" + _old_url, flush=True)
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass

if not shutil.which("nginx"):
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "nginx-light"], check=True)
    # apt may start the distribution's default site.
    subprocess.run(["service", "nginx", "stop"], check=False)

# 1. 先优雅停止旧版 app.py 并等待 7860/8188/8190 端口彻底释放（先停旧进程，再原子替换代码，规避竞态）
def _find_matching_app_pids() -> list[int]:
    pids = []
    try:
        if os.path.isfile(_rt_meta_file):
            _old_pid = int(json.load(open(_rt_meta_file, "r", encoding="utf-8")).get("pid") or 0)
            if _old_pid > 0 and os.path.exists(f"/proc/{_old_pid}"):
                pids.append(_old_pid)
    except Exception:
        pass
    for pat in (r"python.*(/content/)?app\.py",):
        try:
            out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True, check=False).stdout
            for ln in out.splitlines():
                if ln.strip().isdigit():
                    p = int(ln.strip())
                    if p != os.getpid() and p not in pids:
                        pids.append(p)
        except Exception:
            pass
    return pids

_old_app_pids = _find_matching_app_pids()
for _p in _old_app_pids:
    try:
        os.kill(_p, signal.SIGTERM)
    except Exception:
        pass
subprocess.run(["pkill", "-TERM", "-f", "llama-server"], check=False)

# 等待旧进程退出及 nginx 入口端口释放（最多 8 秒）
_wait_deadline = time.time() + 8.0
while time.time() < _wait_deadline:
    _alive = [p for p in _old_app_pids if os.path.exists(f"/proc/{p}")]
    _busy_ports = [pt for pt in (7860, 8188, 8190) if _port_open(pt)]
    if not _alive and not _busy_ports:
        break
    if time.time() > (_wait_deadline - 3.0):
        # 超时强杀 stubborn 旧 app.py 与残留的 easycolab-auth nginx
        for _p in _alive:
            try:
                os.kill(_p, signal.SIGKILL)
            except Exception:
                pass
        subprocess.run(["pkill", "-KILL", "-f", "easycolab-auth-"], check=False)
    time.sleep(0.35)

# 2. 原子替换本地上传到暂存区的新服务端代码
if os.path.exists("/content/.easycolab_stage_app.py"):
    shutil.move("/content/.easycolab_stage_app.py", "/content/app.py")
if os.path.exists("/content/.easycolab_stage_runtime_auth.py"):
    shutil.move("/content/.easycolab_stage_runtime_auth.py", "/content/runtime_auth.py")

srv_bundle_path = "/content/.easycolab_stage_server_bundle.json"
if os.path.exists(srv_bundle_path):
    with open(srv_bundle_path, "r", encoding="utf-8") as _bf:
        _srv_files = json.load(_bf)
    _pkg_next = "/content/.easycolab_server_next"
    _pkg_dst = "/content/easycolab_server"
    shutil.rmtree(_pkg_next, ignore_errors=True)
    os.makedirs(_pkg_next, exist_ok=True)
    for _fn, _content in _srv_files.items():
        _safe_fn = os.path.basename(_fn)
        if _safe_fn.endswith(".py"):
            with open(os.path.join(_pkg_next, _safe_fn), "w", encoding="utf-8") as _mf:
                _mf.write(_content)
    shutil.rmtree(_pkg_dst, ignore_errors=True)
    os.replace(_pkg_next, _pkg_dst)
    os.unlink(srv_bundle_path)

with open(_managed_marker_file, "w", encoding="utf-8") as _mf:
    json.dump({"managed_by": "easycolab", "instance_name": INSTANCE_NAME, "updated_at": time.time()}, _mf, ensure_ascii=False)

# 3. 安全配置 Google Drive 自动挂载/高速同步（仅清理先前由 EasyColab 写入的托管配置，不误删用户自有 rclone.conf）
rc_stage = "/content/.easycolab_stage_rclone.conf"
rc_dir = os.path.expanduser("~/.config/rclone")
rc_dst = os.path.join(rc_dir, "rclone.conf")
rc_managed_flag = "/content/.easycolab_rclone_managed"
if os.path.exists(rc_stage):
    os.makedirs(rc_dir, mode=0o700, exist_ok=True)
    shutil.move(rc_stage, rc_dst)
    os.chmod(rc_dst, 0o600)
    with open(rc_managed_flag, "w", encoding="utf-8") as _rf:
        _rf.write("1")
elif os.path.exists(rc_dst) and os.path.exists(rc_managed_flag):
    subprocess.run(["pkill", "-f", "rclone mount easycolab_gdrive"], check=False)
    os.remove(rc_dst)
    try:
        os.remove(rc_managed_flag)
    except Exception:
        pass

if os.path.exists(os.path.expanduser("~/.config/rclone/rclone.conf")):
    if not shutil.which("rclone"):
        subprocess.run(
            ["bash", "-lc", "curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip && unzip -q -o /tmp/rclone.zip -d /tmp && cp /tmp/rclone-*-linux-amd64/rclone /usr/local/bin/rclone && chmod +x /usr/local/bin/rclone && rm -rf /tmp/rclone*"],
            check=False,
        )
    if shutil.which("rclone") and not os.path.ismount("/content/drive/MyDrive"):
        subprocess.run(["rclone", "mkdir", "easycolab_gdrive:EasyColab_Drive"], check=False)
        if os.path.exists("/dev/fuse") and (shutil.which("fusermount3") or shutil.which("fusermount")):
            os.makedirs("/content/drive/MyDrive", exist_ok=True)
            subprocess.Popen(
                [
                    "rclone", "mount", "easycolab_gdrive:EasyColab_Drive", "/content/drive/MyDrive",
                    "--vfs-cache-mode", "writes",
                    "--vfs-write-back", "3s",
                    "--dir-cache-time", "15s",
                    "--drive-chunk-size", "64M",
                    "--buffer-size", "32M",
                    "--allow-non-empty",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(10):
                if os.path.ismount("/content/drive/MyDrive"):
                    print("✅ 已通过 Google OAuth 自动挂载 Google Drive (/content/drive/MyDrive)！", flush=True)
                    break
                time.sleep(0.5)
            else:
                try:
                    if not os.listdir("/content/drive/MyDrive"):
                        os.rmdir("/content/drive/MyDrive")
                except Exception:
                    pass

# 4. 如果尚未安装 DiffSynth-Studio，自动快速安装 (含最新 transformers 以支持 Qwen3.8-27B)
if not os.path.exists("/content/DiffSynth-Studio"):
    print("正在克隆 DiffSynth-Studio...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/modelscope/DiffSynth-Studio.git", "/content/DiffSynth-Studio"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes", "huggingface_hub", "hf_transfer", "qwen-vl-utils", "einops", "kernels>=0.17.2,<0.18.0", "av", "gradio", "modelscope", "imageio", "imageio-ffmpeg", "scipy", "soundfile", "librosa"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "fla-core", "flash-linear-attention"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--only-binary=:all:", "causal-conv1d"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/DiffSynth-Studio"], check=True)
else:
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        import kernels  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "kernels>=0.17.2,<0.18.0"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", "fla-core", "flash-linear-attention"], check=False)

# 5. 后台启动 /content/app.py 并等待 gradio.live 链接生成
log_path = "/content/easycolab_server.log"
if os.path.exists(log_path):
    os.remove(log_path)

with open(log_path, "w") as out_f:
    child_env = {
        **os.environ,
        "EASYCOLAB_ADMIN_PASSWORD": admin_password,
        "EASYCOLAB_INSTANCE_NAME": INSTANCE_NAME,
        "EASYCOLAB_BUNDLE_HASH": BUNDLE_HASH,
    }
    if api_tokens:
        child_env["EASYCOLAB_API_TOKENS"] = api_tokens
    if INITIAL_IDLE_MINUTES is not None:
        child_env["EASYCOLAB_IDLE_TIMEOUT_MINUTES"] = str(INITIAL_IDLE_MINUTES)
    subprocess.Popen([sys.executable, "-u", "/content/app.py"], cwd="/content", env=child_env, stdout=out_f, stderr=subprocess.STDOUT)

url_re = re.compile(r"^EasyColab URL: (https://[a-zA-Z0-9-]+\.gradio\.live)$", re.MULTILINE)
found_url = None
for _ in range(90):
    time.sleep(2)
    if os.path.exists(log_path):
        txt = open(log_path, "r", errors="replace").read()
        m = url_re.search(txt)
        if m:
            found_url = m.group(1)
            break

if found_url:
    print("WAKE_URL:" + found_url, flush=True)
else:
    tail = open(log_path, "r", errors="replace").read()[-600:] if os.path.exists(log_path) else "no log"
    print("WAKE_ERROR: 启动超时，日志尾部: " + tail, flush=True)
"""

        def _on_kid(kid):
            target_session.kernel_id = kid
            state.store.add(target_session)

        def _on_sid(sid):
            target_session.session_id = sid
            state.store.add(target_session)

        def _run_boot_on_runtime(use_saved_kernel: bool):
            rt = ColabRuntime(
                target_session.url,
                target_session.token,
                session_name=target_session.name,
                history=state.history,
                kernel_id=target_session.kernel_id if use_saved_kernel else None,
                session_id=target_session.session_id if use_saved_kernel else None,
                on_kernel_started=_on_kid,
                on_session_started=_on_sid,
            )
            try:
                stage = ".easycolab-management-" + uuid.uuid4().hex
                token_stage = (".easycolab-apitokens-" + uuid.uuid4().hex) if initial_api_tokens else None
                with tempfile.TemporaryDirectory() as directory:
                    password_file = os.path.join(directory, "management")
                    with open(password_file, "w", encoding="ascii") as stream:
                        stream.write(password)
                    contents.upload(password_file, "content/" + stage)
                    if token_stage:
                        token_file = os.path.join(directory, "apitokens")
                        with open(token_file, "w", encoding="utf-8") as stream:
                            stream.write(initial_api_tokens)
                        contents.upload(token_file, "content/" + token_stage)
                code = (
                    "ADMIN_STAGE = " + repr("/content/" + stage) + "\n"
                    + "API_TOKEN_STAGE = " + repr(("/content/" + token_stage) if token_stage else None) + "\n"
                    + "INITIAL_IDLE_MINUTES = " + repr(initial_idle_minutes) + "\n"
                    + "INSTANCE_NAME = " + repr(target_session.name) + "\n"
                    + "BUNDLE_HASH = " + repr(bundle_hash) + "\n"
                    + "PASSWORD_SHA256 = " + repr(pw_sha256) + "\n"
                    + "FORCE_REDEPLOY = " + repr(bool(force_redeploy)) + "\n"
                    + boot_code
                )
                return rt.execute_code(code, allow_stdin=False, timeout=300)
            finally:
                try:
                    rt.stop()
                except Exception:
                    pass

        try:
            outputs = _run_boot_on_runtime(use_saved_kernel=True)
        except Exception:
            target_session.kernel_id = None
            target_session.session_id = None
            outputs = _run_boot_on_runtime(use_saved_kernel=False)

        wake_url_found = ""
        skipped_reload = False
        for out in outputs:
            if "text" in out:
                for line in out["text"].splitlines():
                    if line.startswith("WAKE_URL:"):
                        wake_url_found = line[len("WAKE_URL:"):].strip()
                    elif line.strip() == "WAKE_SKIPPED_RELOAD:1":
                        skipped_reload = True
                        continue
                    print(line, flush=True)
            elif out.get("output_type") == "error":
                print("WAKE_ERROR: " + out.get("evalue", "Execution error"), flush=True)

        if wake_url_found:
            _upsert_instance_registry(
                target_session.endpoint,
                name=target_session.name,
                accelerator=getattr(target_session, "accelerator", "") or requested_accel.value,
                server_url=wake_url_found,
                bundle_hash=bundle_hash,
                managed=True,
            )
            print(
                "WAKE_INSTANCE:" + json.dumps({
                    "endpoint": target_session.endpoint,
                    "name": target_session.name,
                    "accelerator": getattr(target_session, "accelerator", "") or requested_accel.value,
                    "server_url": wake_url_found,
                    "reused": reused_existing_vm,
                    "skipped_reload": skipped_reload,
                    "bundle_hash": bundle_hash,
                }, ensure_ascii=False),
                flush=True,
            )

    except Exception as e:
        print(f"WAKE_ERROR: {str(e)}", flush=True)


class _ProxyAuthExpiredError(RuntimeError):
    """Runtime Proxy Token 已过期或失效 (HTTP 401/403)，需立即通过 list_assignments 刷新"""


def _session_token_ttl_seconds(s: SessionState) -> float:
    exp = getattr(s, "token_expires_at", None)
    if exp is None:
        return 3600.0
    try:
        if isinstance(exp, (int, float)):
            return float(exp) - time.time()
        if hasattr(exp, "timestamp"):
            return float(exp.timestamp()) - time.time()
        if isinstance(exp, str) and exp.strip():
            dt = datetime.fromisoformat(exp.strip().replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp() - time.time()
    except Exception:
        pass
    return 3600.0


def _sync_and_adopt_sessions():
    local_sessions, assignments = _safe_sync_sessions()
    ep_map = {a.endpoint: a for a in assignments}
    registry = _load_instance_registry()
    known_eps = {getattr(s, "endpoint", None) for s in local_sessions.values() if getattr(s, "endpoint", None)}

    # 自动纳管已在 registry 登记但尚未加入 state.store 的托管多实例，确保 keepalive_daemon 同时保活所有实例
    for a in assignments:
        ep = getattr(a, "endpoint", None)
        if not ep or ep in known_eps:
            continue
        if ep in registry or not local_sessions:
            s_name = _default_session_name(
                a.accelerator.value,
                ep,
                local_sessions,
                registry=registry,
            ) if (ep in registry or local_sessions) else "easycolab"
            s_auto = SessionState(
                name=s_name,
                token=a.runtime_proxy_info.token,
                url=a.runtime_proxy_info.url,
                endpoint=ep,
                token_expires_at=a.runtime_proxy_info.expires_at(),
                session_id=uuid.uuid4().hex,
                variant=Variant.GPU.value,
                accelerator=a.accelerator.value,
                machine_shape=Shape.HIGH_RAM.name if a.machine_shape == Shape.HIGH_RAM else Shape.STANDARD.name,
            )
            with _quiet_colab_cli():
                state.store.add(s_auto)
            local_sessions[s_name] = s_auto
            known_eps.add(ep)
            if not registry:
                # 首次从无本地记录状态收编首个实例后，不再自动收编未知的后续非托管 Notebook
                break

    for s in list(local_sessions.values()):
        changed = False
        if s.endpoint in ep_map:
            info_proxy = ep_map[s.endpoint].runtime_proxy_info
            if s.token != info_proxy.token or s.url != info_proxy.url:
                changed = True
            s.token = info_proxy.token
            s.url = info_proxy.url
            s.token_expires_at = info_proxy.expires_at()
        if not getattr(s, "session_id", None):
            s.session_id = uuid.uuid4().hex
            changed = True
        if changed:
            with _quiet_colab_cli():
                state.store.add(s)

    return local_sessions, assignments


def _touch_colab_tunnel_keepalive(endpoint: str) -> bool:
    if not endpoint:
        return False
    try:
        keepalive_url = f"{state.client.colab_domain.rstrip('/')}/tun/m/{endpoint}/keep-alive/"
        state.client._issue_request(
            keepalive_url,
            method="GET",
            headers={"X-Colab-Tunnel": "Google"},
        )
        return True
    except Exception:
        return False


def _resolve_or_start_remote_kernel(s: SessionState, timeout_s: float = 12.0) -> str:
    """
    通过 Colab 官方 Runtime Proxy 查询或启动远端 Jupyter 内核。
    严禁删除其他已有内核，避免误杀并发调试或子任务内核。
    """
    base_url = (s.url or "").rstrip("/")
    if not base_url or not s.token:
        raise RuntimeError("Session missing runtime proxy URL or token")

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Colab-Client-Agent": "vscode",
        "X-Colab-Runtime-Proxy-Token": s.token,
        "Authorization": f"Bearer {s.token}",
    }
    params = {"authuser": "0", "colab-runtime-proxy-token": s.token}

    r = requests.get(f"{base_url}/api/kernels", headers=headers, params=params, timeout=timeout_s)
    if r.status_code in (401, 403):
        raise _ProxyAuthExpiredError(f"Runtime proxy HTTP {r.status_code}")
    if r.status_code != 200:
        raise RuntimeError(f"GET /api/kernels returned HTTP {r.status_code}")

    kernels = r.json() if r.text else []
    active_kid = None
    if isinstance(kernels, list) and kernels:
        known_ids = [str(k.get("id")) for k in kernels if isinstance(k, dict) and k.get("id")]
        if known_ids:
            active_kid = s.kernel_id if (s.kernel_id in known_ids) else known_ids[0]

    if not active_kid:
        pr = requests.post(
            f"{base_url}/api/kernels",
            headers=headers,
            params=params,
            json={"name": "python3"},
            timeout=timeout_s,
        )
        if pr.status_code in (401, 403):
            raise _ProxyAuthExpiredError(f"POST /api/kernels HTTP {pr.status_code}")
        if pr.status_code not in (200, 201):
            raise RuntimeError(f"POST /api/kernels returned HTTP {pr.status_code}")
        pdata = pr.json() if pr.text else {}
        active_kid = str(pdata.get("id") or "")

    if not active_kid:
        raise RuntimeError("Unable to obtain remote Jupyter kernel_id")

    changed = False
    if s.kernel_id != active_kid:
        s.kernel_id = active_kid
        changed = True
    if not s.session_id:
        s.session_id = uuid.uuid4().hex
        changed = True
    if changed:
        with _quiet_colab_cli():
            state.store.add(s)
    return active_kid


def _open_colab_proxy_ws(s: SessionState, kernel_id: str, timeout_s: float = 15.0) -> websocket.WebSocket:
    """
    建立直连 Google Colab 官方外部 Runtime Proxy (*.prod.r.colab.dev) 的常驻 WebSocket 通道，
    显式携带 colab-runtime-proxy-token、token、authuser=0 查询参数与认证请求头，
    确保 Google 外部代理层始终观测到活跃 WebSocket 连接 (connections >= 1)。
    """
    base_url = (s.url or "").rstrip("/")
    if base_url.startswith("https://"):
        ws_base = "wss://" + base_url[len("https://"):]
    elif base_url.startswith("http://"):
        ws_base = "ws://" + base_url[len("http://"):]
    else:
        ws_base = "wss://" + base_url

    sid = s.session_id or uuid.uuid4().hex
    qs = urllib.parse.urlencode({
        "session_id": sid,
        "token": s.token,
        "colab-runtime-proxy-token": s.token,
        "authuser": "0",
    })
    ws_url = f"{ws_base}/api/kernels/{kernel_id}/channels?{qs}"
    headers = [
        f"X-Colab-Runtime-Proxy-Token: {s.token}",
        f"Authorization: Bearer {s.token}",
        "X-Colab-Client-Agent: vscode",
        "User-Agent: EasyColab/2.0",
    ]
    return websocket.create_connection(
        ws_url,
        timeout=timeout_s,
        origin=base_url,
        header=headers,
        enable_multithread=True,
    )


def _drain_proxy_ws_frames(ws: websocket.WebSocket, wait_s: float = 1.0, stop_on_msg_id: str = "") -> bool:
    """排空服务端下发的 Jupyter 消息与控制帧（自动回复 Ping -> Pong，遇到 Close 立即返回 False）"""
    deadline = time.time() + max(0.05, wait_s)
    while True:
        rem = deadline - time.time()
        if rem <= 0:
            return True
        try:
            ws.settimeout(min(rem, 1.0))
            opcode, frame = ws.recv_data(control_frame=True)
            if opcode == ABNF.OPCODE_CLOSE:
                return False
            if stop_on_msg_id and opcode == ABNF.OPCODE_TEXT and frame:
                raw_txt = frame.decode("utf-8", errors="replace") if isinstance(frame, (bytes, bytearray)) else str(frame)
                if stop_on_msg_id in raw_txt and '"execute_reply"' in raw_txt:
                    return True
        except websocket.WebSocketTimeoutException:
            return True
        except Exception:
            return False


def _send_proxy_ws_heartbeat(ws: websocket.WebSocket, session_id: str, touch_client_hb: bool = False) -> bool:
    """在已打开的外部 Runtime Proxy WebSocket 上发送 Ping 与 Jupyter execute_request，刷新代理与内核活跃度"""
    msg_id = uuid.uuid4().hex
    hb_file = "/tmp/.easycolab_client_hb" if touch_client_hb else "/tmp/.easycolab_proxy_hb"
    code = f"import pathlib; pathlib.Path('{hb_file}').touch(); _easycolab_hb = 1"
    msg = {
        "channel": "shell",
        "header": {
            "msg_id": msg_id,
            "username": "easycolab",
            "session": session_id,
            "date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": False,
        },
    }
    try:
        ws.settimeout(10.0)
        ws.ping("easycolab")
        ws.send(json.dumps(msg))
        return _drain_proxy_ws_frames(ws, wait_s=2.0, stop_on_msg_id=msg_id)
    except Exception:
        return False


def cmd_keepalive():
    """
    单次向活跃的 Colab Headless 实例发送官方 Runtime Proxy 探活与轻量 Jupyter 内核脉冲（兼容 CLI / Windows / Linux 调用）。
    macOS 客户端优先使用长连接常驻子进程 keepalive_daemon。
    """
    if not _has_valid_token():
        print(json.dumps({"ok": True, "logged_in": False}))
        return

    try:
        local_sessions, _ = _sync_and_adopt_sessions()
        touched = 0
        warnings = []

        for s in list(local_sessions.values()):
            _touch_colab_tunnel_keepalive(s.endpoint)
            if not s.url or not s.token:
                continue

            ws = None
            try:
                active_kid = _resolve_or_start_remote_kernel(s, timeout_s=12.0)
                ws = _open_colab_proxy_ws(s, active_kid, timeout_s=15.0)
                if _send_proxy_ws_heartbeat(ws, s.session_id or uuid.uuid4().hex, touch_client_hb=True):
                    touched += 1
                else:
                    warnings.append(f"{s.name}: heartbeat closed")
            except Exception as touch_err:
                warnings.append(f"{s.name}: {touch_err}")
            finally:
                if ws is not None:
                    try:
                        ws.close(timeout=1)
                    except Exception:
                        pass

        info = state.client.get_consumption_user_info()
        payload = {
            "ok": True,
            "logged_in": True,
            "touched_kernels": touched,
            "paid_balance": round(float(info.paid_compute_units_balance), 2),
            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
            "active_assignments": int(info.assignments_count),
        }
        if warnings:
            payload["warning"] = "; ".join(warnings)
        print(json.dumps(payload))
    except Exception as e:
        print(json.dumps({"ok": False, "logged_in": True, "error": str(e)}))


def cmd_keepalive_daemon():
    """
    macOS 原生客户端专属常驻保活守护进程：
    - 始终保持一条直连 Google Colab 官方外部 Runtime Proxy (wss://*.prod.r.colab.dev/api/kernels/<id>/channels)
      的真实 WebSocket 长连接，使代理层与 Jupyter 内核的 connections 持续 >= 1（杜绝无浏览器模式下 ~20 分钟孤儿实例强杀）；
    - 每 25 秒在长连接上发送一次 WebSocket Ping + Jupyter execute_request 脉冲；
    - 每 45 秒调用 /tun/m/<endpoint>/keep-alive/ 并拉取实时 CU 余额/速率，通过 KEEPALIVE_JSON: 行流式推送给客户端；
    - 在代理 Token 临近过期 (<10 分钟) 或连接断开时自动刷新 runtimeProxyInfo 并无缝重连；
    - 监听父进程 stdin 管道 EOF 与进程生命周期：一旦 EasyColab.app 断开实例或退出，立即关闭 WebSocket 并零残留退出。
    """
    stop_event = threading.Event()
    initial_ppid = os.getppid()

    def _handle_sig(_signum, _frame):
        stop_event.set()

    for sig_name in ("SIGTERM", "SIGINT"):
        sig_obj = getattr(signal, sig_name, None)
        if sig_obj is not None:
            try:
                signal.signal(sig_obj, _handle_sig)
            except Exception:
                pass

    if not sys.stdin.isatty():
        def _watch_stdin_eof():
            try:
                while not stop_event.is_set():
                    data = sys.stdin.read(1)
                    if data == "":
                        stop_event.set()
                        break
            except Exception:
                stop_event.set()

        threading.Thread(target=_watch_stdin_eof, daemon=True).start()

    def _emit(obj: dict):
        try:
            print("KEEPALIVE_JSON:" + json.dumps(obj, ensure_ascii=False), flush=True)
        except Exception:
            stop_event.set()

    if not _has_valid_token():
        _emit({"ok": True, "logged_in": False, "ws_connected": False})
        return

    ws_map: dict[str, tuple[websocket.WebSocket, str, str]] = {}  # endpoint -> (ws, url, token)

    def _close_ws(ep: str):
        entry = ws_map.pop(ep, None)
        if entry is not None:
            try:
                entry[0].close(timeout=1)
            except Exception:
                pass

    def _close_all_ws():
        for ep in list(ws_map.keys()):
            _close_ws(ep)

    last_sync_ts = 0.0
    last_hb_ts = 0.0
    last_usage_ts = 0.0
    force_resync = True
    local_sessions: dict[str, SessionState] = {}

    try:
        while not stop_event.is_set():
            if os.name != "nt" and os.getppid() != initial_ppid:
                stop_event.set()
                break

            now = time.time()
            min_ttl = min((_session_token_ttl_seconds(s) for s in local_sessions.values()), default=3600.0)
            if force_resync or (now - last_sync_ts) >= 480.0 or min_ttl < 600.0:
                try:
                    local_sessions, _ = _sync_and_adopt_sessions()
                    last_sync_ts = time.time()
                    force_resync = False
                    active_eps = {s.endpoint for s in local_sessions.values() if s.endpoint}
                    for stale_ep in list(ws_map.keys()):
                        if stale_ep not in active_eps:
                            _close_ws(stale_ep)
                except Exception as sync_err:
                    _emit({"ok": False, "logged_in": True, "ws_connected": bool(ws_map), "warning": f"sync: {sync_err}"})

            if not local_sessions:
                if (now - last_usage_ts) >= 30.0:
                    try:
                        info = state.client.get_consumption_user_info()
                        last_usage_ts = time.time()
                        _emit({
                            "ok": True,
                            "logged_in": True,
                            "ws_connected": False,
                            "paid_balance": round(float(info.paid_compute_units_balance), 2),
                            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
                            "active_assignments": int(info.assignments_count),
                        })
                    except Exception:
                        pass
                stop_event.wait(6.0)
                force_resync = True
                continue

            # 确保每个活跃会话都持有常驻外部 Runtime Proxy WebSocket
            need_hb = (time.time() - last_hb_ts) >= 25.0
            any_ws_ok = False
            ws_endpoints_status: dict[str, bool] = {}
            last_warn = ""

            for s in list(local_sessions.values()):
                if stop_event.is_set():
                    break
                ep = s.endpoint or s.name
                ep_ok = False
                existing = ws_map.get(ep)
                if existing is not None:
                    old_ws, old_url, old_tok = existing
                    if not getattr(old_ws, "connected", False) or old_url != s.url or old_tok != s.token:
                        _close_ws(ep)
                        existing = None

                if existing is None:
                    try:
                        kid = _resolve_or_start_remote_kernel(s, timeout_s=12.0)
                        new_ws = _open_colab_proxy_ws(s, kid, timeout_s=15.0)
                        if _send_proxy_ws_heartbeat(new_ws, s.session_id or uuid.uuid4().hex, touch_client_hb=False):
                            ws_map[ep] = (new_ws, s.url, s.token)
                            any_ws_ok = True
                            ep_ok = True
                            last_hb_ts = time.time()
                        else:
                            try:
                                new_ws.close(timeout=1)
                            except Exception:
                                pass
                            force_resync = True
                    except _ProxyAuthExpiredError as auth_err:
                        last_warn = str(auth_err)
                        force_resync = True
                    except Exception as conn_err:
                        last_warn = str(conn_err)
                        force_resync = True
                else:
                    cur_ws = existing[0]
                    if need_hb:
                        if _send_proxy_ws_heartbeat(cur_ws, s.session_id or uuid.uuid4().hex, touch_client_hb=False):
                            any_ws_ok = True
                            ep_ok = True
                        else:
                            _close_ws(ep)
                            force_resync = True
                            last_warn = "ws heartbeat lost, reconnecting"
                    else:
                        if _drain_proxy_ws_frames(cur_ws, wait_s=1.0):
                            any_ws_ok = True
                            ep_ok = True
                        else:
                            _close_ws(ep)
                            force_resync = True
                            last_warn = "ws closed by peer, reconnecting"
                ws_endpoints_status[ep] = ep_ok

            if need_hb and any_ws_ok:
                last_hb_ts = time.time()

            now2 = time.time()
            if (now2 - last_usage_ts) >= 45.0 or (last_usage_ts == 0.0):
                for s in list(local_sessions.values()):
                    _touch_colab_tunnel_keepalive(s.endpoint)
                try:
                    info = state.client.get_consumption_user_info()
                    last_usage_ts = now2
                    payload = {
                        "ok": True,
                        "logged_in": True,
                        "ws_connected": any_ws_ok,
                        "ws_endpoints": ws_endpoints_status,
                        "paid_balance": round(float(info.paid_compute_units_balance), 2),
                        "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
                        "active_assignments": int(info.assignments_count),
                    }
                    if last_warn:
                        payload["warning"] = last_warn
                    _emit(payload)
                except Exception as usage_err:
                    last_usage_ts = now2
                    _emit({"ok": False, "logged_in": True, "ws_connected": any_ws_ok, "ws_endpoints": ws_endpoints_status, "warning": str(usage_err)})

            if not any_ws_ok and not stop_event.is_set():
                stop_event.wait(3.0)
            elif need_hb and not stop_event.is_set():
                stop_event.wait(1.0)
    finally:
        _close_all_ws()


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("auth_status")
    sub.add_parser("auth_url")
    sub.add_parser("auth_logout")

    p_code = sub.add_parser("auth_code")
    p_code.add_argument("code", type=str)

    p_wake = sub.add_parser("wake")
    p_wake.add_argument("--gpu", type=str, default="G4")
    p_wake.add_argument(
        "--app-py",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"),
    )
    p_wake.add_argument("--endpoint", type=str, default="")
    p_wake.add_argument("--new", action="store_true", dest="new_instance")
    p_wake.add_argument("--name", type=str, default="")
    p_wake.add_argument("--replace-existing", action="store_true", dest="replace_existing")
    p_wake.add_argument("--force-redeploy", action="store_true", dest="force_redeploy")

    sub.add_parser("keepalive")
    sub.add_parser("keepalive_daemon")

    p_stop_one = sub.add_parser("stop_instance")
    p_stop_one.add_argument("--endpoint", type=str, default="")
    p_stop_one.add_argument("--name", type=str, default="")

    p_probe = sub.add_parser("probe_instance")
    p_probe.add_argument("--endpoint", type=str, required=True)

    p_stop_all = sub.add_parser("stop_all")
    p_stop_all.add_argument("--managed-only", action="store_true", dest="managed_only")

    args = parser.parse_args(argv)
    if args.cmd == "auth_status":
        cmd_auth_status()
    elif args.cmd == "auth_url":
        cmd_auth_url()
    elif args.cmd == "auth_logout":
        cmd_auth_logout()
    elif args.cmd == "auth_code":
        cmd_auth_code(args.code)
    elif args.cmd == "wake":
        cmd_wake(
            args.gpu,
            args.app_py,
            endpoint=args.endpoint or None,
            new_instance=bool(args.new_instance),
            instance_name=args.name or None,
            replace_existing=bool(args.replace_existing),
            force_redeploy=bool(args.force_redeploy),
        )
    elif args.cmd == "keepalive":
        cmd_keepalive()
    elif args.cmd == "keepalive_daemon":
        cmd_keepalive_daemon()
    elif args.cmd == "stop_instance":
        cmd_stop_instance(endpoint=args.endpoint, name=args.name)
    elif args.cmd == "probe_instance":
        cmd_probe_instance(args.endpoint)
    elif args.cmd == "stop_all":
        cmd_stop_all(managed_only=bool(args.managed_only))


def run_cli(argv=None):
    """供独立 CLI 或打包后的 Windows exe (--bridge-cli) 子进程模式统一调用"""
    try:
        main(argv)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    run_cli()
