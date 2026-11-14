"""EasyColab 服务端主入口与向后兼容模块门面。

所有核心业务逻辑已按职责拆分至 `easycolab_server` 子模块包中；
本文件保留 CLI 启动编排（`--local` 本地预览与 Colab 鉴权隧道部署）及对历史单文件导出符号的兼容代理。
"""

import os
import sys
import contextlib
import signal
import types
import urllib.parse

# Remove the deployment secret before any submodule or model installer starts subprocesses.
_DEPLOY_PASSWORD = os.environ.pop("EASYCOLAB_ADMIN_PASSWORD", "") if __name__ == "__main__" else ""

from easycolab_server import *  # noqa: F401,F403
from easycolab_server import runtime_state as _rs

# 需与 easycolab_server.runtime_state 保持实时双向同步的可变全局状态名称
_SYNCED_STATE_ATTRS = frozenset(
    {
        "SELECTED_MODEL_ID",
        "LOADED_MODEL_ID",
        "CURRENT_MODEL_MODE",
        "PIPELINE",
        "LLM_BUNDLE",
        "QWEN_CONTEXT_LENGTH",
        "API_AUTH_KEY",
        "EXTRA_API_TOKENS",
        "API_REQUEST_COUNT",
        "IDLE_TIMEOUT_MINUTES",
        "LAST_ACTIVE_TIME",
        "_ACTIVE_API_STREAMS",
        "CURRENT_STAGE_BAR_LABEL",
        "GRADIO_BACKEND",
        "GRADIO_INGRESS",
        "COMFY_BACKEND",
        "COMFY_INGRESS",
        "TOYBOX_BACKEND",
        "TOYBOX_INGRESS",
        "DEFAULT_OUTPUT_DIR",
        "_DEPLOY_PASSWORD",
    }
)


class _AppCompatModule(types.ModuleType):
    """将可变状态标量读写实时代理至 easycolab_server.runtime_state，保障单文件历史引用与测试兼容。"""

    def __getattribute__(self, name: str):
        if name in _SYNCED_STATE_ATTRS:
            return getattr(_rs, name)
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value):
        if name in _SYNCED_STATE_ATTRS:
            setattr(_rs, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _AppCompatModule


if __name__ == "__main__":
    import getpass
    from gradio import networking

    def stop_server(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop_server)

    if _IS_LOCAL_PREVIEW:
        local_port = int(os.environ.get("EASYCOLAB_PORT", str(_rs.GRADIO_INGRESS)))
        ui = create_ui()
        ui.queue()
        app, _, _ = ui.launch(
            server_name="127.0.0.1",
            server_port=local_port,
            share=False,
            inbrowser=True,
            inline=False,
            prevent_thread_lock=True,
            css=CUSTOM_CSS,
        )
        attach_macos_api_routes(app)
        print(f"🌐 EasyColab 本地网页调试端已启动: http://127.0.0.1:{local_port}", flush=True)
        try:
            ui.block_thread()
        finally:
            ui.close()
    else:
        if authenticated_ingress is not None and (_DEPLOY_PASSWORD or sys.stdin.isatty()):
            password = _DEPLOY_PASSWORD or getpass.getpass("Management password (24+ URL-safe characters): ")
            ingress_ctx = authenticated_ingress(password)
        else:
            ingress_ctx = contextlib.nullcontext()

        with ingress_ctx:
            ui = create_ui()
            ui.queue()
            app, _, _ = ui.launch(
                server_name="127.0.0.1",
                server_port=_rs.GRADIO_BACKEND,
                share=False,
                inline=False,
                prevent_thread_lock=True,
                css=CUSTOM_CSS,
            )
            attach_macos_api_routes(app)
            tunnel = networking.setup_tunnel(
                local_host="127.0.0.1",
                local_port=_rs.GRADIO_INGRESS,
                share_token=ui.share_token,
                share_server_address=None,
                share_server_tls_certificate=None,
            )
            share_url = urllib.parse.urlsplit(tunnel)._replace(scheme="https").geturl()
            print(f"EasyColab URL: {share_url}", flush=True)
            _runtime_meta_file = "/content/.easycolab_runtime.json"
            _managed_marker_file = "/content/.easycolab_managed.json"
            if os.path.isdir("/content"):
                try:
                    import hashlib as _hashlib
                    import json as _json
                    with open(_managed_marker_file, "w", encoding="utf-8") as _mf:
                        _json.dump({
                            "managed_by": "easycolab",
                            "instance_name": _rs.INSTANCE_NAME or "easycolab",
                            "updated_at": time.time(),
                        }, _mf, ensure_ascii=False)
                    _pw_digest = _hashlib.sha256(_DEPLOY_PASSWORD.encode("utf-8")).hexdigest() if _DEPLOY_PASSWORD else ""
                    _tmp_rt = _runtime_meta_file + ".tmp"
                    with open(_tmp_rt, "w", encoding="utf-8") as _rf:
                        _json.dump({
                            "managed_by": "easycolab",
                            "pid": os.getpid(),
                            "instance_name": _rs.INSTANCE_NAME or "easycolab",
                            "bundle_hash": _rs.BUNDLE_HASH,
                            "share_url": share_url,
                            "password_sha256": _pw_digest,
                            "started_at": time.time(),
                        }, _rf, ensure_ascii=False)
                    os.replace(_tmp_rt, _runtime_meta_file)
                except Exception:
                    pass
            try:
                ui.block_thread()
            finally:
                if os.path.isfile(_runtime_meta_file):
                    try:
                        import json as _json
                        with open(_runtime_meta_file, "r", encoding="utf-8") as _rf:
                            _cur_meta = _json.load(_rf)
                        if int(_cur_meta.get("pid") or -1) == os.getpid():
                            os.remove(_runtime_meta_file)
                    except Exception:
                        pass
                ui.close()
