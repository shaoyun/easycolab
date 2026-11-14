"""EasyColab 服务端模块: status_service."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import time

from . import runtime_state as rs
from .comfyui_service import get_comfyui_status_dict
from .model_catalog import MODEL_CATALOG, _get_cached_model_files_status, _get_custom_models_public_list, _normalize_builtin_model_alias
from .runtime_state import BOOT_TIME, INITIAL_API_KEY, LOADED_RUNTIMES, MODEL_DOWNLOAD_STATE, _ENV_API_TOKENS, _get_valid_api_tokens, _has_rclone_drive_remote, get_live_console_Lines
from .telemetry_service import get_live_telemetry_snapshot
from .toybox_service import get_toybox_status_dict

def get_gpu_status_dict(model_id: Optional[str] = None) -> Dict[str, Any]:
    """返回结构化的 GPU 状态、实例磁盘空间、多模型注册信息、CU 算力消耗账本、闲置倒计时与实时终端进度流"""
    # (shared state on runtime_state)
    norm_id = _normalize_builtin_model_alias(model_id)
    if norm_id and norm_id in MODEL_CATALOG:
        rs.SELECTED_MODEL_ID = norm_id
    target_id = rs.SELECTED_MODEL_ID

    file_status = _get_cached_model_files_status(target_id)
    hw = get_live_telemetry_snapshot()
    uptime_seconds = int(time.time() - BOOT_TIME)
    idle_seconds = int(max(0.0, time.time() - rs.LAST_ACTIVE_TIME))
    idle_remaining_seconds = (
        max(0, rs.IDLE_TIMEOUT_MINUTES * 60 - idle_seconds) if rs.IDLE_TIMEOUT_MINUTES > 0 else -1
    )

    gpu_name = hw["gpu_name"]
    cuda_avail = hw["cuda_available"]
    free_gb = hw["vram_free_gb"]
    total_gb = hw["vram_total_gb"]
    used_gb = hw["vram_used_gb"]
    cu_rate = hw["cu_burn_rate_per_hour"]

    session_cu_used = round((uptime_seconds / 3600.0) * cu_rate, 3)
    loaded_ids = list(LOADED_RUNTIMES.keys())
    loaded_details = [
        {
            "id": mid,
            "name": rt.get("name", mid),
            "mode": rt.get("mode_label", ""),
            "vram_gb": rt.get("vram_gb", 0.0),
            "context_length": int((rt.get("llm_bundle") or {}).get("context_length") or 0),
        }
        for mid, rt in LOADED_RUNTIMES.items()
    ]
    loaded_ctx = 0
    if target_id in LOADED_RUNTIMES and LOADED_RUNTIMES[target_id].get("llm_bundle"):
        loaded_ctx = int(LOADED_RUNTIMES[target_id]["llm_bundle"].get("context_length") or 0)
    else:
        for rt in LOADED_RUNTIMES.values():
            if rt.get("llm_bundle") and rt["llm_bundle"].get("context_length"):
                loaded_ctx = int(rt["llm_bundle"]["context_length"])
                break

    any_loaded = bool(LOADED_RUNTIMES) or (rs.PIPELINE is not None) or (rs.LLM_BUNDLE is not None)
    loaded_for_selected = (target_id in LOADED_RUNTIMES) or (any_loaded and rs.LOADED_MODEL_ID == target_id)
    summary_mode = (
        LOADED_RUNTIMES[target_id]["mode_label"]
        if target_id in LOADED_RUNTIMES
        else (" + ".join(rt["name"] for rt in LOADED_RUNTIMES.values()) if LOADED_RUNTIMES else (rs.CURRENT_MODEL_MODE or "未加载"))
    )

    base = {
        "ok": True,
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "vram_free_gb": free_gb,
        "vram_total_gb": total_gb,
        "vram_used_gb": used_gb,
        "model_loaded": loaded_for_selected,
        "any_model_loaded": any_loaded,
        "loaded_model_id": target_id if target_id in LOADED_RUNTIMES else rs.LOADED_MODEL_ID,
        "loaded_model_ids": loaded_ids,
        "loaded_models_detail": loaded_details,
        "current_model": summary_mode,
        "context_length": rs.QWEN_CONTEXT_LENGTH,
        "loaded_context_length": loaded_ctx,
        "is_downloading": MODEL_DOWNLOAD_STATE["is_downloading"],
        "is_loading_vram": MODEL_DOWNLOAD_STATE["is_loading_vram"],
        "task_progress": MODEL_DOWNLOAD_STATE["progress"],
        "task_message": MODEL_DOWNLOAD_STATE["message"],
        "uptime_seconds": uptime_seconds,
        "cu_burn_rate_per_hour": cu_rate,
        "session_cu_used": session_cu_used,
        "idle_timeout_minutes": rs.IDLE_TIMEOUT_MINUTES,
        "idle_remaining_seconds": idle_remaining_seconds,
        "api_key_enabled": True,
        "api_key_configured": bool(_ENV_API_TOKENS) or (rs.API_AUTH_KEY != INITIAL_API_KEY),
        "api_token_count": len(_get_valid_api_tokens()),
        "api_request_count": rs.API_REQUEST_COUNT,
        "console_lines": get_live_console_Lines(),
        "custom_models": _get_custom_models_public_list(),
    }
    for k in (
        "ram_total_gb", "ram_used_gb", "ram_free_gb",
        "disk_total_gb", "disk_used_gb", "disk_free_gb",
        "shm_total_gb", "shm_used_gb", "shm_free_gb",
        "drive_total_gb", "drive_used_gb", "drive_free_gb",
    ):
        base[k] = hw.get(k, 0.0)
    base.update(file_status)
    if _has_rclone_drive_remote():
        base["drive_persisted"] = True
    base["comfyui"] = get_comfyui_status_dict()
    if "get_toybox_status_dict" in globals():
        base["toybox"] = get_toybox_status_dict()
    return base

def get_gpu_and_model_summary_text() -> str:
    info = get_gpu_status_dict()
    lines = []
    if not info["cuda_available"]:
        lines.append("GPU 状态: 未检测到 GPU（请在 Colab 中切换为 A100 / L4 GPU）")
    else:
        lines.append(
            f"GPU: {info['gpu_name']} | 显存可用: {info['vram_free_gb']:.1f} / {info['vram_total_gb']:.1f} GB"
        )
    if info.get("ram_total_gb", 0) > 0:
        lines.append(
            f"实例内存 (RAM): 已用 {info['ram_used_gb']:.1f} GB | 可用 {info['ram_free_gb']:.1f} / {info['ram_total_gb']:.1f} GB"
        )
    if info.get("disk_total_gb", 0) > 0:
        shm_part = (
            f" | /dev/shm 内存盘可用: {info['shm_free_gb']:.1f} / {info['shm_total_gb']:.1f} GB"
            if info.get("shm_total_gb", 0) >= 1.0
            else ""
        )
        lines.append(
            f"实例磁盘: 已用 {info['disk_used_gb']:.1f} GB | 可用 {info['disk_free_gb']:.1f} / {info['disk_total_gb']:.1f} GB{shm_part}"
        )
    lines.append(
        f"算力计费: 费率 ≈ {info['cu_burn_rate_per_hour']:.2f} CU/小时 | 本次开机已用: {info['session_cu_used']:.2f} CU (运行 {info['uptime_seconds']//60} 分钟)"
    )
    ram_tag = " + /dev/shm 内存盘(2.4GB/s)" if info.get("ramdisk_enabled") else ""
    storage_tag = ("Google Drive 云盘持久化" if info["drive_persisted"] else "Colab 高速磁盘") + ram_tag
    lines.append(
        f"当前模型 [{info['selected_model_name']}] 权重 [{storage_tag}]: 已就绪 {info['downloaded_count']}/{info['total_count']} 个文件 ({info['total_size_gb']:.2f} GB)"
    )
    for f in info["files"][:8]:
        mark = "[已就绪]" if f["exists"] else "[未下载]"
        sz = f"({f['size_gb']:.2f} GB)" if f["exists"] else ""
        lines.append(f"   {mark} {f['name']} {sz}")
    if len(info["files"]) > 8:
        lines.append(f"   ... 其余 {len(info['files']) - 8} 个分片文件")

    vram_state = f"已加载进显存 ({info['current_model']})" if info["any_model_loaded"] else "尚未加载进显存"
    lines.append(f"显存引擎状态: {vram_state}")
    return "\n".join(lines)
