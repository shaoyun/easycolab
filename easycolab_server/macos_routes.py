"""EasyColab 服务端模块: macos_routes."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import re
import threading
import time
import uuid

from . import runtime_state as rs
from .comfyui_service import _resolve_comfyui_persistence_paths, download_comfyui_model_core, free_comfyui_vram_core, get_comfyui_status_dict, install_comfyui_preset_pack_core, start_comfyui_core, stop_comfyui_core, sync_comfyui_to_drive_core
from .llm_runtime import _get_qwen_llm_bundle, _is_llm_model_id
from .media_pipeline import _run_async_generation_job, _sync_legacy_globals, load_model_to_vram_core, unload_model_from_vram_core
from .model_catalog import DEFAULT_OUTPUT_DIR, LEGACY_DRIVE_OUTPUT_DIRS, MODEL_CATALOG, _normalize_builtin_model_alias, download_model_weights_core, fetch_hf_model_detail_core, fetch_hf_trending_models_core, register_custom_hf_model_core, remove_custom_hf_model_core
from .openai_gateway import attach_openai_routes
from .qwen_kernels import _clamp_context_length
from .runtime_state import ABORT_GENERATION_EVENT, ACTIVE_TQDM_BARS, API_KEY_LOCK, COMFYUI_STATE, CONSOLE_LOCK, DRIVE_MOUNTED, GPU_LOCK, INITIAL_API_KEY, JOBS, LOADED_RUNTIMES, MODEL_DOWNLOAD_STATE, TOYBOX_STATE, _ENV_API_TOKENS, _JOB_CANCEL_EVENTS, get_live_console_Lines, touch_activity
from .schemas import ApiKeyConfigRequest, ComfyInstallPackRequest, ComfyStartRequest, ComfyWeightDownloadRequest, ContextConfigRequest, DownloadModelRequest, GenerateVideoRequest, IdleConfigRequest, LoadModelRequest, RegisterCustomModelRequest, RemoveCustomModelRequest, SelectModelRequest, ToyboxPrepareRequest, ToyboxStartRequest, ToyboxStopRequest, UnloadModelRequest
from .status_service import get_gpu_status_dict
from .telemetry_service import disconnect_colab_runtime, get_live_telemetry_snapshot
from .toybox_service import get_toybox_status_dict, prepare_toybox_assets_core, start_toybox_core, stop_toybox_core

def attach_macos_api_routes(app: FastAPI):
    """在 Gradio 底层的 FastAPI 实例上挂载原生 macOS 客户端接口与 OpenAI / Anthropic 兼容网关。"""
    attach_openai_routes(app)

    @app.get("/minimax_api/status")
    def api_status(model_id: Optional[str] = None, touch: Optional[int] = None):
        if touch:
            touch_activity()
        return get_gpu_status_dict(model_id=model_id)

    @app.post("/minimax_api/select_model")
    def api_select_model(req: SelectModelRequest):
        touch_activity()
        norm_id = _normalize_builtin_model_alias(req.model_id, req.repo_id or "")
        if norm_id and norm_id not in MODEL_CATALOG and req.repo_id:
            register_custom_hf_model_core(
                repo_id=req.repo_id,
                custom_name=req.custom_name,
                include_pattern=req.include_pattern,
                modality=req.modality or "auto",
                hf_token=req.hf_token,
                model_id_override=norm_id,
            )
        return get_gpu_status_dict(model_id=norm_id)

    @app.post("/minimax_api/config_idle")
    def api_config_idle(req: IdleConfigRequest):
        # (shared state on runtime_state)
        rs.IDLE_TIMEOUT_MINUTES = req.idle_timeout_minutes
        touch_activity()
        return get_gpu_status_dict()

    @app.post("/minimax_api/config_context")
    def api_config_context(req: ContextConfigRequest):
        # (shared state on runtime_state)
        rs.QWEN_CONTEXT_LENGTH = _clamp_context_length(req.context_length)
        # BF16 Transformers 运行时支持直接热更新动态上下文上限；GGUF 需重载以重分配 KV 槽位
        for mid, rt in LOADED_RUNTIMES.items():
            b = rt.get("llm_bundle")
            if b and b.get("backend") == "transformers":
                b["context_length"] = rs.QWEN_CONTEXT_LENGTH
                mname = MODEL_CATALOG.get(mid, {}).get("name", mid)
                rt["mode_label"] = f"{mname} · BF16 原生 · {rs.QWEN_CONTEXT_LENGTH // 1024}K"
        _sync_legacy_globals()
        touch_activity()
        print(f"[上下文配置] 目标最大上下文长度已设为 {rs.QWEN_CONTEXT_LENGTH // 1024}K ({rs.QWEN_CONTEXT_LENGTH} tokens)")
        return get_gpu_status_dict(model_id=rs.SELECTED_MODEL_ID)

    @app.get("/minimax_api/hf_trending")
    def api_hf_trending(
        category: str = "all",
        limit: int = 12,
        hf_token: str = "",
        force_refresh: bool = False,
        search_query: str = "",
    ):
        items = fetch_hf_trending_models_core(
            category=category,
            limit=limit,
            hf_token=hf_token,
            force_refresh=force_refresh,
            search_query=search_query,
        )
        return {"ok": True, "category": category, "search_query": search_query, "models": items}

    @app.get("/minimax_api/hf_model_detail")
    def api_hf_model_detail(repo_id: str = "", hf_token: str = ""):
        return fetch_hf_model_detail_core(repo_id=repo_id, hf_token=hf_token)

    @app.post("/minimax_api/custom_model/register")
    def api_register_custom_model(req: RegisterCustomModelRequest):
        touch_activity()
        eff_mod = req.modality if (req.modality and req.modality != "auto") else req.modality_hint
        ok, msg, spec = register_custom_hf_model_core(
            repo_id=req.repo_id,
            custom_name=req.custom_name,
            include_pattern=req.include_pattern,
            modality=eff_mod,
            hf_token=req.hf_token,
        )
        if ok and spec and req.auto_download and not MODEL_DOWNLOAD_STATE["is_downloading"]:
            threading.Thread(
                target=rs.resolve_callable("download_model_weights_core", download_model_weights_core),
                args=(spec["id"], req.hf_token),
                daemon=True,
            ).start()
        res = get_gpu_status_dict(model_id=spec["id"] if (ok and spec) else rs.SELECTED_MODEL_ID)
        res["action_ok"] = ok
        res["action_message"] = msg
        if spec and spec.get("is_custom"):
            res["registered_model"] = {
                "id": spec["id"],
                "name": spec["name"],
                "repo_id": spec["repo_id"],
                "modality": spec["modality"],
                "category": spec["category"],
                "approx_size_gb": spec["approx_size_gb"],
                "include_pattern": spec.get("include_pattern", ""),
                "pipeline_arch": spec.get("pipeline_arch", "gguf"),
                "is_custom": True,
            }
        return res

    @app.post("/minimax_api/custom_model/remove")
    def api_remove_custom_model(req: RemoveCustomModelRequest):
        touch_activity()
        ok, msg = remove_custom_hf_model_core(req.model_id, delete_files=req.delete_files)
        res = get_gpu_status_dict(model_id=rs.SELECTED_MODEL_ID)
        res["action_ok"] = ok
        res["action_message"] = msg
        return res

    @app.post("/minimax_api/config_api_key")
    def api_config_api_key(req: ApiKeyConfigRequest):
        # (shared state on runtime_state)
        tokens = [t.strip() for t in re.split(r"[\s,;]+", req.api_key or "") if t.strip()]
        if not tokens:
            raise HTTPException(status_code=422, detail="An inference API key is required")
        with API_KEY_LOCK:
            if not req.initialize_only or (not _ENV_API_TOKENS and rs.API_AUTH_KEY == INITIAL_API_KEY):
                rs.API_AUTH_KEY = tokens[0]
                if len(tokens) > 1:
                    rs.EXTRA_API_TOKENS = tokens[1:]
        touch_activity()
        return get_gpu_status_dict()


    @app.post("/minimax_api/download_model")
    def api_download_model(req: DownloadModelRequest):
        touch_activity()
        norm_id = _normalize_builtin_model_alias(req.model_id, req.repo_id or "")
        if norm_id and norm_id not in MODEL_CATALOG and req.repo_id:
            register_custom_hf_model_core(
                repo_id=req.repo_id,
                custom_name=req.custom_name,
                include_pattern=req.include_pattern,
                modality=req.modality or "auto",
                hf_token=req.hf_token,
                model_id_override=norm_id,
            )
        target_id = norm_id if (norm_id and norm_id in MODEL_CATALOG) else rs.SELECTED_MODEL_ID
        if not MODEL_DOWNLOAD_STATE["is_downloading"]:
            t = threading.Thread(
                target=rs.resolve_callable("download_model_weights_core", download_model_weights_core),
                args=(target_id, req.hf_token),
                daemon=True,
            )
            t.start()
        return get_gpu_status_dict(model_id=target_id)

    @app.post("/minimax_api/load_model")
    def api_load_model(req: LoadModelRequest):
        # (shared state on runtime_state)
        touch_activity()
        if req.context_length is not None:
            rs.QWEN_CONTEXT_LENGTH = _clamp_context_length(req.context_length)
        norm_id = _normalize_builtin_model_alias(req.model_id, req.repo_id or "")
        if norm_id and norm_id not in MODEL_CATALOG and req.repo_id:
            register_custom_hf_model_core(
                repo_id=req.repo_id,
                custom_name=req.custom_name,
                include_pattern=req.include_pattern,
                modality=req.modality or "auto",
                hf_token=req.hf_token,
                model_id_override=norm_id,
            )
        target_id = norm_id if (norm_id and norm_id in MODEL_CATALOG) else rs.SELECTED_MODEL_ID
        if not MODEL_DOWNLOAD_STATE["is_loading_vram"]:
            t = threading.Thread(
                target=rs.resolve_callable("load_model_to_vram_core", load_model_to_vram_core),
                args=(target_id, req.precision_mode, req.offload_mode, req.hf_token, req.context_length),
                daemon=True,
            )
            t.start()
        return get_gpu_status_dict(model_id=target_id)

    @app.post("/minimax_api/unload_model")
    def api_unload_model(req: UnloadModelRequest):
        touch_activity()
        rs.resolve_callable("unload_model_from_vram_core", unload_model_from_vram_core)(req.model_id or None)
        return get_gpu_status_dict(model_id=rs.SELECTED_MODEL_ID)

    @app.post("/minimax_api/cancel")
    def api_cancel_generation():
        # (shared state on runtime_state)
        touch_activity()
        ABORT_GENERATION_EVENT.set()
        for ev in list(_JOB_CANCEL_EVENTS.values()):
            ev.set()
        for jid, j in list(JOBS.items()):
            if j.get("status") in ("queued", "running"):
                j["status"] = "failed"
                j["progress"] = 0.0
                j["message"] = "已手动停止当前推理任务"
        rs.CURRENT_STAGE_BAR_LABEL = None
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()
        print("[任务控制] 已收到停止指令，正在中断当前 GPU 推理并清空进度缓存...")

        def _clear_abort_after_release():
            with GPU_LOCK:
                ABORT_GENERATION_EVENT.clear()
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS.clear()

        threading.Thread(target=_clear_abort_after_release, daemon=True).start()
        return get_gpu_status_dict(model_id=rs.SELECTED_MODEL_ID)

    @app.post("/minimax_api/generate")
    def api_generate(req: GenerateVideoRequest):
        touch_activity()
        norm_req_id = _normalize_builtin_model_alias(req.model_id)
        target_model_id = (
            norm_req_id
            if (norm_req_id and norm_req_id in MODEL_CATALOG)
            else rs.SELECTED_MODEL_ID
        )
        req.model_id = target_model_id
        if not LOADED_RUNTIMES and rs.PIPELINE is None and rs.LLM_BUNDLE is None:
            return {
                "job_id": "none",
                "status": "failed",
                "progress": 0.0,
                "message": "尚未有任何模型加载至 GPU 显存！请先在左侧点击「下载权重」与「加载到显存」。",
                "seed": req.seed,
                "elapsed": 0.0,
                "filename": None,
                "video_url": None,
            }

        # 若请求的是 LLM/VLM 但当前显存中驻留的是另一变体（如选中了 BF16 但已加载 GGUF，或反之），自动无缝路由至已就绪变体
        if _is_llm_model_id(target_model_id) and target_model_id not in LOADED_RUNTIMES:
            alt_bundle = _get_qwen_llm_bundle(preferred_model_id=target_model_id)
            if alt_bundle and alt_bundle.get("model_id") in LOADED_RUNTIMES:
                target_model_id = str(alt_bundle["model_id"])
                req.model_id = target_model_id

        if target_model_id not in LOADED_RUNTIMES and rs.LOADED_MODEL_ID != target_model_id:
            resident_names = "、".join(rt["name"] for rt in LOADED_RUNTIMES.values()) or "无"
            req_name = MODEL_CATALOG.get(target_model_id, {}).get("name", target_model_id)
            return {
                "job_id": "none",
                "status": "failed",
                "progress": 0.0,
                "message": f"{req_name} 尚未加载到显存（当前已常驻：{resident_names}）。请点击左侧「加载到显存」，显存充裕时可同时常驻无需卸载现有模型。",
                "seed": req.seed,
                "elapsed": 0.0,
                "filename": None,
                "video_url": None,
            }

        # 若此前有残留未结束的任务，先置位其专属中断信号并清空进度条，防止旧任务继续占用 GPU 或污染新任务进度
        for old_jid, old_j in list(JOBS.items()):
            if old_j.get("status") in ("queued", "running"):
                old_ev = _JOB_CANCEL_EVENTS.get(old_jid)
                if old_ev is not None:
                    old_ev.set()
                old_j["status"] = "failed"
                old_j["progress"] = 0.0
                old_j["message"] = "已被新提交任务接替中断"
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()

        is_llm_req = _is_llm_model_id(target_model_id)
        batch_count = 1 if is_llm_req else max(1, min(8, int(getattr(req, "batch_count", 1) or 1)))
        job_id = uuid.uuid4().hex[:12]
        _JOB_CANCEL_EVENTS[job_id] = threading.Event()
        JOBS[job_id] = {
            "job_id": job_id,
            "model_id": target_model_id,
            "status": "queued",
            "progress": 0.0,
            "message": f"任务已提交至 Colab 队列 (共 {batch_count} 项)..." if batch_count > 1 else "任务已提交至 Colab 队列...",
            "seed": req.seed,
            "started_at": time.time(),
            "elapsed": 0.0,
            "filename": None,
            "video_url": None,
            "media_type": "text" if is_llm_req else (
                "image" if MODEL_CATALOG.get(target_model_id, {}).get("modality") == "image" else "video"
            ),
            "text_output": "",
            "thinking_output": "",
            "batch_count": batch_count,
            "completed_count": 0,
            "completed_videos": [],
        }
        t = threading.Thread(
            target=rs.resolve_callable("_run_async_generation_job", _run_async_generation_job),
            args=(job_id, req),
            daemon=True,
        )
        t.start()
        return JOBS[job_id]

    @app.get("/minimax_api/job/{job_id}")
    def api_job_status(job_id: str):
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Job not found")
        res = dict(JOBS[job_id])
        if res.get("status") in ("queued", "running"):
            touch_activity()
            if res.get("started_at"):
                res["elapsed"] = round(max(0.0, time.time() - float(res["started_at"])), 1)
        res["console_lines"] = get_live_console_Lines()
        res["telemetry"] = get_live_telemetry_snapshot()
        return res

    @app.get("/minimax_api/video/{filename}")
    @app.get("/minimax_api/media/{filename}")
    def api_download_media(filename: str):
        touch_activity()
        safe_name = os.path.basename(filename)
        file_path = os.path.join(rs.DEFAULT_OUTPUT_DIR, safe_name)
        if not os.path.exists(file_path) and DRIVE_MOUNTED:
            for leg_out in LEGACY_DRIVE_OUTPUT_DIRS:
                alt_path = os.path.join(leg_out, safe_name)
                if os.path.exists(alt_path):
                    file_path = alt_path
                    break
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Media file not found")
        ext = os.path.splitext(safe_name)[1].lower()
        mime = (
            "image/png" if ext == ".png"
            else "image/jpeg" if ext in (".jpg", ".jpeg")
            else "image/webp" if ext == ".webp"
            else "video/mp4"
        )
        return FileResponse(file_path, media_type=mime, filename=safe_name)

    @app.post("/minimax_api/disconnect")
    def api_disconnect():
        msg = disconnect_colab_runtime()
        return {"ok": True, "message": msg}

    # --------------------------------------------------------------------------
    # ComfyUI 云端节点工作台与 Google Drive 持久化专用接口 (/minimax_api/comfyui/*)
    # --------------------------------------------------------------------------
    @app.get("/minimax_api/comfyui/status")
    def api_comfyui_status():
        return {"ok": True, "comfyui": get_comfyui_status_dict()}

    @app.post("/minimax_api/comfyui/start")
    def api_comfyui_start(req: ComfyStartRequest):
        touch_activity()
        if not COMFYUI_STATE["is_starting"]:
            threading.Thread(
                target=rs.resolve_callable("start_comfyui_core", start_comfyui_core),
                args=(req.unload_existing, req.force_restart),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=rs.SELECTED_MODEL_ID)

    @app.post("/minimax_api/comfyui/stop")
    def api_comfyui_stop():
        touch_activity()
        rs.resolve_callable("stop_comfyui_core", stop_comfyui_core)(sync_before_stop=True)
        return get_gpu_status_dict(model_id=rs.SELECTED_MODEL_ID)

    @app.post("/minimax_api/comfyui/free_vram")
    def api_comfyui_free_vram():
        touch_activity()
        ok, msg = rs.resolve_callable("free_comfyui_vram_core", free_comfyui_vram_core)()
        res = get_gpu_status_dict(model_id=rs.SELECTED_MODEL_ID)
        res["action_ok"] = ok
        res["action_message"] = msg
        return res

    @app.post("/minimax_api/comfyui/sync_drive")
    def api_comfyui_sync_drive():
        touch_activity()
        ok, msg = rs.resolve_callable("sync_comfyui_to_drive_core", sync_comfyui_to_drive_core)(quiet=False)
        res = get_gpu_status_dict(model_id=rs.SELECTED_MODEL_ID)
        res["action_ok"] = ok
        res["action_message"] = msg
        return res

    @app.post("/minimax_api/comfyui/download_weight")
    def api_comfyui_download_weight(req: ComfyWeightDownloadRequest):
        touch_activity()
        if not COMFYUI_STATE["is_downloading_weight"]:
            threading.Thread(
                target=rs.resolve_callable("download_comfyui_model_core", download_comfyui_model_core),
                args=(req.url_or_repo, req.filename, req.subfolder, req.hf_token),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=rs.SELECTED_MODEL_ID)

    @app.post("/minimax_api/comfyui/install_pack")
    def api_comfyui_install_pack(req: ComfyInstallPackRequest):
        touch_activity()
        if not COMFYUI_STATE["is_downloading_weight"] and not COMFYUI_STATE.get("installing_pack_id"):
            threading.Thread(
                target=rs.resolve_callable("install_comfyui_preset_pack_core", install_comfyui_preset_pack_core),
                args=(req.pack_id, req.auto_start, req.hf_token),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=rs.SELECTED_MODEL_ID)

    @app.get("/minimax_api/comfyui/output_media/{rel_path:path}")
    def api_comfyui_output_media(rel_path: str):
        touch_activity()
        comfy_root, persist_root, _, _, _ = rs.resolve_callable("_resolve_comfyui_persistence_paths", _resolve_comfyui_persistence_paths)()
        for base_out in (os.path.join(comfy_root, "output"), os.path.join(persist_root, "output")):
            if not os.path.isdir(base_out):
                continue
            cand = os.path.normpath(os.path.join(base_out, rel_path))
            if not os.path.abspath(cand).startswith(os.path.abspath(base_out)):
                raise HTTPException(status_code=400, detail="Invalid output path")
            if os.path.isfile(cand):
                safe_name = os.path.basename(cand)
                ext = os.path.splitext(safe_name)[1].lower()
                mime = (
                    "image/png" if ext == ".png"
                    else "image/jpeg" if ext in (".jpg", ".jpeg")
                    else "image/webp" if ext == ".webp"
                    else "image/gif" if ext == ".gif"
                    else "video/webm" if ext == ".webm"
                    else "video/mp4"
                )
                return FileResponse(cand, media_type=mime, filename=safe_name)
        raise HTTPException(status_code=404, detail="ComfyUI output file not found")

    # --------------------------------------------------------------------------
    # H3 Toybox 独立创作台与 ComfyUI 共享引擎接口 (/minimax_api/toybox/*)
    # --------------------------------------------------------------------------
    @app.get("/minimax_api/toybox/status")
    def api_toybox_status():
        return {"ok": True, "toybox": get_toybox_status_dict(), "comfyui": get_comfyui_status_dict()}

    @app.post("/minimax_api/toybox/prepare")
    def api_toybox_prepare(req: ToyboxPrepareRequest):
        touch_activity()
        if not TOYBOX_STATE.get("is_preparing_assets") and not COMFYUI_STATE.get("is_downloading_weight"):
            threading.Thread(
                target=rs.resolve_callable("prepare_toybox_assets_core", prepare_toybox_assets_core),
                args=(req.include_rewriter, req.include_image, req.hf_token, req.auto_start),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=rs.SELECTED_MODEL_ID)

    @app.post("/minimax_api/toybox/start")
    def api_toybox_start(req: ToyboxStartRequest):
        touch_activity()
        if not TOYBOX_STATE.get("is_starting"):
            threading.Thread(
                target=rs.resolve_callable("start_toybox_core", start_toybox_core),
                args=(req.unload_existing, req.force_restart, req.hf_token, req.save_api_keys),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=rs.SELECTED_MODEL_ID)

    @app.post("/minimax_api/toybox/stop")
    def api_toybox_stop(req: ToyboxStopRequest):
        touch_activity()
        rs.resolve_callable("stop_toybox_core", stop_toybox_core)(stop_comfyui_too=req.stop_comfyui_too)
        return get_gpu_status_dict(model_id=rs.SELECTED_MODEL_ID)
