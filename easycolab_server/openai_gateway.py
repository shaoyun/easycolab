"""EasyColab 服务端模块: openai_gateway."""

from PIL import Image
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from typing import Any, Dict, List, Optional, Sequence, Tuple
import base64
import io
import json
import os
import re
import secrets
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid

from . import runtime_state as rs
from .runtime_state import torch
from .llm_runtime import _IncrementalQwenStreamParser, _count_bundle_tokens, _extract_openai_chat_media_inputs, _extract_qwen_tool_calls, _get_qwen_llm_bundle, _normalize_hf_tools, _normalize_to_hf_messages_and_images, _prepare_qwen_streamer_generation
from .media_pipeline import MAX_REF2VA_AUDIOS, MAX_REF2VA_IMAGES, MAX_REF2VA_VIDEOS, _align_minimax_h3_frame_count, _run_async_generation_job, run_minimax_h3_image_inference
from .model_catalog import DEFAULT_OUTPUT_DIR, MODEL_CATALOG, _get_custom_models_public_list, _normalize_builtin_model_alias
from .runtime_state import GPU_LOCK, JOBS, LOADED_RUNTIMES, _JOB_CANCEL_EVENTS, _get_valid_api_tokens, _load_pil_from_url_or_b64, torch, touch_activity
from .schemas import GenerateVideoRequest
from .telemetry_service import _ActiveApiStreamGuard

# ==============================================================================
# 标准对外网关：OpenAI (/v1/models, /v1/chat/completions, /v1/images/*, /v1/videos*)
#            与 Anthropic (/v1/messages, /v1/messages/count_tokens) 双协议兼容引擎
# ==============================================================================
_INFERENCE_EXACT_PATHS = {
    "/models", "/chat/completions", "/images/generations", "/images/edits", "/videos",
    "/messages/count_tokens", "/messages",
    "/v1/models", "/v1/chat/completions", "/v1/images/generations", "/v1/images/edits", "/v1/videos",
    "/v1/messages/count_tokens", "/v1/messages",
    "/v1/v1/messages/count_tokens", "/v1/v1/messages",
}

_INFERENCE_REGEX_LIST = (
    re.compile(r"^(?:/v1)?/models/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)?$"),
    re.compile(r"^(?:/v1)?/videos/[A-Za-z0-9_-]+(?:/content)?$"),
    re.compile(r"^(?:/v1)?/images/files/[A-Za-z0-9._-]+$"),
)

def _is_inference_api_path(path: str) -> bool:
    if not path or ".." in path.split("/"):
        return False
    if path in _INFERENCE_EXACT_PATHS:
        return True
    return any(pat.match(path) for pat in _INFERENCE_REGEX_LIST)

def _extract_api_token_from_request(request: Request, allow_query_token: bool = False) -> str:
    """提取 API Token（优先 Bearer / x-api-key，忽略 Basic 登录头以避免与登录认证冲突）。"""
    auth_hdr = (request.headers.get("authorization") or "").strip()
    bearer_tok = auth_hdr[7:].strip() if auth_hdr.lower().startswith("bearer ") else ""
    x_api_tok = (request.headers.get("x-api-key") or "").strip()
    x_goog_tok = (request.headers.get("x-goog-api-key") or "").strip()
    provided = bearer_tok or x_api_tok or x_goog_tok
    if not provided and allow_query_token:
        provided = (
            request.query_params.get("api_key")
            or request.query_params.get("token")
            or request.query_params.get("key")
            or ""
        ).strip()
    return provided

def _verify_external_api_auth(
    request: Request,
    protocol: str = "openai",
    allow_query_token: bool = False,
) -> Optional[JSONResponse]:
    """同时支持 Authorization: Bearer <key> 与 x-api-key: <key> 鉴权（不触发 Basic 弹窗）。"""
    valid_tokens = _get_valid_api_tokens()
    provided = _extract_api_token_from_request(request, allow_query_token=allow_query_token)
    if provided and any(
        secrets.compare_digest(provided.encode("utf-8"), exp.encode("utf-8"))
        for exp in valid_tokens
    ):
        return None
    if protocol == "anthropic":
        return JSONResponse(
            status_code=401,
            content={"type": "error", "error": {"type": "authentication_error", "message": "Invalid API Key"}},
        )
    return JSONResponse(
        status_code=401,
        content={"error": {"message": "Invalid API Key", "type": "invalid_request_error", "code": "invalid_api_key"}},
    )

_OPENAI_MODEL_ALIASES: Dict[str, str] = {
    "minimax-h3": "minimax_h3",
    "minimax_h3": "minimax_h3",
    "minimaxai/minimax-h3": "minimax_h3",
    "minimax-h3-fl2va": "minimax_h3",
    "sora-2": "minimax_h3",
    "minimax-h3-ref2va": "minimax_h3_ref2va",
    "minimax_h3_ref2va": "minimax_h3_ref2va",
    "minimaxai/minimax-h3-ref2va": "minimax_h3_ref2va",
    "ref2va": "minimax_h3_ref2va",
    "minimax-h3-image": "minimax_h3_image",
    "minimax_h3_image": "minimax_h3_image",
    "minimaxai/minimax-h3-image": "minimax_h3_image",
    "qwen-image-2.1": "qwen_image_21",
    "qwen_image_21": "qwen_image_21",
    "qwen/qwen-image-2.1": "qwen_image_21",
    "qwen-image": "qwen_image_21",
    "z-image-turbo": "z_image_turbo",
    "z_image_turbo": "z_image_turbo",
    "tongyi-mai/z-image-turbo": "z_image_turbo",
    "z-image": "z_image_turbo",
    "qwen3.8-27b": "qwen38_27b",
    "qwen/qwen3.8-27b": "qwen38_27b",
    "qwen38_27b": "qwen38_27b",
    "unsloth/qwen3.8-27b-gguf": "qwen38_27b_gguf",
    "qwen3.8-27b-ud-q4_k_m": "qwen38_27b_gguf",
    "qwen38_27b_gguf": "qwen38_27b_gguf",
}

def _is_video_modality(mod: Optional[str]) -> bool:
    return str(mod or "").strip().lower() in ("video", "video_audio")

def _resolve_openai_model_id(
    raw_model: Optional[str],
    expected_modality: Optional[str] = None,
) -> Optional[str]:
    """将 OpenAI 请求中的 model 字符串解析为 MODEL_CATALOG 内部 model_id。"""
    raw = (raw_model or "").strip()
    if raw:
        norm = _normalize_builtin_model_alias(raw)
        if norm and norm in MODEL_CATALOG:
            return norm
        low = raw.lower()
        if low in _OPENAI_MODEL_ALIASES:
            return _OPENAI_MODEL_ALIASES[low]
        for mid, spec in MODEL_CATALOG.items():
            if low in (
                mid.lower(),
                str(spec.get("repo_id", "")).lower(),
                str(spec.get("name", "")).lower(),
            ):
                return mid
        for cm in _get_custom_models_public_list():
            if low in (cm["id"].lower(), cm["repo_id"].lower(), cm["name"].lower()):
                return cm["id"]
        # 模糊匹配同类模态关键字（兼容部分客户端传入自定义显示名）
        if "ref2va" in low:
            return "minimax_h3_ref2va"
        if "minimax" in low and "image" in low:
            return "minimax_h3_image"
        if "minimax" in low or "h3" in low:
            return "minimax_h3"
        if "z-image" in low or "z_image" in low:
            return "z_image_turbo"
        if "qwen" in low and "image" in low:
            return "qwen_image_21"
        if "qwen" in low:
            return "qwen38_27b_gguf" if "gguf" in low else "qwen38_27b"
        return None

    if expected_modality:
        preferred_order = (
            ("minimax_h3_ref2va", "minimax_h3")
            if expected_modality == "video"
            else ("minimax_h3_image", "z_image_turbo", "qwen_image_21")
            if expected_modality == "image"
            else ("qwen38_27b_gguf", "qwen38_27b")
        )
        for cand in preferred_order:
            if cand in LOADED_RUNTIMES:
                return cand
        for mid, rt in LOADED_RUNTIMES.items():
            rt_mod = MODEL_CATALOG.get(mid, {}).get("modality") or rt.get("modality")
            if (expected_modality == "video" and _is_video_modality(rt_mod)) or rt_mod == expected_modality:
                return mid
        return preferred_order[0]
    return None

def _build_openai_model_entries() -> List[Dict[str, Any]]:
    """构建 /v1/models 与 /v1/models/{id} 使用的标准模型列表。"""
    now_ts = int(time.time())
    any_qwen_loaded = ("qwen38_27b" in LOADED_RUNTIMES) or ("qwen38_27b_gguf" in LOADED_RUNTIMES)
    entries = [
        {
            "id": "unsloth/Qwen3.8-27B-GGUF",
            "object": "model",
            "created": now_ts,
            "owned_by": "unsloth",
            "loaded": "qwen38_27b_gguf" in LOADED_RUNTIMES,
            "modality": "llm",
            "supported_endpoints": ["/v1/chat/completions", "/v1/messages"],
        },
        {
            "id": "Qwen3.8-27B-UD-Q4_K_M",
            "object": "model",
            "created": now_ts,
            "owned_by": "unsloth",
            "loaded": "qwen38_27b_gguf" in LOADED_RUNTIMES,
            "modality": "llm",
            "supported_endpoints": ["/v1/chat/completions", "/v1/messages"],
        },
        {
            "id": "Qwen/Qwen3.8-27B",
            "object": "model",
            "created": now_ts,
            "owned_by": "qwen",
            "loaded": any_qwen_loaded,
            "modality": "llm",
            "supported_endpoints": ["/v1/chat/completions", "/v1/messages"],
        },
        {
            "id": "qwen3.8-27b",
            "object": "model",
            "created": now_ts,
            "owned_by": "qwen",
            "loaded": any_qwen_loaded,
            "modality": "llm",
            "supported_endpoints": ["/v1/chat/completions", "/v1/messages"],
        },
        {
            "id": "MiniMaxAI/MiniMax-H3-Image",
            "object": "model",
            "created": now_ts,
            "owned_by": "minimax",
            "loaded": "minimax_h3_image" in LOADED_RUNTIMES,
            "modality": "image",
            "supported_endpoints": ["/v1/images/generations", "/v1/images/edits", "/v1/chat/completions"],
        },
        {
            "id": "Qwen/Qwen-Image-2.1",
            "object": "model",
            "created": now_ts,
            "owned_by": "qwen",
            "loaded": "qwen_image_21" in LOADED_RUNTIMES,
            "modality": "image",
            "supported_endpoints": ["/v1/images/generations", "/v1/images/edits", "/v1/chat/completions"],
        },
        {
            "id": "Tongyi-MAI/Z-Image-Turbo",
            "object": "model",
            "created": now_ts,
            "owned_by": "tongyi",
            "loaded": "z_image_turbo" in LOADED_RUNTIMES,
            "modality": "image",
            "supported_endpoints": ["/v1/images/generations", "/v1/images/edits", "/v1/chat/completions"],
        },
        {
            "id": "MiniMaxAI/MiniMax-H3",
            "object": "model",
            "created": now_ts,
            "owned_by": "minimax",
            "loaded": "minimax_h3" in LOADED_RUNTIMES,
            "modality": "video",
            "supported_endpoints": ["/v1/videos", "/v1/chat/completions"],
            "task_modes": ["t2va", "fl2va", "ref2va"],
        },
        {
            "id": "MiniMaxAI/MiniMax-H3-Ref2VA",
            "object": "model",
            "created": now_ts,
            "owned_by": "minimax",
            "loaded": "minimax_h3_ref2va" in LOADED_RUNTIMES,
            "modality": "video",
            "supported_endpoints": ["/v1/videos", "/v1/chat/completions"],
            "task_modes": ["ref2va", "t2va", "fl2va"],
            "max_ref_images": MAX_REF2VA_IMAGES,
            "max_ref_videos": MAX_REF2VA_VIDEOS,
            "max_ref_audios": MAX_REF2VA_AUDIOS,
        },
    ]
    for cm in _get_custom_models_public_list():
        mod = cm.get("modality") or "llm"
        eps = (
            ["/v1/images/generations", "/v1/images/edits", "/v1/chat/completions"]
            if mod == "image"
            else ["/v1/videos", "/v1/chat/completions"]
            if _is_video_modality(mod)
            else ["/v1/chat/completions", "/v1/messages"]
        )
        entries.append({
            "id": cm["repo_id"],
            "object": "model",
            "created": now_ts,
            "owned_by": "custom-hf",
            "loaded": cm["id"] in LOADED_RUNTIMES,
            "modality": mod,
            "supported_endpoints": eps,
        })
    return entries

def _parse_openai_size(size_str: Optional[str], default_w: int, default_h: int) -> Tuple[int, int]:
    """解析 OpenAI size 参数（支持 WxH、W*H、16:9 / 9:16 / 1:1 / 720p）。"""
    s = (size_str or "").strip().lower()
    if not s:
        return default_w, default_h
    preset_map = {
        "16:9": (832, 480),
        "9:16": (480, 832),
        "1:1": (640, 640) if default_w < 1000 else (1024, 1024),
        "720p": (1280, 720),
        "480p": (832, 480),
    }
    if s in preset_map:
        return preset_map[s]
    m = re.match(r"^\s*(\d{2,4})\s*[x×*]\s*(\d{2,4})\s*$", s)
    if m:
        w = max(256, min(4096, int(m.group(1))))
        h = max(256, min(4096, int(m.group(2))))
        return w, h
    return default_w, default_h

def _seconds_to_minimax_frames(seconds_val: Any, fps: int = 24) -> int:
    try:
        secs = float(str(seconds_val).strip().rstrip("s"))
    except Exception:
        secs = 5.0
    secs = max(1.0, min(16.0, secs))
    raw_frames = int(round(secs * max(16, int(fps))))
    aligned = _align_minimax_h3_frame_count(max(90, raw_frames))
    return min(362, aligned)

def _format_openai_video_object(
    job_id: str,
    job: Dict[str, Any],
    base_url: str = "",
    token_q: str = "",
) -> Dict[str, Any]:
    """将内部 JOBS[job_id] 映射为标准 OpenAI Video 对象 (/v1/videos)。"""
    raw_status = str(job.get("status") or "queued")
    status_map = {
        "queued": "queued",
        "running": "in_progress",
        "completed": "completed",
        "failed": "failed",
    }
    oai_status = status_map.get(raw_status, "queued")
    raw_prog = float(job.get("progress") or 0.0)
    progress_pct = 100 if oai_status == "completed" else max(0, min(99, int(round(raw_prog * 100))))

    vid_id = job.get("openai_video_id") or f"video_{job_id}"
    model_id = str(job.get("model_id") or "minimax_h3")
    model_name = MODEL_CATALOG.get(model_id, {}).get("repo_id") or model_id

    content_rel = f"/v1/videos/{vid_id}/content"
    if token_q:
        content_rel = f"{content_rel}?api_key={urllib.parse.quote(token_q, safe='')}"
    content_url = f"{base_url.rstrip('/')}{content_rel}" if base_url else content_rel

    err_obj = None
    if oai_status == "failed":
        err_obj = {
            "code": "video_generation_failed",
            "message": str(job.get("message") or "Video generation failed"),
        }

    created_at = int(job.get("created_at") or job.get("started_at") or time.time())
    completed_at = int(job["completed_at"]) if job.get("completed_at") else None
    if oai_status == "completed" and completed_at is None:
        completed_at = int(created_at + float(job.get("elapsed") or 0.0))

    return {
        "id": vid_id,
        "object": "video",
        "created_at": created_at,
        "status": oai_status,
        "model": model_name,
        "progress": progress_pct,
        "seconds": str(job.get("seconds") or round(float(job.get("num_frames") or 124) / float(job.get("fps") or 24), 1)),
        "size": str(job.get("size") or "832x480"),
        "task_mode": job.get("task_mode") or "t2va",
        "seed": job.get("seed"),
        "elapsed": job.get("elapsed", 0.0),
        "message": job.get("message", ""),
        "filename": job.get("filename"),
        "url": content_url if oai_status == "completed" else None,
        "completed_at": completed_at,
        "expires_at": None,
        "error": err_obj,
    }

def _find_job_by_openai_video_id(video_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    clean_id = (video_id or "").strip()
    if clean_id in JOBS:
        return clean_id, JOBS[clean_id]
    stripped = clean_id[len("video_"):] if clean_id.startswith("video_") else clean_id
    if stripped in JOBS:
        return stripped, JOBS[stripped]
    for jid, j in JOBS.items():
        if j.get("openai_video_id") == clean_id:
            return jid, j
    return None

async def _coerce_media_item_to_b64_or_path(
    item: Any,
    kind: str = "image",
    cleanup_paths: Optional[List[str]] = None,
) -> Optional[str]:
    """将 UploadFile、URL、data URI 或裸 base64 统一转换为可读字符串（图片返回 base64，音视频返回临时文件路径或 base64）。"""
    if item is None:
        return None
    raw_bytes: Optional[bytes] = None
    suffix = ".png" if kind == "image" else (".mp4" if kind == "video" else ".wav")

    if hasattr(item, "read") and callable(item.read):
        maybe = item.read()
        raw_bytes = await maybe if hasattr(maybe, "__await__") else maybe
        fname = getattr(item, "filename", "") or ""
        ext = os.path.splitext(fname)[1].lower()
        if ext:
            suffix = ext
    elif isinstance(item, str):
        s = item.strip()
        if not s:
            return None
        if s.startswith("http://") or s.startswith("https://"):
            req = urllib.request.Request(s, headers={"User-Agent": "EasyColab-API-Gateway/2.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_bytes = resp.read()
        else:
            return s
    else:
        return None

    if not raw_bytes:
        return None
    if kind == "image":
        return base64.b64encode(raw_bytes).decode("ascii")
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix=f"easycolab_api_{kind}_")
    with os.fdopen(fd, "wb") as tf:
        tf.write(raw_bytes)
    if cleanup_paths is not None:
        cleanup_paths.append(tmp_path)
    return tmp_path

def _run_loaded_image_generation(
    img_target_id: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    n: int = 1,
    seed: Optional[int] = None,
    negative_prompt: str = "",
    ref_images: Optional[List[Image.Image]] = None,
    denoising_strength: float = 0.70,
    include_b64: bool = True,
) -> List[Dict[str, Any]]:
    """统一执行已加载图像模型的文生图 / 图像编辑推理，并落盘保存 PNG。"""
    img_pipe = LOADED_RUNTIMES[img_target_id]["pipeline"]
    os.makedirs(rs.DEFAULT_OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_seed = int(seed) if (seed is not None and int(seed) >= 0) else None
    mode_tag = "edit" if ref_images else "t2i"
    results: List[Dict[str, Any]] = []

    with GPU_LOCK:
        for idx in range(max(1, min(4, int(n)))):
            cur_seed = (base_seed + idx) if base_seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
            if img_target_id == "minimax_h3_image":
                img = rs.resolve_callable("run_minimax_h3_image_inference", run_minimax_h3_image_inference)(
                    active_pipe=img_pipe,
                    prompt=prompt,
                    width=width,
                    height=height,
                    seed=cur_seed,
                    ref_images=ref_images if ref_images else None,
                )
            elif img_target_id == "z_image_turbo":
                call_kw: Dict[str, Any] = {
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": steps,
                    "seed": cur_seed,
                    "rand_device": "cuda",
                }
                if negative_prompt:
                    call_kw["negative_prompt"] = negative_prompt
                if ref_images:
                    call_kw["input_image"] = ref_images[0].resize((width, height))
                    call_kw["denoising_strength"] = max(0.15, min(1.0, float(denoising_strength or 0.70)))
                img = img_pipe(**call_kw)
            elif img_target_id == "qwen_image_21":
                img = img_pipe(
                    prompt=prompt,
                    negative_prompt=(negative_prompt or "").strip() or " ",
                    edit_image=ref_images if ref_images else None,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    seed=cur_seed,
                )
            else:
                gen = torch.Generator(device="cuda").manual_seed(cur_seed)
                out_res = img_pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    generator=gen,
                )
                img = out_res.images[0] if hasattr(out_res, "images") else out_res

            filename = f"{img_target_id}_{mode_tag}_{ts}_{idx}_seed{cur_seed}.png"
            out_path = os.path.join(rs.DEFAULT_OUTPUT_DIR, filename)
            img.save(out_path, format="PNG")
            item: Dict[str, Any] = {"filename": filename, "seed": cur_seed}
            if include_b64:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                item["b64_json"] = base64.b64encode(buf.getvalue()).decode("ascii")
            results.append(item)
            touch_activity()
    return results


def attach_openai_routes(app: FastAPI):
    """挂载 OpenAI (/v1/*) 与 Anthropic (/v1/messages*) 兼容网关及推理鉴权/CORS 中间件。"""

    @app.middleware("http")
    async def _inference_cors_middleware(request: Request, call_next):
        if not _is_inference_api_path(request.url.path):
            return await call_next(request)
        if request.method.upper() == "OPTIONS":
            return JSONResponse(
                status_code=204,
                content=None,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type, X-API-Key, Anthropic-Version, OpenAI-Beta",
                    "Access-Control-Max-Age": "86400",
                },
            )
        response = await call_next(request)
        response.headers.setdefault("Access-Control-Allow-Origin", "*")
        response.headers.setdefault("Access-Control-Expose-Headers", "Content-Disposition")
        return response


    # --------------------------------------------------------------------------
    # 1. OpenAI 兼容协议: /v1/models, /v1/chat/completions, /v1/images/*, /v1/videos*
    # --------------------------------------------------------------------------
    @app.get("/v1/models")
    @app.get("/models")
    def openai_list_models(request: Request):
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()
        return {"object": "list", "data": _build_openai_model_entries()}

    @app.get("/v1/models/{model_id:path}")
    @app.get("/models/{model_id:path}")
    def openai_retrieve_model(model_id: str, request: Request):
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()
        clean_id = (model_id or "").strip().strip("/")
        for item in _build_openai_model_entries():
            if item["id"].lower() == clean_id.lower():
                return item
        resolved = _resolve_openai_model_id(clean_id)
        if resolved and resolved in MODEL_CATALOG:
            spec = MODEL_CATALOG[resolved]
            return {
                "id": clean_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "easycolab",
                "loaded": resolved in LOADED_RUNTIMES,
                "modality": spec.get("modality"),
            }
        return JSONResponse(
            status_code=404,
            content={"error": {"message": f"Model '{clean_id}' not found", "type": "invalid_request_error", "code": "model_not_found"}},
        )

    def _build_public_base_url(request: Request) -> str:
        proto = (request.headers.get("x-forwarded-proto") or request.url.scheme or "https").split(",")[0].strip()
        host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc or "").split(",")[0].strip()
        return f"{proto}://{host}" if host else ""

    async def _handle_openai_chat_media_model(
        request: Request,
        body: Dict[str, Any],
        req_model: str,
        resolved_model_id: str,
        stream: bool,
    ):
        # (shared state on runtime_state)
        spec = MODEL_CATALOG.get(resolved_model_id, {})
        modality = spec.get("modality")
        prompt_txt, img_srcs, vid_srcs, aud_srcs = _extract_openai_chat_media_inputs(body.get("messages", []))
        if not prompt_txt:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Prompt message cannot be empty", "type": "invalid_request_error"}},
            )

        tok = _extract_api_token_from_request(request, allow_query_token=False)
        base_url = _build_public_base_url(request)
        tok_qs = f"?api_key={urllib.parse.quote(tok, safe='')}" if tok else ""
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
        created_ts = int(time.time())

        if modality == "image":
            if resolved_model_id not in LOADED_RUNTIMES:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": f"{spec.get('name', req_model)} 尚未加载至 Colab GPU 显存，请先点击「加载到显存」。",
                            "type": "server_error",
                            "code": "model_not_loaded",
                        }
                    },
                )
            rs.API_REQUEST_COUNT += 1
            default_w, default_h = (2048, 2048) if resolved_model_id == "minimax_h3_image" else (1024, 1024)
            w, h = _parse_openai_size(body.get("size"), default_w, default_h)
            steps = int(body.get("steps") or (8 if resolved_model_id == "z_image_turbo" else 40))
            n = max(1, min(4, int(body.get("n") or 1)))
            ref_pils = [im for im in (_load_pil_from_url_or_b64(s) for s in img_srcs) if im is not None]
            gen_items = rs.resolve_callable("_run_loaded_image_generation", _run_loaded_image_generation)(
                img_target_id=resolved_model_id,
                prompt=prompt_txt,
                width=w,
                height=h,
                steps=steps,
                n=n,
                seed=body.get("seed"),
                negative_prompt=str(body.get("negative_prompt") or ""),
                ref_images=ref_pils or None,
                denoising_strength=float(body.get("denoising_strength") or 0.70),
                include_b64=False,
            )
            lines = []
            for idx, it in enumerate(gen_items, start=1):
                img_url = f"{base_url}/v1/images/files/{it['filename']}{tok_qs}"
                lines.append(f"![Generated Image {idx}]({img_url})\n\n- 文件: `{it['filename']}` · Seed: `{it['seed']}`\n- 下载地址: {img_url}")
            answer_md = "\n\n".join(lines)
        else:
            # 视频模型（MiniMax-H3 / MiniMax-H3-Ref2VA）
            active_vid_id = resolved_model_id
            if active_vid_id not in LOADED_RUNTIMES:
                if active_vid_id == "minimax_h3_ref2va" and "minimax_h3" in LOADED_RUNTIMES:
                    active_vid_id = "minimax_h3"
                elif active_vid_id == "minimax_h3" and "minimax_h3_ref2va" in LOADED_RUNTIMES:
                    active_vid_id = "minimax_h3_ref2va"
            if active_vid_id not in LOADED_RUNTIMES and rs.LOADED_MODEL_ID != active_vid_id:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": f"{spec.get('name', req_model)} 尚未加载至 Colab GPU 显存，请先点击「加载到显存」。",
                            "type": "server_error",
                            "code": "model_not_loaded",
                        }
                    },
                )
            rs.API_REQUEST_COUNT += 1
            cleanup_paths: List[str] = []
            ref_img_b64s: List[str] = []
            for src in img_srcs[:MAX_REF2VA_IMAGES]:
                v = await _coerce_media_item_to_b64_or_path(src, kind="image", cleanup_paths=cleanup_paths)
                if v:
                    ref_img_b64s.append(v)
            ref_vid_items: List[str] = []
            for src in vid_srcs[:MAX_REF2VA_VIDEOS]:
                v = await _coerce_media_item_to_b64_or_path(src, kind="video", cleanup_paths=cleanup_paths)
                if v:
                    ref_vid_items.append(v)
            ref_aud_items: List[str] = []
            for src in aud_srcs[:MAX_REF2VA_AUDIOS]:
                v = await _coerce_media_item_to_b64_or_path(src, kind="audio", cleanup_paths=cleanup_paths)
                if v:
                    ref_aud_items.append(v)

            w, h = _parse_openai_size(body.get("size"), 832, 480)
            fps = int(body.get("fps") or 24)
            n_frames = int(body["num_frames"]) if body.get("num_frames") else _seconds_to_minimax_frames(body.get("seconds") or 5, fps=fps)
            task_mode = str(body.get("task_mode") or "").strip().lower()
            if not task_mode:
                if ref_vid_items or ref_aud_items or len(ref_img_b64s) > 2 or resolved_model_id == "minimax_h3_ref2va":
                    task_mode = "ref2va"
                elif ref_img_b64s:
                    task_mode = "fl2va"
                else:
                    task_mode = "t2va"

            gen_req = GenerateVideoRequest(
                model_id=active_vid_id,
                task_mode=task_mode,
                visual_prompt=prompt_txt,
                soundscape_prompt=str(body.get("soundscape_prompt") or ""),
                music_prompt=str(body.get("music_prompt") or ""),
                first_frame_base64=ref_img_b64s[0] if (task_mode == "fl2va" and len(ref_img_b64s) >= 1) else None,
                last_frame_base64=ref_img_b64s[1] if (task_mode == "fl2va" and len(ref_img_b64s) >= 2) else None,
                ref_images_base64=ref_img_b64s or None,
                ref_videos_base64=ref_vid_items or None,
                ref_audios_base64=ref_aud_items or None,
                width=w,
                height=h,
                num_frames=n_frames,
                num_inference_steps=int(body.get("steps") or body.get("num_inference_steps") or 35),
                fps=fps,
                seed=int(body["seed"]) if body.get("seed") is not None else 42,
                randomize_seed=(body.get("seed") is None) if body.get("randomize_seed") is None else bool(body.get("randomize_seed")),
            )
            job_id = uuid.uuid4().hex[:12]
            vid_id = f"video_{job_id}"
            _JOB_CANCEL_EVENTS[job_id] = threading.Event()
            JOBS[job_id] = {
                "job_id": job_id,
                "openai_video_id": vid_id,
                "model_id": active_vid_id,
                "status": "queued",
                "progress": 0.0,
                "message": "任务已通过 OpenAI Chat 接口提交...",
                "seed": gen_req.seed,
                "created_at": time.time(),
                "started_at": time.time(),
                "elapsed": 0.0,
                "size": f"{w}x{h}",
                "seconds": round(n_frames / float(fps), 1),
                "num_frames": n_frames,
                "fps": fps,
                "task_mode": task_mode,
                "filename": None,
                "video_url": None,
                "media_type": "video",
                "completed_count": 0,
                "completed_videos": [],
                "_temp_cleanup_paths": cleanup_paths,
            }
            worker = threading.Thread(
                target=rs.resolve_callable("_run_async_generation_job", _run_async_generation_job),
                args=(job_id, gen_req),
                daemon=True,
            )
            worker.start()

            if stream:
                def _video_chat_sse():
                    first = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req_model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
                    last_sent = ""
                    while worker.is_alive():
                        job = JOBS.get(job_id, {})
                        msg = str(job.get("message") or "")
                        if msg and msg != last_sent:
                            last_sent = msg
                            chunk = {
                                "id": cmpl_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": req_model,
                                "choices": [{"index": 0, "delta": {"content": f"> {msg}\n"}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        else:
                            yield ": keepalive\n\n"
                        worker.join(timeout=3.0)

                    job = JOBS.get(job_id, {})
                    if job.get("status") == "completed":
                        v_url = f"{base_url}/v1/videos/{vid_id}/content{tok_qs}"
                        fin_txt = f"\n✅ 视频生成完成（Seed: `{job.get('seed')}` · 耗时: `{job.get('elapsed')}s`）\n\n[下载 / 播放 MP4 视频]({v_url})\n"
                    else:
                        fin_txt = f"\n❌ 生成失败: {job.get('message', '未知错误')}\n"
                    yield f"data: {json.dumps({'id': cmpl_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': req_model, 'choices': [{'index': 0, 'delta': {'content': fin_txt}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'id': cmpl_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': req_model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    _video_chat_sse(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
                )

            worker.join()
            job = JOBS.get(job_id, {})
            if job.get("status") != "completed":
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": str(job.get("message") or "Video generation failed"), "type": "server_error"}},
                )
            v_url = f"{base_url}/v1/videos/{vid_id}/content{tok_qs}"
            answer_md = f"✅ 视频生成完成（Seed: `{job.get('seed')}` · 耗时: `{job.get('elapsed')}s`）\n\n[下载 / 播放 MP4 视频]({v_url})"

        if stream:
            def _img_chat_sse():
                yield f"data: {json.dumps({'id': cmpl_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': req_model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': answer_md}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'id': cmpl_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': req_model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                _img_chat_sse(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )

        return {
            "id": cmpl_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": req_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer_md}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt_txt), "completion_tokens": len(answer_md), "total_tokens": len(prompt_txt) + len(answer_md)},
        }

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def openai_chat_completions(request: Request):
        # (shared state on runtime_state)
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()

        body = await request.json()
        req_model = str(body.get("model") or "Qwen/Qwen3.8-27B")
        stream = bool(body.get("stream", False))
        resolved_mid = _resolve_openai_model_id(req_model)
        resolved_mod = MODEL_CATALOG.get(resolved_mid, {}).get("modality") if resolved_mid else None
        if resolved_mid and (resolved_mod == "image" or _is_video_modality(resolved_mod)):
            return await _handle_openai_chat_media_model(request, body, req_model, resolved_mid, stream)

        bundle = _get_qwen_llm_bundle(preferred_model_id=req_model)
        if bundle is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Qwen3.8-27B / Qwen3.8-27B-GGUF 尚未加载至 Colab GPU 显存，请先在 Mac 客户端点击「加载到显存」。",
                        "type": "server_error",
                        "code": "model_not_loaded",
                    }
                },
            )

        rs.API_REQUEST_COUNT += 1
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 16384)

        # 支持通过 enable_thinking / chat_template_kwargs / reasoning_effort 控制思考模式与思考深度
        enable_thinking = True
        effort_val: Optional[str] = None
        ctk = body.get("chat_template_kwargs") if isinstance(body.get("chat_template_kwargs"), dict) else {}
        if "enable_thinking" in body:
            enable_thinking = bool(body["enable_thinking"])
        elif "enable_thinking" in ctk:
            enable_thinking = bool(ctk["enable_thinking"])
        raw_eff = str(body.get("reasoning_effort") or ctk.get("reasoning_effort") or "").strip().lower()
        if raw_eff in ("none", "disabled", "off"):
            enable_thinking = False
        elif raw_eff:
            effort_val = raw_eff

        temperature = float(body["temperature"]) if body.get("temperature") is not None else (1.0 if enable_thinking else 0.7)
        top_p = float(body["top_p"]) if body.get("top_p") is not None else (0.95 if enable_thinking else 0.80)
        top_k = int(body["top_k"]) if body.get("top_k") is not None else 20
        min_p = float(body["min_p"]) if body.get("min_p") is not None else 0.0
        presence_penalty = float(body["presence_penalty"]) if body.get("presence_penalty") is not None else (0.0 if enable_thinking else 1.5)
        repeat_penalty = float(body["repeat_penalty"]) if body.get("repeat_penalty") is not None else 1.0

        print(f"[OpenAI API] POST /v1/chat/completions | model={req_model} | backend={bundle.get('backend')} | stream={stream} | thinking={enable_thinking}")

        hf_messages, pil_images = _normalize_to_hf_messages_and_images(body.get("messages", []), protocol="openai")
        hf_tools = _normalize_hf_tools(body.get("tools"), protocol="openai")

        streamer, prompt_tokens, start_in_thinking, gen_thread, err_box, cancel_event = _prepare_qwen_streamer_generation(
            hf_messages=hf_messages,
            pil_images=pil_images,
            hf_tools=hf_tools,
            enable_thinking=enable_thinking,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            preferred_model_id=str(req_model),
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            reasoning_effort=effort_val,
        )

        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
        created_ts = int(time.time())

        if not stream:
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            think_parts: List[str] = []
            text_parts: List[str] = []
            out_chunks = 0
            with _ActiveApiStreamGuard():
                try:
                    for piece in streamer:
                        out_chunks += 1
                        if out_chunks % 64 == 0:
                            touch_activity()
                        for kind, val in parser.feed(piece):
                            if kind == "thinking":
                                think_parts.append(val)
                            elif kind == "text":
                                text_parts.append(val)
                finally:
                    cancel_event.set()
            tail_events, tool_calls = parser.finish()
            for kind, val in tail_events:
                if kind == "thinking":
                    think_parts.append(val)
                elif kind == "text":
                    text_parts.append(val)
            gen_thread.join(timeout=5.0)
            if err_box[0] is not None:
                return JSONResponse(status_code=500, content={"error": {"message": str(err_box[0]), "type": "server_error"}})

            full_thinking = "".join(think_parts).strip()
            full_text, extra_tools = _extract_qwen_tool_calls("".join(text_parts).strip())
            all_tools = tool_calls + extra_tools

            msg_obj: Dict[str, Any] = {"role": "assistant", "content": full_text if (full_text or not all_tools) else None}
            if full_thinking:
                msg_obj["reasoning_content"] = full_thinking
            if all_tools:
                msg_obj["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
                    }
                    for tc in all_tools
                ]

            final_prompt_tokens = int(getattr(streamer, "prompt_tokens", prompt_tokens) or prompt_tokens)
            comp_tokens = int(getattr(streamer, "generated_tokens", 0)) or (
                _count_bundle_tokens(bundle, full_thinking + full_text) if (full_thinking or full_text) else max(1, out_chunks)
            )
            finish_reason = "tool_calls" if all_tools else "stop"

            return {
                "id": cmpl_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": req_model,
                "choices": [{"index": 0, "message": msg_obj, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": final_prompt_tokens,
                    "completion_tokens": comp_tokens,
                    "total_tokens": final_prompt_tokens + comp_tokens,
                },
            }

        def _openai_sse_stream():
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            comp_chars = 0
            chunk_idx = 0
            with _ActiveApiStreamGuard():
                try:
                    first_chunk = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req_model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

                    for piece in streamer:
                        comp_chars += len(piece)
                        chunk_idx += 1
                        if chunk_idx % 64 == 0:
                            touch_activity()
                        for kind, val in parser.feed(piece):
                            if not val:
                                continue
                            delta_payload = {"reasoning_content": val} if kind == "thinking" else {"content": val}
                            chunk_obj = {
                                "id": cmpl_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": req_model,
                                "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n"

                    tail_events, tool_calls = parser.finish()
                    for kind, val in tail_events:
                        if not val:
                            continue
                        delta_payload = {"reasoning_content": val} if kind == "thinking" else {"content": val}
                        chunk_obj = {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": req_model,
                            "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n"

                    gen_thread.join(timeout=5.0)
                    if getattr(streamer, "generated_tokens", 0) > 0:
                        print(f"[OpenAI API] 流式完成 | {streamer.generated_tokens} tokens | {streamer.tok_per_sec} tok/s")
                finally:
                    cancel_event.set()

            if tool_calls:
                tc_deltas = [
                    {
                        "index": idx,
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
                    }
                    for idx, tc in enumerate(tool_calls)
                ]
                tc_chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {"tool_calls": tc_deltas}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"

            final_prompt_toks = int(getattr(streamer, "prompt_tokens", prompt_tokens) or prompt_tokens)
            est_out_toks = int(getattr(streamer, "generated_tokens", 0)) or max(1, comp_chars // 3)
            final_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}],
                "usage": {
                    "prompt_tokens": final_prompt_toks,
                    "completion_tokens": est_out_toks,
                    "total_tokens": final_prompt_toks + est_out_toks,
                },
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _openai_sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    async def _openai_images_common(request: Request, is_edit: bool = False):
        """统一处理 OpenAI /v1/images/generations 与 /v1/images/edits（支持 JSON 与 multipart/form-data）。"""
        # (shared state on runtime_state)
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()

        content_type = (request.headers.get("content-type") or "").lower()
        body: Dict[str, Any] = {}
        ref_pils: List[Image.Image] = []
        if "multipart/form-data" in content_type:
            form = await request.form()
            for k in form.keys():
                vals = form.getlist(k)
                if k in ("image", "image[]", "images", "ref_images"):
                    for v in vals:
                        if hasattr(v, "read") and callable(v.read):
                            raw_b = await v.read()
                            if raw_b:
                                ref_pils.append(Image.open(io.BytesIO(raw_b)).convert("RGB"))
                        elif isinstance(v, str) and v.strip():
                            im = _load_pil_from_url_or_b64(v.strip())
                            if im is not None:
                                ref_pils.append(im)
                elif vals:
                    body[k] = vals[0]
        else:
            body = await request.json()
            raw_imgs: List[Any] = []
            for key_name in ("image", "images", "ref_images"):
                val = body.get(key_name)
                if isinstance(val, list):
                    raw_imgs.extend(val)
                elif val:
                    raw_imgs.append(val)
            for s in raw_imgs:
                if isinstance(s, str) and s.strip():
                    im = _load_pil_from_url_or_b64(s.strip())
                    if im is not None:
                        ref_pils.append(im)

        req_model_raw = str(body.get("model") or "").strip()
        img_target_id = _resolve_openai_model_id(req_model_raw, expected_modality="image")
        if not img_target_id or img_target_id not in LOADED_RUNTIMES or MODEL_CATALOG.get(img_target_id, {}).get("modality") != "image":
            # 若调用方未指定具体图像模型（或指定名未命中），回退到当前已加载的图像模型
            fallback_img = None
            if not req_model_raw:
                for cand in ("minimax_h3_image", "z_image_turbo", "qwen_image_21"):
                    if cand in LOADED_RUNTIMES:
                        fallback_img = cand
                        break
            img_target_id = fallback_img if (fallback_img and fallback_img in LOADED_RUNTIMES) else img_target_id

        if not img_target_id or img_target_id not in LOADED_RUNTIMES:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "请先在客户端加载 MiniMax-H3-Image、Qwen-Image-2.1 或 Z-Image-Turbo 至显存。", "type": "server_error", "code": "model_not_loaded"}},
            )

        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "prompt is required", "type": "invalid_request_error"}},
            )
        if is_edit and not ref_pils:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "At least one reference image is required for /v1/images/edits", "type": "invalid_request_error"}},
            )

        rs.API_REQUEST_COUNT += 1
        default_w, default_h = (2048, 2048) if img_target_id == "minimax_h3_image" else (1024, 1024)
        w, h = _parse_openai_size(str(body.get("size") or ""), default_w, default_h)
        try:
            n = max(1, min(4, int(body.get("n") or 1)))
        except Exception:
            n = 1
        default_steps = 6 if img_target_id == "minimax_h3_image" else (8 if img_target_id == "z_image_turbo" else 40)
        steps = int(body.get("steps") or body.get("num_inference_steps") or default_steps)
        response_format = str(body.get("response_format") or "b64_json").strip().lower()
        need_b64 = response_format != "url"

        seed_val = body.get("seed")
        try:
            parsed_seed = int(seed_val) if seed_val is not None and str(seed_val).strip() != "" else None
        except Exception:
            parsed_seed = None

        gen_items = rs.resolve_callable("_run_loaded_image_generation", _run_loaded_image_generation)(
            img_target_id=img_target_id,
            prompt=prompt,
            width=w,
            height=h,
            steps=steps,
            n=n,
            seed=parsed_seed,
            negative_prompt=str(body.get("negative_prompt") or ""),
            ref_images=ref_pils or None,
            denoising_strength=float(body.get("denoising_strength") or 0.70),
            include_b64=need_b64,
        )

        tok = _extract_api_token_from_request(request, allow_query_token=False)
        base_url = _build_public_base_url(request)
        tok_qs = f"?api_key={urllib.parse.quote(tok, safe='')}" if tok else ""
        data_out: List[Dict[str, Any]] = []
        for it in gen_items:
            file_url = f"{base_url}/v1/images/files/{it['filename']}{tok_qs}" if base_url else f"/v1/images/files/{it['filename']}{tok_qs}"
            if response_format == "url":
                data_out.append({"url": file_url, "revised_prompt": prompt})
            else:
                data_out.append({"b64_json": it["b64_json"], "url": file_url, "revised_prompt": prompt})
        return {"created": int(time.time()), "data": data_out}

    @app.post("/v1/images/generations")
    @app.post("/images/generations")
    async def openai_images_generations(request: Request):
        """OpenAI 标准文生图接口（支持 response_format=b64_json/url，兼容 JSON/seed/steps）"""
        return await _openai_images_common(request, is_edit=False)

    @app.post("/v1/images/edits")
    @app.post("/images/edits")
    async def openai_images_edits(request: Request):
        """OpenAI 标准图像编辑接口（支持 multipart 上传及 JSON URL/base64）"""
        return await _openai_images_common(request, is_edit=True)

    @app.get("/v1/images/files/{filename}")
    @app.get("/images/files/{filename}")
    def openai_download_image_file(filename: str, request: Request):
        auth_err = _verify_external_api_auth(request, protocol="openai", allow_query_token=True)
        if auth_err:
            return auth_err
        touch_activity()
        safe_name = os.path.basename(filename)
        if not safe_name or safe_name != filename or ".." in safe_name:
            raise HTTPException(status_code=400, detail="Invalid filename")
        out_root = os.path.realpath(rs.DEFAULT_OUTPUT_DIR)
        file_path = os.path.realpath(os.path.join(out_root, safe_name))
        if not file_path.startswith(out_root + os.sep) or not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        ext = os.path.splitext(safe_name)[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else ("image/webp" if ext == ".webp" else "image/png")
        return FileResponse(file_path, media_type=mime, filename=safe_name)

    # --------------------------------------------------------------------------
    # 1.5 OpenAI Videos API: POST/GET /v1/videos, GET/DELETE /v1/videos/{video_id}, GET /v1/videos/{video_id}/content
    #     完整支持 MiniMax-H3 (T2VA / FL2VA) 与 MiniMax-H3-Ref2VA (多模态参考生成音视频)
    # --------------------------------------------------------------------------
    @app.post("/v1/videos")
    @app.post("/videos")
    async def openai_create_video(request: Request):
        # (shared state on runtime_state)
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()

        content_type = (request.headers.get("content-type") or "").lower()
        params: Dict[str, Any] = {}
        cleanup_paths: List[str] = []
        raw_first_items: List[Any] = []
        raw_last_items: List[Any] = []
        raw_ref_img_items: List[Any] = []
        raw_ref_vid_items: List[Any] = []
        raw_ref_aud_items: List[Any] = []

        if "multipart/form-data" in content_type:
            form = await request.form()
            for k in form.keys():
                vals = form.getlist(k)
                if k in ("input_reference", "first_frame"):
                    raw_first_items.extend(vals)
                elif k == "last_frame":
                    raw_last_items.extend(vals)
                elif k in ("ref_images", "ref_images[]", "input_images", "images"):
                    raw_ref_img_items.extend(vals)
                elif k in ("ref_videos", "ref_videos[]", "videos"):
                    raw_ref_vid_items.extend(vals)
                elif k in ("ref_audios", "ref_audios[]", "audios"):
                    raw_ref_aud_items.extend(vals)
                elif vals:
                    params[k] = vals[0]
        else:
            params = await request.json()
            for key_name, bucket in (
                ("input_reference", raw_first_items),
                ("first_frame", raw_first_items),
                ("first_frame_base64", raw_first_items),
                ("last_frame", raw_last_items),
                ("last_frame_base64", raw_last_items),
                ("ref_images", raw_ref_img_items),
                ("ref_images_base64", raw_ref_img_items),
                ("input_images", raw_ref_img_items),
                ("ref_videos", raw_ref_vid_items),
                ("ref_videos_base64", raw_ref_vid_items),
                ("ref_audios", raw_ref_aud_items),
                ("ref_audios_base64", raw_ref_aud_items),
            ):
                val = params.get(key_name)
                if isinstance(val, list):
                    bucket.extend(val)
                elif val:
                    bucket.append(val)

        prompt = str(params.get("prompt") or params.get("visual_prompt") or "").strip()
        if not prompt:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "prompt is required", "type": "invalid_request_error"}},
            )

        req_model_str = str(params.get("model") or "").strip()
        target_vid_id = _resolve_openai_model_id(req_model_str, expected_modality="video")
        if not target_vid_id or not _is_video_modality(MODEL_CATALOG.get(target_vid_id, {}).get("modality")):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"Unsupported video model '{req_model_str}'", "type": "invalid_request_error", "code": "model_not_found"}},
            )

        # 检查视频模型（或其兼容兄弟管线）是否已驻留显存
        sibling_id = "minimax_h3" if target_vid_id == "minimax_h3_ref2va" else "minimax_h3_ref2va"
        if target_vid_id not in LOADED_RUNTIMES and sibling_id not in LOADED_RUNTIMES and rs.LOADED_MODEL_ID not in (target_vid_id, sibling_id):
            req_name = MODEL_CATALOG.get(target_vid_id, {}).get("name", target_vid_id)
            return JSONResponse(
                status_code=503,
                content={"error": {"message": f"{req_name} 尚未加载到 GPU 显存，请先点击「加载到显存」。", "type": "server_error", "code": "model_not_loaded"}},
            )

        first_b64 = None
        for item in raw_first_items:
            first_b64 = await _coerce_media_item_to_b64_or_path(item, kind="image", cleanup_paths=cleanup_paths)
            if first_b64:
                break
        last_b64 = None
        for item in raw_last_items:
            last_b64 = await _coerce_media_item_to_b64_or_path(item, kind="image", cleanup_paths=cleanup_paths)
            if last_b64:
                break
        ref_img_b64s: List[str] = []
        for item in raw_ref_img_items[:MAX_REF2VA_IMAGES]:
            val = await _coerce_media_item_to_b64_or_path(item, kind="image", cleanup_paths=cleanup_paths)
            if val:
                ref_img_b64s.append(val)
        ref_vid_items: List[str] = []
        for item in raw_ref_vid_items[:MAX_REF2VA_VIDEOS]:
            val = await _coerce_media_item_to_b64_or_path(item, kind="video", cleanup_paths=cleanup_paths)
            if val:
                ref_vid_items.append(val)
        ref_aud_items: List[str] = []
        for item in raw_ref_aud_items[:MAX_REF2VA_AUDIOS]:
            val = await _coerce_media_item_to_b64_or_path(item, kind="audio", cleanup_paths=cleanup_paths)
            if val:
                ref_aud_items.append(val)

        task_mode = str(params.get("task_mode") or "").strip().lower()
        if not task_mode:
            if ref_img_b64s or ref_vid_items or ref_aud_items or target_vid_id == "minimax_h3_ref2va":
                if first_b64 and not (ref_img_b64s or ref_vid_items or ref_aud_items):
                    task_mode = "fl2va"
                elif ref_img_b64s or ref_vid_items or ref_aud_items or first_b64:
                    task_mode = "ref2va"
                else:
                    task_mode = "t2va"
            elif first_b64 or last_b64:
                task_mode = "fl2va"
            else:
                task_mode = "t2va"

        w, h = _parse_openai_size(str(params.get("size") or ""), 832, 480)
        fps = max(16, min(30, int(params.get("fps") or 24)))
        if params.get("num_frames") is not None and str(params.get("num_frames")).strip() != "":
            n_frames = _align_minimax_h3_frame_count(max(22, min(362, int(params["num_frames"]))))
        else:
            n_frames = _seconds_to_minimax_frames(params.get("seconds") or 5, fps=fps)

        steps = max(10, min(60, int(params.get("steps") or params.get("num_inference_steps") or 35)))
        seed_raw = params.get("seed")
        has_seed = seed_raw is not None and str(seed_raw).strip() != ""
        seed_int = int(seed_raw) if has_seed else 42
        rand_raw = params.get("randomize_seed")
        randomize_seed = (not has_seed) if rand_raw is None else (str(rand_raw).strip().lower() in ("1", "true", "yes") if isinstance(rand_raw, str) else bool(rand_raw))
        batch_count = max(1, min(8, int(params.get("batch_count") or 1)))

        rs.API_REQUEST_COUNT += 1
        gen_req = GenerateVideoRequest(
            model_id=target_vid_id,
            task_mode=task_mode,
            visual_prompt=prompt,
            negative_prompt=str(params.get("negative_prompt") or ""),
            soundscape_prompt=str(params.get("soundscape_prompt") or ""),
            music_prompt=str(params.get("music_prompt") or ""),
            first_frame_base64=first_b64,
            last_frame_base64=last_b64,
            ref_images_base64=ref_img_b64s or None,
            ref_videos_base64=ref_vid_items or None,
            ref_audios_base64=ref_aud_items or None,
            width=w,
            height=h,
            num_frames=n_frames,
            num_inference_steps=steps,
            fps=fps,
            seed=seed_int,
            randomize_seed=randomize_seed,
            batch_count=batch_count,
        )

        job_id = uuid.uuid4().hex[:12]
        vid_id = f"video_{job_id}"
        now_ts = time.time()
        _JOB_CANCEL_EVENTS[job_id] = threading.Event()
        JOBS[job_id] = {
            "job_id": job_id,
            "openai_video_id": vid_id,
            "model_id": target_vid_id,
            "status": "queued",
            "progress": 0.0,
            "message": "任务已提交至 OpenAI Videos 队列...",
            "seed": seed_int,
            "created_at": now_ts,
            "started_at": now_ts,
            "elapsed": 0.0,
            "size": f"{w}x{h}",
            "seconds": round(n_frames / float(fps), 1),
            "num_frames": n_frames,
            "fps": fps,
            "task_mode": task_mode,
            "filename": None,
            "video_url": None,
            "media_type": "video",
            "batch_count": batch_count,
            "completed_count": 0,
            "completed_videos": [],
            "_temp_cleanup_paths": cleanup_paths,
        }
        threading.Thread(
            target=rs.resolve_callable("_run_async_generation_job", _run_async_generation_job),
            args=(job_id, gen_req),
            daemon=True,
        ).start()
        tok = _extract_api_token_from_request(request, allow_query_token=False)
        return _format_openai_video_object(job_id, JOBS[job_id], base_url=_build_public_base_url(request), token_q=tok)

    @app.get("/v1/videos")
    @app.get("/videos")
    def openai_list_videos(request: Request, limit: int = 20):
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()
        tok = _extract_api_token_from_request(request, allow_query_token=False)
        base_url = _build_public_base_url(request)
        vid_jobs = [
            (jid, j)
            for jid, j in JOBS.items()
            if j.get("media_type") == "video" or str(j.get("openai_video_id") or "").startswith("video_")
        ]
        vid_jobs.sort(key=lambda pair: float(pair[1].get("created_at") or pair[1].get("started_at") or 0.0), reverse=True)
        items = [
            _format_openai_video_object(jid, j, base_url=base_url, token_q=tok)
            for jid, j in vid_jobs[:max(1, min(100, int(limit)))]
        ]
        return {"object": "list", "data": items, "has_more": len(vid_jobs) > len(items)}

    @app.get("/v1/videos/{video_id}")
    @app.get("/videos/{video_id}")
    def openai_get_video(video_id: str, request: Request):
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        found = _find_job_by_openai_video_id(video_id)
        if not found:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"Video '{video_id}' not found", "type": "invalid_request_error", "code": "not_found"}},
            )
        jid, j = found
        if j.get("status") in ("queued", "running"):
            touch_activity()
            if j.get("started_at"):
                j["elapsed"] = round(max(0.0, time.time() - float(j["started_at"])), 1)
        tok = _extract_api_token_from_request(request, allow_query_token=False)
        return _format_openai_video_object(jid, j, base_url=_build_public_base_url(request), token_q=tok)

    @app.get("/v1/videos/{video_id}/content")
    @app.get("/videos/{video_id}/content")
    def openai_download_video_content(video_id: str, request: Request):
        auth_err = _verify_external_api_auth(request, protocol="openai", allow_query_token=True)
        if auth_err:
            return auth_err
        touch_activity()
        found = _find_job_by_openai_video_id(video_id)
        if not found:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"Video '{video_id}' not found", "type": "invalid_request_error", "code": "not_found"}},
            )
        _, j = found
        fname = os.path.basename(str(j.get("filename") or ""))
        if not fname:
            return JSONResponse(
                status_code=409,
                content={"error": {"message": f"Video '{video_id}' is not ready yet (status={j.get('status')})", "type": "invalid_request_error"}},
            )
        out_root = os.path.realpath(rs.DEFAULT_OUTPUT_DIR)
        file_path = os.path.realpath(os.path.join(out_root, fname))
        if not file_path.startswith(out_root + os.sep) or not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="Video file not found")
        return FileResponse(file_path, media_type="video/mp4", filename=fname)

    @app.delete("/v1/videos/{video_id}")
    @app.delete("/videos/{video_id}")
    def openai_delete_video(video_id: str, request: Request):
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()
        found = _find_job_by_openai_video_id(video_id)
        if not found:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"Video '{video_id}' not found", "type": "invalid_request_error", "code": "not_found"}},
            )
        jid, j = found
        ev = _JOB_CANCEL_EVENTS.get(jid)
        if ev is not None:
            ev.set()
        vid_out_id = j.get("openai_video_id") or f"video_{jid}"
        JOBS.pop(jid, None)
        return {"id": vid_out_id, "object": "video.deleted", "deleted": True}

    # --------------------------------------------------------------------------
    # 2. Anthropic Messages 协议: POST /v1/messages & POST /v1/messages/count_tokens
    #    完美兼容 Anthropic Python/TS SDK、Claude Code (ANTHROPIC_BASE_URL) 等工具
    # --------------------------------------------------------------------------
    @app.post("/v1/messages/count_tokens")
    @app.post("/messages/count_tokens")
    @app.post("/v1/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request):
        auth_err = _verify_external_api_auth(request, protocol="anthropic")
        if auth_err:
            return auth_err
        touch_activity()
        body = await request.json()
        raw_txt = json.dumps(body.get("messages", []), ensure_ascii=False)
        bundle = _get_qwen_llm_bundle(preferred_model_id=str(body.get("model") or ""))
        toks = _count_bundle_tokens(bundle, raw_txt)
        return {"input_tokens": toks}

    @app.post("/v1/messages")
    @app.post("/messages")
    @app.post("/v1/v1/messages")
    async def anthropic_messages(request: Request):
        # (shared state on runtime_state)
        auth_err = _verify_external_api_auth(request, protocol="anthropic")
        if auth_err:
            return auth_err
        touch_activity()

        body = await request.json()
        req_model = body.get("model") or "Qwen/Qwen3.8-27B"
        bundle = _get_qwen_llm_bundle(preferred_model_id=str(req_model))
        if bundle is None:
            return JSONResponse(
                status_code=503,
                content={
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "Qwen3.8-27B / Qwen3.8-27B-GGUF 尚未加载至 Colab GPU 显存，请先在 Mac 客户端点击「加载到显存」。",
                    },
                },
            )

        rs.API_REQUEST_COUNT += 1
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens") or 16384)

        thinking_cfg = body.get("thinking")
        if isinstance(thinking_cfg, dict):
            enable_thinking = thinking_cfg.get("type") != "disabled"
        else:
            enable_thinking = True

        temperature = float(body["temperature"]) if body.get("temperature") is not None else (1.0 if enable_thinking else 0.7)
        top_p = float(body["top_p"]) if body.get("top_p") is not None else (0.95 if enable_thinking else 0.80)
        top_k = int(body["top_k"]) if body.get("top_k") is not None else 20

        print(f"[Anthropic API] POST /v1/messages | model={req_model} | backend={bundle.get('backend')} | stream={stream} | thinking={enable_thinking}")

        hf_messages, pil_images = _normalize_to_hf_messages_and_images(
            body.get("messages", []),
            system_prompt=body.get("system"),
            protocol="anthropic",
        )
        hf_tools = _normalize_hf_tools(body.get("tools"), protocol="anthropic")

        streamer, prompt_tokens, start_in_thinking, gen_thread, err_box, cancel_event = _prepare_qwen_streamer_generation(
            hf_messages=hf_messages,
            pil_images=pil_images,
            hf_tools=hf_tools,
            enable_thinking=enable_thinking,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            preferred_model_id=str(req_model),
            top_k=top_k,
            min_p=0.0,
            presence_penalty=0.0 if enable_thinking else 1.5,
            repeat_penalty=1.0,
        )

        msg_id = f"msg_{uuid.uuid4().hex[:22]}"

        if not stream:
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            think_parts: List[str] = []
            text_parts: List[str] = []
            out_chunks = 0
            with _ActiveApiStreamGuard():
                try:
                    for piece in streamer:
                        out_chunks += 1
                        if out_chunks % 64 == 0:
                            touch_activity()
                        for kind, val in parser.feed(piece):
                            if kind == "thinking":
                                think_parts.append(val)
                            elif kind == "text":
                                text_parts.append(val)
                finally:
                    cancel_event.set()
            tail_events, tool_calls = parser.finish()
            for kind, val in tail_events:
                if kind == "thinking":
                    think_parts.append(val)
                elif kind == "text":
                    text_parts.append(val)
            gen_thread.join(timeout=5.0)
            if err_box[0] is not None:
                return JSONResponse(
                    status_code=500,
                    content={"type": "error", "error": {"type": "api_error", "message": str(err_box[0])}},
                )

            full_thinking = "".join(think_parts).strip()
            full_text, extra_tools = _extract_qwen_tool_calls("".join(text_parts).strip())
            all_tools = tool_calls + extra_tools

            content_blocks: List[Dict[str, Any]] = []
            if full_thinking:
                content_blocks.append({"type": "thinking", "thinking": full_thinking, "signature": "qwen38-sig"})
            if full_text or not all_tools:
                content_blocks.append({"type": "text", "text": full_text})
            for tc in all_tools:
                content_blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})

            final_prompt_tokens = int(getattr(streamer, "prompt_tokens", prompt_tokens) or prompt_tokens)
            out_tokens = int(getattr(streamer, "generated_tokens", 0)) or (
                _count_bundle_tokens(bundle, full_thinking + full_text) if (full_thinking or full_text) else 1
            )
            stop_reason = "tool_use" if all_tools else "end_turn"

            return {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": req_model,
                "content": content_blocks,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": final_prompt_tokens, "output_tokens": out_tokens},
            }

        def _anthropic_sse_event(ev_name: str, payload: Dict[str, Any]) -> str:
            return f"event: {ev_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        def _anthropic_sse_stream():
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            yield _anthropic_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": req_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": prompt_tokens, "output_tokens": 1},
                    },
                },
            )

            next_block_idx = 0
            active_block_type: Optional[str] = None  # "thinking" | "text" | None
            active_block_idx = 0
            comp_chars = 0

            def _ensure_block(target_type: str):
                nonlocal next_block_idx, active_block_type, active_block_idx
                out_strs: List[str] = []
                if active_block_type == target_type:
                    return out_strs
                if active_block_type is not None:
                    if active_block_type == "thinking":
                        out_strs.append(
                            _anthropic_sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": active_block_idx,
                                    "delta": {"type": "signature_delta", "signature": "qwen38-sig"},
                                },
                            )
                        )
                    out_strs.append(
                        _anthropic_sse_event(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": active_block_idx},
                        )
                    )
                active_block_idx = next_block_idx
                next_block_idx += 1
                active_block_type = target_type
                block_init = (
                    {"type": "thinking", "thinking": ""}
                    if target_type == "thinking"
                    else {"type": "text", "text": ""}
                )
                out_strs.append(
                    _anthropic_sse_event(
                        "content_block_start",
                        {"type": "content_block_start", "index": active_block_idx, "content_block": block_init},
                    )
                )
                return out_strs

            chunk_idx = 0
            with _ActiveApiStreamGuard():
                try:
                    for piece in streamer:
                        comp_chars += len(piece)
                        chunk_idx += 1
                        if chunk_idx % 64 == 0:
                            touch_activity()
                        for kind, val in parser.feed(piece):
                            if not val:
                                continue
                            for s in _ensure_block(kind):
                                yield s
                            delta_obj = (
                                {"type": "thinking_delta", "thinking": val}
                                if kind == "thinking"
                                else {"type": "text_delta", "text": val}
                            )
                            yield _anthropic_sse_event(
                                "content_block_delta",
                                {"type": "content_block_delta", "index": active_block_idx, "delta": delta_obj},
                            )

                    tail_events, tool_calls = parser.finish()
                    for kind, val in tail_events:
                        if not val:
                            continue
                        for s in _ensure_block(kind):
                            yield s
                        delta_obj = (
                            {"type": "thinking_delta", "thinking": val}
                            if kind == "thinking"
                            else {"type": "text_delta", "text": val}
                        )
                        yield _anthropic_sse_event(
                            "content_block_delta",
                            {"type": "content_block_delta", "index": active_block_idx, "delta": delta_obj},
                        )

                    gen_thread.join(timeout=5.0)
                    if getattr(streamer, "generated_tokens", 0) > 0:
                        print(f"[Anthropic API] 流式完成 | {streamer.generated_tokens} tokens | {streamer.tok_per_sec} tok/s")
                finally:
                    cancel_event.set()

            if active_block_type is not None:
                if active_block_type == "thinking":
                    yield _anthropic_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": active_block_idx,
                            "delta": {"type": "signature_delta", "signature": "qwen38-sig"},
                        },
                    )
                yield _anthropic_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": active_block_idx},
                )
                active_block_type = None
            elif not tool_calls:
                # 若全程无输出块，保底发送一个空文本块符合 Anthropic 协议规范
                yield _anthropic_sse_event(
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                )
                yield _anthropic_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
                next_block_idx = 1

            for tc in tool_calls:
                t_idx = next_block_idx
                next_block_idx += 1
                yield _anthropic_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": t_idx,
                        "content_block": {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": {}},
                    },
                )
                yield _anthropic_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": t_idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(tc["arguments"], ensure_ascii=False),
                        },
                    },
                )
                yield _anthropic_sse_event("content_block_stop", {"type": "content_block_stop", "index": t_idx})

            est_out_toks = int(getattr(streamer, "generated_tokens", 0)) or max(1, comp_chars // 3)
            yield _anthropic_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use" if tool_calls else "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": est_out_toks},
                },
            )
            yield _anthropic_sse_event("message_stop", {"type": "message_stop"})

        return StreamingResponse(
            _anthropic_sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

