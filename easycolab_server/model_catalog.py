"""EasyColab 服务端模块: model_catalog."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple
import glob
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request

from . import runtime_state as rs
from .runtime_state import ACTIVE_TQDM_BARS, CONSOLE_LOCK, DRIVE_MOUNTED, LEGACY_DRIVE_MODEL_ROOTS, LOADED_RUNTIMES, MODEL_BASE_ROOT, MODEL_DOWNLOAD_STATE, _StripAuthRedirectHandler, _TELEMETRY_LOCK, _safe_shm_available_bytes, touch_activity

MINIMAX_H3_BF16_FILES = (
    [
        (f"FL2VA/transformer/model-{i:05d}-of-00013.safetensors", f"DiT 音视频主干 BF16 {i}/13")
        for i in range(1, 14)
    ]
    + [
        (f"FL2VA/text_encoder/model-{i:05d}-of-00014.safetensors", f"多模态文本编码器 BF16 {i}/14")
        for i in range(1, 15)
    ]
    + [
        ("FL2VA/video_vae/source/model.safetensors", "3D 视频 VAE 解码器 BF16 (约 9.7 GB)"),
        ("FL2VA/audio_vae/model.safetensors", "32kHz 立体声 VAE 解码器 BF16 (约 0.56 GB)"),
    ]
)

MINIMAX_H3_REF2VA_BF16_FILES = (
    [
        (f"Ref2VA/transformer/model-{i:05d}-of-00013.safetensors", f"Ref2VA 主体参考 DiT 主干 BF16 {i}/13")
        for i in range(1, 14)
    ]
    + [
        (
            f"Ref2VA/text_encoder/model-{i:05d}-of-00014.safetensors",
            f"多模态文本编码器 BF16 {i}/14 (与 MiniMax-H3 自动共享)",
        )
        for i in range(1, 15)
    ]
    + [
        ("Ref2VA/video_vae/source/model.safetensors", "3D 视频 VAE 解码器 BF16 (约 9.7 GB · 与 MiniMax-H3 自动共享)"),
        ("Ref2VA/audio_vae/model.safetensors", "32kHz 立体声 VAE 解码器 BF16 (约 0.56 GB · 与 MiniMax-H3 自动共享)"),
    ]
)

# ==============================================================================
# 开源旗舰多模型注册中心 (Hugging Face Top Models 2026 · 全量免审核 Gated=False)
# ==============================================================================
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "minimax_h3": {
        "id": "minimax_h3",
        "name": "MiniMax-H3 (33B)",
        "repo_id": "MiniMaxAI/MiniMax-H3",
        "modality": "video_audio",
        "category": "全模态音视频生成 · 文生/首尾帧/参考图 · BF16 原生满血 · 32kHz 立体声",
        "approx_size_gb": 134.1,
        "files": MINIMAX_H3_BF16_FILES,
    },
    "minimax_h3_ref2va": {
        "id": "minimax_h3_ref2va",
        "name": "MiniMax-H3-Ref2VA (33B · 多参考图/视频)",
        "repo_id": "MiniMaxAI/MiniMax-H3",
        "modality": "video_audio",
        "pipeline_arch": "minimax_h3_ref2va",
        "category": "多主体参考音视频 (Ref2VA · 最多9张参考图+3段参考视频) · 零拷贝共享 72.4GB 编码器与双 VAE",
        "approx_size_gb": 134.1,
        "files": MINIMAX_H3_REF2VA_BF16_FILES,
    },
    "minimax_h3_image": {
        "id": "minimax_h3_image",
        "name": "MiniMax-H3-Image (33B · 4K)",
        "repo_id": "MiniMaxAI/MiniMax-H3-Image",
        "modality": "image",
        "pipeline_arch": "minimax_h3_image",
        "category": "6 步原生 4K (16MP) 文生图与多参考图指令修图 · 共享编码器零拷贝复用",
        "approx_size_gb": 135.3,
        "files": [
            (
                f"transformer_diffusers/diffusion_pytorch_model-{i:05d}-of-00012.safetensors",
                f"MiniMax-H3-Image 33B 图像 DiT 主干 {i}/12",
            )
            for i in range(1, 13)
        ]
        + [
            (
                f"text_encoder/model-{i:05d}-of-00014.safetensors",
                f"32B 视觉多模态编码器 {i}/14 (可与 MiniMax-H3 自动共享)",
            )
            for i in range(1, 15)
        ]
        + [
            ("vae_diffusers/diffusion_pytorch_model.safetensors", "4K 16MP 高保真图像 VAE 解码器"),
        ],
    },
    "qwen38_27b": {
        "id": "qwen38_27b",
        "name": "Qwen3.8-27B",
        "repo_id": "Qwen/Qwen3.8-27B",
        "modality": "llm_vlm",
        "category": "视觉理解与深度推理大模型 · BF16 原生满血",
        "approx_size_gb": 51.8,
        "files": [
            (f"model-{i:05d}-of-00018.safetensors", f"Qwen3.8-27B BF16 权重分片 {i}/18")
            for i in range(1, 19)
        ],
    },
    "qwen38_27b_gguf": {
        "id": "qwen38_27b_gguf",
        "name": "Qwen3.8-27B-GGUF",
        "repo_id": "unsloth/Qwen3.8-27B-GGUF",
        "modality": "llm_vlm",
        "category": "Unsloth UD-Q4_K_M + MTP 投机加速 · 支持 256K 上下文",
        "approx_size_gb": 18.7,
        "files": [
            ("Qwen3.8-27B-UD-Q4_K_M.gguf", "Unsloth UD-Q4_K_M 主模型权重 (约 16.5 GB)"),
            ("mmproj-F16.gguf", "多模态视觉投影器 mmproj-F16 (约 0.93 GB)"),
            ("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "Unsloth MTP 多词元并行投机解码头 (约 1.31 GB)"),
        ],
    },
    "qwen_image_21": {
        "id": "qwen_image_21",
        "name": "Qwen-Image-2.1",
        "repo_id": "Qwen/Qwen-Image-2.1",
        "modality": "image",
        "category": "旗舰文生图与多图指令修图",
        "approx_size_gb": 30.9,
        "files": [
            ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "DiT 图像主干 1/2"),
            ("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "DiT 图像主干 2/2"),
            ("text_encoder/model-00001-of-00004.safetensors", "多模态编码器 1/4"),
            ("text_encoder/model-00002-of-00004.safetensors", "多模态编码器 2/4"),
            ("text_encoder/model-00003-of-00004.safetensors", "多模态编码器 3/4"),
            ("text_encoder/model-00004-of-00004.safetensors", "多模态编码器 4/4"),
            ("vae/diffusion_pytorch_model.safetensors", "高保真图像 VAE"),
        ],
    },
    "z_image_turbo": {
        "id": "z_image_turbo",
        "name": "Z-Image-Turbo",
        "repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "modality": "image",
        "pipeline_arch": "z_image_turbo",
        "category": "8 步极速超清写实与中英文字生图",
        "approx_size_gb": 30.6,
        "files": [
            ("transformer/diffusion_pytorch_model-00001-of-00003.safetensors", "Z-Image DiT 主干 1/3"),
            ("transformer/diffusion_pytorch_model-00002-of-00003.safetensors", "Z-Image DiT 主干 2/3"),
            ("transformer/diffusion_pytorch_model-00003-of-00003.safetensors", "Z-Image DiT 主干 3/3"),
            ("text_encoder/model-00001-of-00003.safetensors", "文本编码器 1/3"),
            ("text_encoder/model-00002-of-00003.safetensors", "文本编码器 2/3"),
            ("text_encoder/model-00003-of-00003.safetensors", "文本编码器 3/3"),
            ("vae/diffusion_pytorch_model.safetensors", "图像 VAE 解码器"),
        ],
    },
}

MINIMAX_H3_IMAGE_FILES = list(MODEL_CATALOG["minimax_h3_image"]["files"])

BUILTIN_MODEL_IDS = set(MODEL_CATALOG.keys())

CUSTOM_MODELS_LOCK = threading.Lock()

def _normalize_builtin_model_alias(model_id: Optional[str], repo_id: str = "") -> Optional[str]:
    """兼容曾通过自定义仓库手动导入的内置旗舰模型 ID（如 custom_minimaxai_minimax_h3_image -> minimax_h3_image）"""
    mid = (model_id or "").strip()
    rlow = (repo_id or "").strip().lower()
    if mid in ("custom_minimaxai_minimax_h3_image", "minimax-h3-image") or rlow == "minimaxai/minimax-h3-image":
        return "minimax_h3_image"
    if mid in ("custom_minimaxai_minimax_h3_ref2va", "minimax-h3-ref2va"):
        return "minimax_h3_ref2va"
    return mid or None

def _get_custom_models_json_paths() -> List[str]:
    paths = [os.path.join(MODEL_BASE_ROOT, "custom_models.json")]
    local_p = os.path.join(os.getcwd(), "models", "custom_models.json")
    if os.path.abspath(local_p) != os.path.abspath(paths[0]):
        paths.append(local_p)
    return paths

def _make_custom_model_id(repo_id: str, include_pattern: str = "") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", repo_id.strip().lower()).strip("_")
    if include_pattern.strip():
        pat_slug = re.sub(r"[^a-zA-Z0-9]+", "_", include_pattern.strip().lower()).strip("_")
        if pat_slug and pat_slug not in ("gguf", "safetensors"):
            slug = f"{slug}_{pat_slug[:24]}"
    return f"custom_{slug[:64]}"

def _normalize_hf_repo_and_pattern(raw_repo: str, raw_pattern: str = "") -> Tuple[str, str]:
    s = (raw_repo or "").strip()
    pat = (raw_pattern or "").strip()
    s = re.sub(r"^https?://(?:www\.)?(?:huggingface\.co|hf\.co|hf-mirror\.com)/", "", s, flags=re.IGNORECASE)
    s = s.strip("/")
    for marker in ("/blob/", "/resolve/", "/raw/"):
        if marker in s:
            left, right = s.split(marker, 1)
            parts = right.split("/", 1)
            if len(parts) == 2 and parts[1] and not pat:
                pat = parts[1].strip()
            s = left.strip("/")
            break
    if "/tree/" in s:
        s = s.split("/tree/", 1)[0].strip("/")
    tokens = [p for p in s.split("/") if p]
    if len(tokens) >= 2:
        repo_id = f"{tokens[0]}/{tokens[1]}"
        if len(tokens) > 2 and not pat:
            pat = "/".join(tokens[2:])
        return repo_id, pat
    return s, pat

def _load_persisted_custom_models():
    for path in _get_custom_models_json_paths():
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not item.get("id") or not item.get("repo_id"):
                    continue
                mid = str(item["id"])
                item_repo_low = str(item.get("repo_id") or "").strip().lower()
                if mid in BUILTIN_MODEL_IDS or item_repo_low == "minimaxai/minimax-h3-image":
                    continue
                raw_files = item.get("files") or []
                norm_files: List[Tuple[str, str]] = []
                for rf in raw_files:
                    if isinstance(rf, (list, tuple)) and len(rf) >= 2:
                        norm_files.append((str(rf[0]), str(rf[1])))
                    elif isinstance(rf, dict) and rf.get("name"):
                        norm_files.append((str(rf["name"]), str(rf.get("desc") or rf["name"])))
                if not norm_files:
                    continue
                MODEL_CATALOG[mid] = {
                    "id": mid,
                    "name": str(item.get("name") or item["repo_id"].split("/")[-1]),
                    "repo_id": str(item["repo_id"]),
                    "include_pattern": str(item.get("include_pattern") or ""),
                    "modality": str(item.get("modality") or "llm_vlm"),
                    "pipeline_arch": str(item.get("pipeline_arch") or "gguf"),
                    "category": str(item.get("category") or "自定义 HuggingFace 模型"),
                    "approx_size_gb": float(item.get("approx_size_gb") or 0.0),
                    "files": norm_files,
                    "is_custom": True,
                }
            break
        except Exception as e:
            print(f"[自定义模型] 读取 {path} 跳过: {e}")

def _save_persisted_custom_models():
    custom_list = []
    for mid, spec in MODEL_CATALOG.items():
        if mid in BUILTIN_MODEL_IDS or not spec.get("is_custom"):
            continue
        custom_list.append({
            "id": spec["id"],
            "name": spec["name"],
            "repo_id": spec["repo_id"],
            "include_pattern": spec.get("include_pattern", ""),
            "modality": spec.get("modality", "llm_vlm"),
            "pipeline_arch": spec.get("pipeline_arch", "gguf"),
            "category": spec.get("category", "自定义 HuggingFace 模型"),
            "approx_size_gb": round(float(spec.get("approx_size_gb") or 0.0), 2),
            "files": [[fn, desc] for fn, desc in (spec.get("files") or [])],
            "is_custom": True,
        })
    for path in _get_custom_models_json_paths():
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_p = path + ".tmp"
            with open(tmp_p, "w", encoding="utf-8") as wf:
                json.dump(custom_list, wf, ensure_ascii=False, indent=2)
            os.replace(tmp_p, path)
        except Exception as e:
            print(f"[自定义模型] 写入 {path} 跳过: {e}")

def _get_custom_models_public_list() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for mid, spec in MODEL_CATALOG.items():
        if mid in BUILTIN_MODEL_IDS or not spec.get("is_custom"):
            continue
        out.append({
            "id": spec["id"],
            "name": spec["name"],
            "repo_id": spec["repo_id"],
            "include_pattern": spec.get("include_pattern", ""),
            "modality": spec.get("modality", "llm_vlm"),
            "pipeline_arch": spec.get("pipeline_arch", "gguf"),
            "category": spec.get("category", "自定义 HuggingFace 模型"),
            "approx_size_gb": round(float(spec.get("approx_size_gb") or 0.0), 2),
            "files": [
                {"name": fn, "desc": desc, "exists": False, "size_gb": 0.0}
                for fn, desc in (spec.get("files") or [])
            ],
            "is_custom": True,
        })
    return out

_HF_TRENDING_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

_HF_TRENDING_LOCK = threading.Lock()

def _clean_hf_readme_excerpt(raw_md: str) -> str:
    body = raw_md or ""
    if body.startswith("---"):
        end_idx = body.find("\n---", 3)
        if end_idx != -1:
            body = body[end_idx + 4 :]
    body = re.sub(r"<!--[\s\S]*?-->", "", body)
    body = re.sub(r"<a[^>]*>\s*<img[^>]*>\s*</a>", "", body, flags=re.IGNORECASE)
    body = re.sub(r"<img[^>]*>", "", body, flags=re.IGNORECASE)
    body = re.sub(r"<h[1-4][^>]*>(.*?)</h[1-4]>", r"\n### \1\n", body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<[^>]+>", "", body)
    body = re.sub(r"\[!\[[^\]]*\]\([^\)]*\)\]\([^\)]*\)", "", body)
    body = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", body)
    lines_clean = []
    total_chars = 0
    for ln in body.splitlines():
        s_ln = ln.strip()
        if not s_ln or s_ln.startswith(("```", "|---", "| ---")):
            continue
        lines_clean.append(s_ln)
        total_chars += len(s_ln) + 1
        if total_chars >= 1400:
            break
    out = "\n".join(lines_clean).strip()
    # Clean any trailing incomplete markdown link if truncated
    out = re.sub(r"\[([^\]]+)\]\([^\)\n]*$", r"\1", out)
    return out

_UNSUPPORTED_HF_PIPELINE_TAGS = {
    "image-to-3d",
    "text-to-3d",
    "image-segmentation",
    "object-detection",
    "depth-estimation",
    "mask-generation",
    "keypoint-detection",
    "robotics",
    "reinforcement-learning",
    "tabular-classification",
    "tabular-regression",
    "time-series-forecasting",
    "audio-classification",
    "automatic-speech-recognition",
    "text-to-speech",
    "audio-to-audio",
    "feature-extraction",
    "sentence-similarity",
    "token-classification",
    "text-classification",
    "zero-shot-image-classification",
    "zero-shot-classification",
    "fill-mask",
    "question-answering",
    "table-question-answering",
    "video-classification",
}

_SUPPORTED_TRENDING_PIPELINE_TAGS = {
    "text-generation",
    "image-text-to-text",
    "visual-question-answering",
    "any-to-any",
    "text-to-image",
    "image-to-image",
    "text-to-video",
    "image-to-video",
}

def _analyze_hf_model_sizing(
    item: Dict[str, Any], is_gguf: bool, fallback_storage_gb: float = 0.0
) -> Dict[str, Any]:
    """
    提取 HuggingFace 模型参数量 (safetensors.total / gguf.total / 名称正则回退)，
    并根据精度分布与量化状态估算 Colab 单卡显存占用及是否适合部署。
    """
    rid = str(item.get("id") or item.get("modelId") or "")
    ptag = str(item.get("pipeline_tag") or "").strip().lower()
    st = item.get("safetensors") if isinstance(item.get("safetensors"), dict) else {}
    gg = item.get("gguf") if isinstance(item.get("gguf"), dict) else {}
    total_params = int(st.get("total") or gg.get("total") or 0)
    if total_params <= 0:
        m = re.search(r"(?:^|[-_/])(\d+(?:\.\d+)?)\s*([bBmMtT])(?:[-_/\b]|$)", rid)
        if m:
            val = float(m.group(1))
            unit = m.group(2).upper()
            if unit == "T":
                total_params = int(val * 1e12)
            elif unit == "B":
                total_params = int(val * 1e9)
            elif unit == "M":
                total_params = int(val * 1e6)

    if total_params >= 1e12:
        param_label = f"{total_params / 1e12:.1f}T"
    elif total_params >= 1e9:
        b = total_params / 1e9
        param_label = f"{b:.0f}B" if b >= 20 else f"{b:.1f}B".replace(".0B", "B")
    elif total_params >= 1e6:
        param_label = f"{total_params / 1e6:.0f}M"
    else:
        param_label = ""

    params_b = total_params / 1e9 if total_params > 0 else 0.0
    if is_gguf:
        est_vram_gb = round(params_b * 0.62 + 1.2, 1) if params_b > 0 else 0.0
    else:
        pmap = st.get("parameters") if isinstance(st.get("parameters"), dict) else {}
        if pmap:
            bytes_total = 0.0
            for dtype, cnt in pmap.items():
                dt = str(dtype).upper()
                c = float(cnt or 0)
                if "32" in dt:
                    bytes_total += c * 4.0
                elif "16" in dt:
                    bytes_total += c * 2.0
                elif "8" in dt:
                    bytes_total += c * 1.0
                elif "4" in dt:
                    bytes_total += c * 0.55
                else:
                    bytes_total += c * 2.0
            est_vram_gb = round((bytes_total / 1e9) * 1.12 + 1.2, 1)
        elif params_b > 0:
            est_vram_gb = round(params_b * 2.1 + 1.2, 1)
        elif fallback_storage_gb > 0:
            est_vram_gb = round(fallback_storage_gb * 1.12 + 1.0, 1)
        else:
            est_vram_gb = 0.0

    vram_disp = max(1, int(round(est_vram_gb)))
    if not is_gguf and ptag in _UNSUPPORTED_HF_PIPELINE_TAGS:
        status = "too_large"
        zh = f"非内置管线 ({ptag} · 需专用SDK)"
        en = f"Custom Pipeline ({ptag})"
    elif est_vram_gb <= 0:
        status = "unknown"
        zh = "显存待探测"
        en = "VRAM TBD"
    elif est_vram_gb <= 14.2:
        status = "great"
        zh = f"适合 · T4(15G)可跑 (~{vram_disp}G)"
        en = f"Fit: T4 15G (~{vram_disp}G)"
    elif est_vram_gb <= 22.5:
        status = "good"
        zh = f"适合 · 需L4(24G) (~{vram_disp}G)"
        en = f"Fit: L4 24G (~{vram_disp}G)"
    elif est_vram_gb <= 39.0:
        status = "a100"
        zh = f"可部署 · 需A100(40G) (~{vram_disp}G)"
        en = f"Fit: A100 40G (~{vram_disp}G)"
    elif est_vram_gb <= 78.0:
        status = "heavy"
        zh = f"需A100(80G) (~{vram_disp}G)"
        en = f"Needs A100 80G (~{vram_disp}G)"
    elif is_gguf and params_b <= 240:
        status = "heavy"
        zh = f"需A100(80G)+CPU卸载 (~{vram_disp}G)"
        en = f"A100 80G + Offload (~{vram_disp}G)"
    else:
        status = "too_large"
        zh = f"超单卡上限 (~{vram_disp}G · 建议搜GGUF)"
        en = f"Too Large (~{vram_disp}G · Try GGUF)"

    return {
        "param_count": total_params,
        "param_label": param_label,
        "est_vram_gb": est_vram_gb,
        "deploy_status": status,
        "deploy_label_zh": zh,
        "deploy_label_en": en,
    }

def fetch_hf_trending_models_core(
    category: str = "all",
    limit: int = 12,
    hf_token: str = "",
    force_refresh: bool = False,
    search_query: str = "",
) -> List[Dict[str, Any]]:
    """
    实时从 HuggingFace Hub API (带 hf-mirror.com 自动容灾回退与内存缓存) 拉取最新热门趋势或关键词搜索模型，
    同时返回模型参数量 (param_label) 与 Colab 部署适用性评估 (deploy_status / deploy_label_zh / est_vram_gb)。
    """
    cat_key = (category or "all").strip().lower()
    sq = (search_query or "").strip()
    lim = max(4, min(24, int(limit or 12)))
    cache_key = f"{cat_key}:{lim}:{sq.lower()}"
    now = time.time()
    if not force_refresh:
        with _HF_TRENDING_LOCK:
            cached = _HF_TRENDING_CACHE.get(cache_key)
            if cached and (now - cached[0]) < 180.0 and cached[1]:
                return list(cached[1])

    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    headers = {"User-Agent": "EasyColab/2.0", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    expand_qs = (
        "&expand[]=safetensors&expand[]=gguf&expand[]=likes"
        "&expand[]=downloads&expand[]=pipeline_tag&expand[]=tags&expand[]=gated"
    )
    search_qs = f"&search={urllib.parse.quote(sq)}" if sq else ""

    query_paths: List[str] = []
    if cat_key == "gguf":
        query_paths.append(f"/api/models?sort=trendingScore&direction=-1&filter=gguf&limit={lim * 2}{search_qs}{expand_qs}")
    elif cat_key in ("llm", "llm_vlm"):
        query_paths.append(f"/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-generation&limit={lim * 2}{search_qs}{expand_qs}")
        if not sq:
            query_paths.append(f"/api/models?sort=trendingScore&direction=-1&pipeline_tag=image-text-to-text&limit={lim}{expand_qs}")
    elif cat_key in ("image", "video", "image_video"):
        query_paths.append(f"/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-to-image&limit={lim}{search_qs}{expand_qs}")
        query_paths.append(f"/api/models?sort=trendingScore&direction=-1&pipeline_tag=image-to-video&limit={lim}{search_qs}{expand_qs}")
        if not sq:
            query_paths.append(f"/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-to-video&limit={lim}{expand_qs}")
    else:
        query_paths.append(f"/api/models?sort=trendingScore&direction=-1&limit={lim * 2}{search_qs}{expand_qs}")

    raw_items: List[Dict[str, Any]] = []
    seen_ids = set()

    for qpath in query_paths:
        fetched = None
        for host in ("https://huggingface.co", "https://hf-mirror.com"):
            try:
                req = urllib.request.Request(f"{host}{qpath}", headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(data, list):
                        fetched = data
                        break
            except Exception:
                continue
        if fetched:
            for item in fetched:
                if not isinstance(item, dict):
                    continue
                rid = str(item.get("id") or item.get("modelId") or "").strip()
                if not rid or "/" not in rid or rid in seen_ids:
                    continue
                seen_ids.add(rid)
                raw_items.append(item)

    results: List[Dict[str, Any]] = []
    for item in raw_items:
        rid = str(item.get("id") or item.get("modelId") or "").strip()
        rid_low = rid.lower()
        ptag = str(item.get("pipeline_tag") or "").strip().lower()
        tags = [str(t).lower() for t in (item.get("tags") or []) if isinstance(t, str)]
        is_gguf = ("gguf" in rid_low) or ("gguf" in tags)
        gated_raw = item.get("gated")
        is_gated = bool(gated_raw) and str(gated_raw).lower() not in ("false", "0", "none")

        if "video" in ptag or "text-to-video" in tags or "image-to-video" in tags:
            modality = "video_audio"
        elif "text-to-image" in ptag or "image-to-image" in ptag or ("diffusers" in tags and not is_gguf):
            modality = "image"
        else:
            modality = "llm_vlm"

        if not sq and cat_key == "all" and not is_gguf:
            if ptag in _UNSUPPORTED_HF_PIPELINE_TAGS:
                continue
            if ptag and ptag not in _SUPPORTED_TRENDING_PIPELINE_TAGS:
                continue
            if not ptag and not any(t in tags for t in ("transformers", "diffusers", "text-generation", "conversational")):
                continue

        sizing = _analyze_hf_model_sizing(item, is_gguf)
        suggested_pat = "*Q4_K_M*.gguf" if is_gguf else ""
        results.append({
            "repo_id": rid,
            "name": rid.split("/")[-1],
            "pipeline_tag": ptag or ("gguf" if is_gguf else "text-generation"),
            "modality": modality,
            "is_gguf": is_gguf,
            "is_gated": is_gated,
            "suggested_pattern": suggested_pat,
            "likes": int(item.get("likes") or 0),
            "downloads": int(item.get("downloads") or 0),
            "param_count": sizing["param_count"],
            "param_label": sizing["param_label"],
            "est_vram_gb": sizing["est_vram_gb"],
            "deploy_status": sizing["deploy_status"],
            "deploy_label_zh": sizing["deploy_label_zh"],
            "deploy_label_en": sizing["deploy_label_en"],
        })
        if len(results) >= lim:
            break

    if results:
        with _HF_TRENDING_LOCK:
            _HF_TRENDING_CACHE[cache_key] = (now, list(results))
    return results

def fetch_hf_model_detail_core(repo_id: str, hf_token: str = "") -> Dict[str, Any]:
    """
    获取单个 HuggingFace 模型仓库的完整详情：
    参数量、架构、上下文长度、开源协议、基座模型、各权重/量化分片精确体积与单文件 Colab 卡型适配评估，以及 README 摘要。
    """
    clean_id = _normalize_hf_repo_id(repo_id)
    if not clean_id or "/" not in clean_id:
        return {"ok": False, "error": "Invalid HuggingFace Repo ID"}

    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    headers = {"User-Agent": "EasyColab/2.0", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    meta: Optional[Dict[str, Any]] = None
    for host in ("https://huggingface.co", "https://hf-mirror.com"):
        try:
            url = f"{host}/api/models/{clean_id}?blobs=true"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, dict) and (data.get("id") or data.get("modelId")):
                    meta = data
                    break
        except Exception:
            continue

    if not meta:
        return {"ok": False, "repo_id": clean_id, "error": f"无法获取仓库 {clean_id} 详情，请检查名称或填写 HF Token"}

    rid = str(meta.get("id") or meta.get("modelId") or clean_id).strip()
    rid_low = rid.lower()
    ptag = str(meta.get("pipeline_tag") or "").strip().lower()
    tags = [str(t) for t in (meta.get("tags") or []) if isinstance(t, str)]
    tags_low = [t.lower() for t in tags]
    is_gguf = ("gguf" in rid_low) or ("gguf" in tags_low) or bool(meta.get("gguf"))
    gated_raw = meta.get("gated")
    is_gated = bool(gated_raw) and str(gated_raw).lower() not in ("false", "0", "none")

    if "video" in ptag or "text-to-video" in tags_low or "image-to-video" in tags_low:
        modality = "video_audio"
    elif "text-to-image" in ptag or "image-to-image" in ptag or ("diffusers" in tags_low and not is_gguf):
        modality = "image"
    else:
        modality = "llm_vlm"

    card_data = meta.get("cardData") if isinstance(meta.get("cardData"), dict) else {}
    config_data = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    gguf_data = meta.get("gguf") if isinstance(meta.get("gguf"), dict) else {}

    lic = card_data.get("license") or ""
    if isinstance(lic, list):
        lic = ", ".join(str(x) for x in lic)
    lic = str(lic).strip() or "未标注 (See Model Card)"

    base_model = card_data.get("base_model") or ""
    if isinstance(base_model, list):
        base_model = ", ".join(str(x) for x in base_model[:2])
    base_model = str(base_model).strip()

    archs = config_data.get("architectures")
    arch_str = ""
    if isinstance(archs, list) and archs:
        arch_str = str(archs[0])
    elif gguf_data.get("architecture"):
        arch_str = str(gguf_data.get("architecture"))
    elif config_data.get("model_type"):
        arch_str = str(config_data.get("model_type"))

    ctx_len = int(gguf_data.get("context_length") or 0)
    langs = card_data.get("language") or []
    if isinstance(langs, str):
        langs = [langs]
    langs = [str(x) for x in langs if isinstance(x, str)][:8]

    used_storage_bytes = float(meta.get("usedStorage") or 0.0)
    used_storage_gb = round(used_storage_bytes / (1024.0 ** 3), 2) if used_storage_bytes > 0 else 0.0

    weight_files: List[Dict[str, Any]] = []
    total_weight_files_gb = 0.0
    for sib in meta.get("siblings") or []:
        if not isinstance(sib, dict):
            continue
        fname = str(sib.get("rfilename") or "").strip()
        f_low = fname.lower()
        if not fname or not f_low.endswith((".gguf", ".safetensors", ".bin", ".pth", ".ckpt")):
            continue
        lfs = sib.get("lfs") if isinstance(sib.get("lfs"), dict) else {}
        f_bytes = float(sib.get("size") or lfs.get("size") or 0.0)
        size_gb = round(f_bytes / (1024.0 ** 3), 2) if f_bytes > 0 else 0.0
        total_weight_files_gb += size_gb
        vram_need = round(size_gb * 1.12 + 1.0, 1) if size_gb > 0 else 0.0
        if vram_need <= 0:
            f_status, f_zh, f_en = "unknown", "体积未知", "Size TBD"
        elif vram_need <= 14.5:
            f_status, f_zh, f_en = "great", "T4(15G)可跑", "T4 15G Ready"
        elif vram_need <= 22.5:
            f_status, f_zh, f_en = "good", "L4(24G)可跑", "L4 24G Ready"
        elif vram_need <= 39.0:
            f_status, f_zh, f_en = "a100", "需A100(40G)", "A100 40G"
        elif vram_need <= 78.0:
            f_status, f_zh, f_en = "heavy", "需A100(80G)", "A100 80G"
        else:
            f_status, f_zh, f_en = "too_large", "超单卡显存", ">80G VRAM"
        weight_files.append({
            "filename": fname,
            "size_gb": size_gb,
            "fit_status": f_status,
            "fit_zh": f_zh,
            "fit_en": f_en,
        })

    fallback_gb = total_weight_files_gb if (not is_gguf and total_weight_files_gb > 0) else used_storage_gb
    sizing = _analyze_hf_model_sizing(meta, is_gguf, fallback_storage_gb=fallback_gb)

    readme_excerpt = ""
    for host in ("https://huggingface.co", "https://hf-mirror.com"):
        try:
            r_url = f"{host}/{rid}/raw/main/README.md"
            r_req = urllib.request.Request(r_url, headers={"User-Agent": "EasyColab/2.0"})
            if token:
                r_req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(r_req, timeout=6) as r_resp:
                raw_md = r_resp.read(32768).decode("utf-8", errors="ignore")
                readme_excerpt = _clean_hf_readme_excerpt(raw_md)
                if readme_excerpt:
                    break
        except Exception:
            continue

    return {
        "ok": True,
        "repo_id": rid,
        "name": rid.split("/")[-1],
        "author": str(meta.get("author") or rid.split("/")[0]),
        "pipeline_tag": ptag or ("gguf" if is_gguf else "text-generation"),
        "modality": modality,
        "is_gguf": is_gguf,
        "is_gated": is_gated,
        "license": lic,
        "base_model": base_model,
        "architecture": arch_str,
        "context_length": ctx_len,
        "languages": langs,
        "likes": int(meta.get("likes") or 0),
        "downloads": int(meta.get("downloads") or 0),
        "last_modified": str(meta.get("lastModified") or "")[:10],
        "used_storage_gb": used_storage_gb,
        "param_count": sizing["param_count"],
        "param_label": sizing["param_label"],
        "est_vram_gb": sizing["est_vram_gb"],
        "deploy_status": sizing["deploy_status"],
        "deploy_label_zh": sizing["deploy_label_zh"],
        "deploy_label_en": sizing["deploy_label_en"],
        "weight_files": weight_files[:36],
        "readme_excerpt": readme_excerpt,
    }

def inspect_and_resolve_hf_repo(
    repo_id: str,
    include_pattern: str = "",
    preferred_modality: str = "auto",
    hf_token: str = "",
) -> Dict[str, Any]:
    """
    通过 HuggingFace Hub API 实时扫描任意自定义模型仓库，支持通配符过滤 (include_pattern)、
    多档位 GGUF 智能优选、分片自动聚合与架构识别。
    """
    import fnmatch

    repo_clean, pat_clean = _normalize_hf_repo_and_pattern(repo_id, include_pattern)
    if "/" not in repo_clean or len(repo_clean.split("/")) != 2:
        raise ValueError(f"无效的 HuggingFace 仓库地址: '{repo_id}'（格式应为 Owner/Repo）")

    repo_low = repo_clean.lower()
    pat_low = pat_clean.lower()
    # 兼容已知特定多模态/图像生成仓库结构（如用户在客户端手动导入）
    if repo_low == "minimaxai/minimax-h3" and "ref2va" in pat_low:
        return {
            "repo_id": repo_clean,
            "include_pattern": "Ref2VA/*",
            "modality": "video_audio",
            "pipeline_arch": "minimax_h3_ref2va",
            "category": "原生内置专享 · 多张角色/主体参考图生视频 (Ref2VA) · 32kHz 立体声",
            "approx_size_gb": 134.1,
            "files": list(MINIMAX_H3_REF2VA_BF16_FILES),
        }
    if not pat_clean:
        if repo_low == "minimaxai/minimax-h3-image" or ("minimax-h3-image" in repo_low and "gguf" not in repo_low):
            m_img_spec = MODEL_CATALOG["minimax_h3_image"]
            return {
                "repo_id": repo_clean,
                "include_pattern": "",
                "modality": "image",
                "pipeline_arch": "minimax_h3_image",
                "category": "原生内置专享 · 6 步 4K (16MP) 文生图与多参考图指令修图",
                "approx_size_gb": m_img_spec["approx_size_gb"],
                "files": list(MINIMAX_H3_IMAGE_FILES),
            }
        if repo_low == "minimaxai/minimax-h3-ref2va" or ("minimax-h3" in repo_low and "ref2va" in repo_low):
            return {
                "repo_id": "MiniMaxAI/MiniMax-H3",
                "include_pattern": "Ref2VA/*",
                "modality": "video_audio",
                "pipeline_arch": "minimax_h3_ref2va",
                "category": "原生内置专享 · 多张角色/主体参考图生视频 (Ref2VA) · 32kHz 立体声",
                "approx_size_gb": 134.1,
                "files": list(MINIMAX_H3_REF2VA_BF16_FILES),
            }
        if repo_low == "minimaxai/minimax-h3" or ("minimax-h3" in repo_low and "image" not in repo_low and "gguf" not in repo_low):
            return {
                "repo_id": repo_clean,
                "include_pattern": "",
                "modality": "video_audio",
                "pipeline_arch": "minimax_h3",
                "category": "自定义导入 · 全模态音视频生成 · BF16 · 32kHz 立体声",
                "approx_size_gb": 134.1,
                "files": list(MINIMAX_H3_BF16_FILES),
            }
        if repo_low == "qwen/qwen-image-2.1" or ("qwen-image-2.1" in repo_low and "gguf" not in repo_low):
            q_spec = MODEL_CATALOG["qwen_image_21"]
            return {
                "repo_id": repo_clean,
                "include_pattern": "",
                "modality": "image",
                "pipeline_arch": "qwen_image_21",
                "category": "自定义导入 · 旗舰文生图与多图指令修图",
                "approx_size_gb": q_spec["approx_size_gb"],
                "files": list(q_spec["files"]),
            }
        if repo_low == "tongyi-mai/z-image-turbo" or ("z-image-turbo" in repo_low and "gguf" not in repo_low):
            z_spec = MODEL_CATALOG["z_image_turbo"]
            return {
                "repo_id": repo_clean,
                "include_pattern": "",
                "modality": "image",
                "pipeline_arch": "z_image_turbo",
                "category": "自定义导入 · 8 步极速超清写实生图",
                "approx_size_gb": z_spec["approx_size_gb"],
                "files": list(z_spec["files"]),
            }

    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
    headers = {"User-Agent": "EasyColab/2.0", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    meta = None
    last_http_code = 0
    last_err = None
    for host in ("https://huggingface.co", "https://hf-mirror.com"):
        api_url = f"{host}/api/models/{repo_clean}?blobs=true"
        req = urllib.request.Request(api_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                meta = json.loads(resp.read().decode("utf-8"))
                break
        except urllib.error.HTTPError as he:
            last_http_code = he.code
            last_err = he
            break
        except Exception as ex:
            last_err = ex
            continue

    if meta is None:
        if last_http_code in (401, 403):
            if not token:
                raise RuntimeError(
                    f"访问受限/私有仓库 {repo_clean} 需要登录鉴权 (HTTP {last_http_code})："
                    f"请先在弹窗填写 HuggingFace Access Token (hf_...) 后重试（获取地址: https://huggingface.co/settings/tokens）。"
                )
            raise RuntimeError(
                f"访问受限仓库 {repo_clean} 被拒绝 (HTTP {last_http_code})："
                f"请先在浏览器打开 https://huggingface.co/{repo_clean} 登录并点击「Agree」同意该模型许可协议，同时确认 HF Token 具备 Read 权限。"
            )
        elif last_http_code == 404:
            if not token:
                raise RuntimeError(
                    f"未找到仓库 {repo_clean} (HTTP 404)：请检查 Repo ID 拼写；若该仓库为私有仓库，请填写有效的 HF Token (hf_...)。"
                )
            raise RuntimeError(f"在 HuggingFace 上未找到仓库: {repo_clean} (HTTP 404)")
        raise RuntimeError(f"查询 HuggingFace 仓库 {repo_clean} 失败: {last_err}")

    gated_raw = meta.get("gated")
    is_gated = bool(gated_raw) and str(gated_raw).lower() not in ("false", "0", "none")
    if is_gated and not token:
        raise RuntimeError(
            f"模型 {repo_clean} 为受限授权模型 (Gated Repo 🔒)："
            f"请先在网页端 https://huggingface.co/{repo_clean} 点击 Agree 同意协议，并在下方填写您的 HuggingFace Token (hf_...) 后再导入。"
        )

    siblings = meta.get("siblings") or []
    file_entries: List[Tuple[str, int]] = []
    all_rfilenames: List[str] = []
    for s in siblings:
        rfn = str(s.get("rfilename") or "").strip()
        if not rfn:
            continue
        all_rfilenames.append(rfn.lower())
        sz = int(s.get("size") or (s.get("lfs") or {}).get("size") or 0)
        file_entries.append((rfn, sz))

    if not file_entries:
        raise RuntimeError(f"仓库 {repo_clean} 中未包含任何可用文件")

    def _fmt_sz(b: int) -> str:
        gb = b / (1024 ** 3)
        if gb >= 0.1:
            return f"约 {gb:.2f} GB"
        mb = b / (1024 ** 2)
        return f"约 {mb:.1f} MB" if mb >= 1.0 else "权重文件"

    raw_patterns = [p.strip() for p in re.split(r"[,;\s]+", pat_clean) if p.strip()] if pat_clean else []
    selected_entries: List[Tuple[str, int]] = []

    if raw_patterns:
        for rfn, sz in file_entries:
            rfn_low = rfn.lower()
            base_low = os.path.basename(rfn_low)
            matched = False
            for pat in raw_patterns:
                pl = pat.lower()
                glob_pat = pl if ("*" in pl or "?" in pl or "[" in pl) else f"*{pl}*"
                if fnmatch.fnmatch(rfn_low, glob_pat) or fnmatch.fnmatch(base_low, glob_pat) or pl in rfn_low:
                    matched = True
                    break
            if matched and not rfn_low.endswith((".md", ".txt", ".gitattributes", ".png", ".jpg", ".gif")):
                selected_entries.append((rfn, sz))
        if not selected_entries:
            raise RuntimeError(
                f"在仓库 {repo_clean} 中未找到匹配 '{pat_clean}' 的权重文件，请检查通配符或文件名是否正确"
            )
        # 若匹配的是 GGUF 主模型且仓库内含有 mmproj 多模态投影器，自动附带最佳 mmproj
        if any(rfn.lower().endswith(".gguf") for rfn, _ in selected_entries) and not any(
            "mmproj" in rfn.lower() for rfn, _ in selected_entries
        ):
            mmprojs = [(r, s) for r, s in file_entries if r.lower().endswith(".gguf") and "mmproj" in r.lower()]
            if mmprojs:
                mmprojs.sort(key=lambda x: (0 if "f16" in x[0].lower() else (1 if "bf16" in x[0].lower() else 2), x[0]))
                selected_entries.append(mmprojs[0])
    else:
        gguf_all = [(r, s) for r, s in file_entries if r.lower().endswith(".gguf")]
        if gguf_all:
            mmprojs = [(r, s) for r, s in gguf_all if "mmproj" in r.lower()]
            mtps = [(r, s) for r, s in gguf_all if "mtp" in r.lower() and "mmproj" not in r.lower()]
            mains = [
                (r, s)
                for r, s in gguf_all
                if "mmproj" not in r.lower()
                and "mtp" not in r.lower()
                and "imatrix" not in r.lower()
                and not os.path.basename(r.lower()).startswith("ggml-vocab")
            ]
            if not mains:
                mains = gguf_all

            # 将分片文件 (如 *-00001-of-00003.gguf) 归并为同一量化组
            groups: Dict[str, List[Tuple[str, int]]] = {}
            for r, s in mains:
                key = re.sub(r"-\d{5}-of-\d{5}\.gguf$", ".gguf", r, flags=re.IGNORECASE)
                groups.setdefault(key, []).append((r, s))

            quant_priority = [
                "ud-q4_k_m", "q4_k_m", "q4_k_l", "iq4_xs", "q4_k_s",
                "ud-q5_k_m", "q5_k_m", "q6_k", "q8_0", "q4_0",
                "ud-q3_k_m", "q3_k_m", "iq3_m", "iq2_xs", "bf16", "f16",
            ]
            best_key = next(iter(groups.keys()))
            best_rank = 999
            for k in groups.keys():
                kl = k.lower()
                for idx_p, qp in enumerate(quant_priority):
                    if qp in kl:
                        if idx_p < best_rank:
                            best_rank = idx_p
                            best_key = k
                        break

            selected_entries.extend(sorted(groups[best_key], key=lambda x: x[0]))
            if mmprojs:
                mmprojs.sort(key=lambda x: (0 if "f16" in x[0].lower() else (1 if "bf16" in x[0].lower() else 2), x[0]))
                selected_entries.append(mmprojs[0])
            if mtps:
                mtps.sort(key=lambda x: (0 if "q4_0" in x[0].lower() else 1, x[0]))
                selected_entries.append(mtps[0])
        else:
            st_files = [
                (r, s)
                for r, s in file_entries
                if r.lower().endswith(".safetensors") and not os.path.basename(r).startswith(".")
            ]
            if st_files:
                selected_entries.extend(sorted(st_files, key=lambda x: x[0]))
            else:
                bin_files = [
                    (r, s)
                    for r, s in file_entries
                    if r.lower().endswith((".bin", ".pt", ".pth"))
                    and "training_args" not in r.lower()
                    and "optimizer" not in r.lower()
                ]
                if bin_files:
                    selected_entries.extend(sorted(bin_files, key=lambda x: x[0]))

    if not selected_entries:
        raise RuntimeError(f"在仓库 {repo_clean} 中未检测到 .gguf 或 .safetensors 模型权重文件")

    has_gguf = any(r.lower().endswith(".gguf") for r, _ in selected_entries)
    total_bytes = sum(max(0, s) for _, s in selected_entries)
    approx_gb = round(total_bytes / (1024 ** 3), 2) if total_bytes > 0 else 8.0

    pipeline_tag = str(meta.get("pipeline_tag") or "").lower()
    tags_low = [str(t).lower() for t in (meta.get("tags") or [])]
    has_config_json = any(r == "config.json" or r.endswith("/config.json") for r in all_rfilenames)
    has_model_index = any(r == "model_index.json" or r.endswith("/model_index.json") for r in all_rfilenames)

    if not has_gguf and pipeline_tag in _UNSUPPORTED_HF_PIPELINE_TAGS:
        raise RuntimeError(
            f"仓库 {repo_clean} 为「{pipeline_tag}」专用算法管线，仅包含自定义组件权重，"
            f"无法作为标准对话/视觉/生图大模型直接装载。请选用 GGUF 量化模型、标准 Transformers 对话/多模态模型或 Diffusers 图像模型。"
        )

    if preferred_modality in ("llm_vlm", "image", "video_audio"):
        modality = preferred_modality
    elif has_gguf:
        modality = "llm_vlm"
    elif "text-to-video" in pipeline_tag or "image-to-video" in pipeline_tag or "video" in pipeline_tag:
        modality = "video_audio"
    elif "text-to-image" in pipeline_tag or "image-to-image" in pipeline_tag or "diffusers" in tags_low or (has_model_index and not has_config_json):
        modality = "image"
    else:
        modality = "llm_vlm"

    if modality == "llm_vlm" and not has_gguf and not has_config_json:
        raise RuntimeError(
            f"仓库 {repo_clean} 未包含标准 Transformers 配置文件 (config.json) 或 GGUF 权重，"
            f"仅包含独立权重分片（如 3D 重建 / 专用 SDK / ComfyUI 单体组件），无法作为对话或视觉大模型直接加载。"
        )

    if has_gguf:
        pipeline_arch = "gguf"
    elif modality == "video_audio":
        pipeline_arch = "minimax_h3"
    elif modality == "image":
        if "minimax-h3-image" in repo_low:
            pipeline_arch = "minimax_h3_image"
        elif "z-image" in repo_low:
            pipeline_arch = "z_image_turbo"
        elif "qwen-image" in repo_low:
            pipeline_arch = "qwen_image_21"
        else:
            pipeline_arch = "diffusers_image"
    else:
        pipeline_arch = "transformers"

    files_spec: List[Tuple[str, str]] = []
    total_cnt = len(selected_entries)
    for i, (rfn, sz) in enumerate(selected_entries, start=1):
        rl = rfn.lower()
        if "mmproj" in rl:
            desc = f"多模态视觉投影器 ({_fmt_sz(sz)})"
        elif "mtp" in rl and rl.endswith(".gguf"):
            desc = f"MTP 多词元投机解码头 ({_fmt_sz(sz)})"
        elif rl.endswith(".gguf"):
            desc = f"GGUF 量化权重 {i}/{total_cnt} ({_fmt_sz(sz)})" if total_cnt > 1 else f"GGUF 主模型权重 ({_fmt_sz(sz)})"
        else:
            desc = f"模型权重分片 {i}/{total_cnt} ({_fmt_sz(sz)})"
        files_spec.append((rfn, desc))

    arch_badge = "GGUF (llama.cpp)" if has_gguf else ("Safetensors (BF16)" if modality == "llm_vlm" else "DiffSynth / Diffusers")
    pat_tag = f" · 过滤: {pat_clean}" if pat_clean else ""
    category = f"自定义 HF 模型 · {arch_badge} · {approx_gb:.1f} GB{pat_tag}"

    return {
        "repo_id": repo_clean,
        "include_pattern": pat_clean,
        "modality": modality,
        "pipeline_arch": pipeline_arch,
        "category": category,
        "approx_size_gb": approx_gb,
        "files": files_spec,
    }

def register_custom_hf_model_core(
    repo_id: str,
    custom_name: str = "",
    include_pattern: str = "",
    modality: str = "auto",
    hf_token: str = "",
    model_id_override: str = "",
    modality_hint: str = "",
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    # (shared state on runtime_state)
    touch_activity()
    try:
        effective_modality = (modality_hint or modality or "auto").strip()
        if hf_token and hf_token.strip():
            os.environ["HF_TOKEN"] = hf_token.strip()
        resolved = inspect_and_resolve_hf_repo(
            repo_id=repo_id,
            include_pattern=include_pattern,
            preferred_modality=effective_modality,
            hf_token=hf_token,
        )
        repo_clean = resolved["repo_id"]
        pat_clean = resolved["include_pattern"]
        if resolved.get("pipeline_arch") == "minimax_h3_ref2va":
            rs.SELECTED_MODEL_ID = "minimax_h3_ref2va"
            builtin_spec = MODEL_CATALOG["minimax_h3_ref2va"]
            msg = (
                "已自动切换至内置专属引擎「MiniMax-H3-Ref2VA (33B · 多参考图)」"
                "（仅需增量下载 13 个 Ref2VA DiT 分片，自动零拷贝共享 MiniMax-H3 的 72.4GB 编码器与音视频 VAE）！"
            )
            MODEL_DOWNLOAD_STATE["message"] = msg
            print(f"[MiniMax-H3-Ref2VA] {msg}")
            return True, msg, builtin_spec
        if repo_clean.lower() == "minimaxai/minimax-h3-image" and not pat_clean:
            with CUSTOM_MODELS_LOCK:
                stale_ids = [
                    cid for cid, cspec in MODEL_CATALOG.items()
                    if cspec.get("is_custom") and str(cspec.get("repo_id") or "").lower() == "minimaxai/minimax-h3-image"
                ]
                for cid in stale_ids:
                    MODEL_CATALOG.pop(cid, None)
                if stale_ids:
                    _save_persisted_custom_models()
            rs.SELECTED_MODEL_ID = "minimax_h3_image"
            builtin_spec = MODEL_CATALOG["minimax_h3_image"]
            msg = (
                "已自动切换至内置专属引擎「MiniMax-H3-Image (33B · 4K)」"
                "（仅下载 Diffusers 必需 27 个分片，并自动零拷贝共享 MiniMax-H3 的 66.7GB 编码器权重）！"
            )
            MODEL_DOWNLOAD_STATE["message"] = msg
            print(f"[MiniMax-H3-Image] {msg}")
            return True, msg, builtin_spec
        mid = (model_id_override or "").strip() or _make_custom_model_id(repo_clean, pat_clean)
        if mid in BUILTIN_MODEL_IDS:
            mid = f"custom_{mid}"

        default_title = repo_clean.split("/")[-1]
        if pat_clean and len(pat_clean) <= 24 and "*" in pat_clean:
            clean_pat_lbl = pat_clean.replace("*", "").replace(".gguf", "").strip("-_.")
            if clean_pat_lbl and clean_pat_lbl.lower() not in default_title.lower():
                default_title = f"{default_title} ({clean_pat_lbl})"
        display_name = (custom_name or "").strip() or default_title

        spec = {
            "id": mid,
            "name": display_name,
            "repo_id": repo_clean,
            "include_pattern": pat_clean,
            "modality": resolved["modality"],
            "pipeline_arch": resolved["pipeline_arch"],
            "category": resolved["category"],
            "approx_size_gb": resolved["approx_size_gb"],
            "files": resolved["files"],
            "is_custom": True,
        }
        with CUSTOM_MODELS_LOCK:
            MODEL_CATALOG[mid] = spec
            _save_persisted_custom_models()
        with _TELEMETRY_LOCK:
            _CACHED_FILE_STATUS_MAP.pop(mid, None)

        rs.SELECTED_MODEL_ID = mid
        msg = (
            f"已成功导入自定义模型「{display_name}」({repo_clean} · 共 {len(spec['files'])} 个文件 / "
            f"约 {spec['approx_size_gb']:.1f} GB)，可立即下载权重并加载到显存！"
        )
        MODEL_DOWNLOAD_STATE["message"] = msg
        print(f"[自定义模型] {msg}")
        return True, msg, spec
    except Exception as e:
        err = f"导入自定义 HuggingFace 仓库失败: {str(e)}"
        print(f"[自定义模型] {err}")
        return False, err, None

def remove_custom_hf_model_core(model_id: str, delete_files: bool = False) -> Tuple[bool, str]:
    # (shared state on runtime_state)
    touch_activity()
    mid = (model_id or "").strip()
    if not mid or mid in BUILTIN_MODEL_IDS or mid not in MODEL_CATALOG:
        return False, "只能移除用户自定义添加的模型"
    name = MODEL_CATALOG[mid].get("name", mid)
    if mid in LOADED_RUNTIMES:
        from .media_pipeline import unload_model_from_vram_core
        rs.resolve_callable("unload_model_from_vram_core", unload_model_from_vram_core)(mid)
    with CUSTOM_MODELS_LOCK:
        MODEL_CATALOG.pop(mid, None)
        _save_persisted_custom_models()
    with _TELEMETRY_LOCK:
        _CACHED_FILE_STATUS_MAP.pop(mid, None)
    if rs.SELECTED_MODEL_ID == mid:
        rs.SELECTED_MODEL_ID = "z_image_turbo" if "z_image_turbo" in MODEL_CATALOG else next(iter(MODEL_CATALOG.keys()))
    msg = f"已从模型中心移除自定义模型「{name}」"
    MODEL_DOWNLOAD_STATE["message"] = msg
    print(f"[自定义模型] {msg}")
    return True, msg

_load_persisted_custom_models()

DEFAULT_OUTPUT_DIR = rs.DEFAULT_OUTPUT_DIR

LEGACY_DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/MiniMax_H3_Outputs"

LEGACY_DRIVE_OUTPUT_DIRS = [
    "/content/drive/MyDrive/Aevix_Outputs",
    "/content/drive/MyDrive/MiniMax_H3_Outputs",
]

def fast_parallel_hf_download(
    repo_id: str,
    filename: str,
    dest_dir: str,
    hf_token: str = "",
    num_connections: int = 8,
) -> str:
    """
    使用 8 路并发 HTTP Range 连接直接从 HuggingFace CDN 下载大权重文件至内存盘/本地盘，
    规避单连接 ~60MB/s 限速，在 Colab 上可达 400~700 MB/s。
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    opener = urllib.request.build_opener(_StripAuthRedirectHandler())
    headers = {"User-Agent": "EasyColab/2.0"}
    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        # Probe final CDN URL & Content-Length via Range: bytes=0-0
        probe_req = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-0"})
        with opener.open(probe_req, timeout=20) as resp:
            cdn_url = resp.geturl()
            content_range = resp.headers.get("Content-Range", "")
            if "/" in content_range:
                total_size = int(content_range.split("/")[-1])
            else:
                total_size = int(resp.headers.get("Content-Length", 0))

        # 若文件较小 (<32MB) 或未返回支持分片的大小，直接走 hf_hub_download
        if total_size < 32 * 1024 * 1024:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir, token=token or None)

        with open(temp_path, "wb") as f:
            f.truncate(total_size)

        downloaded_bytes = 0
        dl_lock = threading.Lock()
        start_ts = time.time()
        last_print_ts = [0.0]

        part_size = (total_size + num_connections - 1) // num_connections
        ranges = [
            (i * part_size, min((i + 1) * part_size - 1, total_size - 1))
            for i in range(num_connections)
            if i * part_size < total_size
        ]

        def _download_chunk(byte_range: Tuple[int, int]):
            nonlocal downloaded_bytes
            start_b, end_b = byte_range
            chunk_req = urllib.request.Request(
                cdn_url,
                headers={"User-Agent": "EasyColab/2.0", "Range": f"bytes={start_b}-{end_b}"},
            )
            with urllib.request.urlopen(chunk_req, timeout=60) as r, open(temp_path, "r+b") as wf:
                wf.seek(start_b)
                while True:
                    buf = r.read(4 * 1024 * 1024)  # 4MB buffer
                    if not buf:
                        break
                    wf.write(buf)
                    now = time.time()
                    with dl_lock:
                        downloaded_bytes += len(buf)
                        if now - last_print_ts[0] >= 0.35 or downloaded_bytes >= total_size:
                            last_print_ts[0] = now
                            elapsed = max(0.001, now - start_ts)
                            speed_mbs = (downloaded_bytes / (1024 ** 2)) / elapsed
                            pct = int(downloaded_bytes * 100 / total_size)
                            cur_gb = downloaded_bytes / (1024 ** 3)
                            tot_gb = total_size / (1024 ** 3)
                            eta_s = int(((total_size - downloaded_bytes) / (1024 ** 2)) / max(0.1, speed_mbs))
                            sys.stdout.write(
                                f"\r{filename}: {pct}%|{'█' * (pct // 10):<10}| "
                                f"{cur_gb:.2f}G/{tot_gb:.2f}G [{int(elapsed):02d}s<{eta_s:02d}s, {speed_mbs:.1f}MB/s]\n"
                            )
                            sys.stdout.flush()

        with ThreadPoolExecutor(max_workers=num_connections) as pool:
            futures = [pool.submit(_download_chunk, r) for r in ranges]
            for fut in as_completed(futures):
                fut.result()

        os.replace(temp_path, dest_path)
        return dest_path

    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "403" in err_str:
            if not token:
                raise RuntimeError(
                    f"下载受限仓库 {repo_id} 需要鉴权 (HTTP 401/403)：请先填写 HuggingFace Token (hf_...) 后重新下载。"
                )
            raise RuntimeError(
                f"下载受限仓库 {repo_id} 被拒绝 (HTTP 403 Gated Repo)：请先用浏览器打开 https://huggingface.co/{repo_id} 点击 Agree 同意协议后再试。"
            )
        print(f"ℹ️ 多线程分片直连回退至标准 HF 下载 ({e})...")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir, token=token or None)

def _get_model_dirs(model_id: str) -> Tuple[Optional[str], str, str]:
    """返回指定模型的 (RAM内存盘目录或None, 本地高速NVMe目录, Google Drive/持久盘目录)"""
    spec = MODEL_CATALOG.get(model_id, MODEL_CATALOG["minimax_h3"])
    repo_sub = spec["repo_id"]
    pipeline_arch = spec.get("pipeline_arch") or model_id
    ram_dir = None
    # MiniMax-H3 / MiniMax-H3-Ref2VA / MiniMax-H3-Image (~135GB) 推理期依赖 CPU DDR5 内存热备或 ComponentsManager 轮换，禁止拷入 /dev/shm 以免与匿名内存争抢 Cgroup 上限
    if os.path.exists("/dev/shm") and pipeline_arch not in ("minimax_h3_image", "minimax_h3", "minimax_h3_ref2va"):
        try:
            shm_free_gb = _safe_shm_available_bytes(reserve_ram_gb=34.0) / (1024 ** 3)
            # 仅当 /dev/shm 配额与容器真实可用物理内存均足以容纳模型时才把新模型放入内存盘
            if shm_free_gb >= spec["approx_size_gb"] + 3.0:
                ram_dir = f"/dev/shm/EasyColab_Models/{model_id}"
            elif spec["approx_size_gb"] <= 60.0 and (
                os.path.exists(f"/dev/shm/EasyColab_Models/{model_id}")
                or os.path.exists(f"/dev/shm/Aevix_Models/{model_id}")
                or os.path.exists(f"/dev/shm/OpenModels/{model_id}")
            ):
                ram_dir = f"/dev/shm/EasyColab_Models/{model_id}"
        except Exception:
            pass
    nvme_dir = os.path.join(os.getcwd(), "models", repo_sub)
    drive_dir = os.path.join(MODEL_BASE_ROOT, repo_sub)
    return ram_dir, nvme_dir, drive_dir

def _find_existing_model_file(model_id: str, fname: str, _allow_peer_share: bool = True) -> Optional[str]:
    """按速度优先级查找已存在的模型权重文件：1. /dev/shm 内存盘 -> 2. 本地 NVMe -> 3. Google Drive (含历史目录与跨模型共享编码器兼容)"""
    ram_dir, nvme_dir, drive_dir = _get_model_dirs(model_id)
    spec = MODEL_CATALOG.get(model_id, MODEL_CATALOG["minimax_h3"])
    repo_sub = spec["repo_id"]
    pipeline_arch = spec.get("pipeline_arch") or model_id
    candidates = []
    if ram_dir:
        candidates.append(os.path.join(ram_dir, fname))
    candidates.append(os.path.join(f"/dev/shm/EasyColab_Models/{model_id}", fname))
    candidates.append(os.path.join(f"/dev/shm/Aevix_Models/{model_id}", fname))
    candidates.append(os.path.join(f"/dev/shm/OpenModels/{model_id}", fname))
    candidates.append(os.path.join(nvme_dir, fname))
    candidates.append(os.path.join(drive_dir, fname))
    if DRIVE_MOUNTED:
        for leg_root in LEGACY_DRIVE_MODEL_ROOTS:
            if os.path.isdir(leg_root):
                candidates.append(os.path.join(leg_root, repo_sub, fname))
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 64 * 1024:
            return p

    # MiniMax-H3 (FL2VA) / MiniMax-H3-Ref2VA / MiniMax-H3-Image 的 14 个 32B VLM text_encoder 分片 (62.1 GB) 二进制 100% 一致，
    # 且 FL2VA 与 Ref2VA 的 3D 视频 VAE (9.7 GB) 及 32kHz 音频 VAE (0.56 GB) 同样 100% 一致，支持三方零拷贝共享免重复下载！
    if _allow_peer_share:
        if pipeline_arch == "minimax_h3_image" and fname.startswith("text_encoder/model-") and fname.endswith("-of-00014.safetensors"):
            for peer_id, peer_prefix in (("minimax_h3", "FL2VA/"), ("minimax_h3_ref2va", "Ref2VA/")):
                peer_hit = _find_existing_model_file(peer_id, f"{peer_prefix}{fname}", _allow_peer_share=False)
                if peer_hit:
                    return peer_hit
        elif pipeline_arch in ("minimax_h3", "minimax_h3_ref2va"):
            own_prefix = "Ref2VA/" if pipeline_arch == "minimax_h3_ref2va" else "FL2VA/"
            alt_video_id = "minimax_h3" if pipeline_arch == "minimax_h3_ref2va" else "minimax_h3_ref2va"
            alt_prefix = "FL2VA/" if pipeline_arch == "minimax_h3_ref2va" else "Ref2VA/"
            if fname.startswith(own_prefix):
                rel_tail = fname[len(own_prefix):]
                if rel_tail.startswith("text_encoder/model-") and rel_tail.endswith("-of-00014.safetensors"):
                    for peer_id, peer_fname in (
                        (alt_video_id, f"{alt_prefix}{rel_tail}"),
                        ("minimax_h3_image", rel_tail),
                    ):
                        peer_hit = _find_existing_model_file(peer_id, peer_fname, _allow_peer_share=False)
                        if peer_hit:
                            return peer_hit
                elif rel_tail in ("video_vae/source/model.safetensors", "audio_vae/model.safetensors"):
                    peer_hit = _find_existing_model_file(alt_video_id, f"{alt_prefix}{rel_tail}", _allow_peer_share=False)
                    if peer_hit:
                        return peer_hit
    return None

def check_model_files_status(model_id: Optional[str] = None) -> Dict[str, Any]:
    """检查指定开源模型在内存盘/磁盘/云盘上的权重文件完整性与大小"""
    norm_id = _normalize_builtin_model_alias(model_id)
    target_id = norm_id if (norm_id and norm_id in MODEL_CATALOG) else rs.SELECTED_MODEL_ID
    spec = MODEL_CATALOG[target_id]
    req_files = spec["files"]
    ram_dir, nvme_dir, drive_dir = _get_model_dirs(target_id)

    files_info = []
    downloaded_count = 0
    total_size_gb = 0.0
    ram_cached_count = 0

    for fname, desc in req_files:
        actual_path = _find_existing_model_file(target_id, fname)
        if actual_path:
            size_gb = round(os.path.getsize(actual_path) / (1024 ** 3), 2)
            total_size_gb += size_gb
            downloaded_count += 1
            in_ram = actual_path.startswith("/dev/shm/")
            if in_ram:
                ram_cached_count += 1
            files_info.append({
                "name": fname,
                "desc": desc + (" [RAM内存盘]" if in_ram else ""),
                "exists": True,
                "size_gb": size_gb,
            })
        else:
            files_info.append({
                "name": fname,
                "desc": desc,
                "exists": False,
                "size_gb": 0.0,
            })

    all_downloaded = downloaded_count == len(req_files) and len(req_files) > 0
    return {
        "selected_model_id": target_id,
        "selected_model_name": spec["name"],
        "selected_model_repo": spec["repo_id"],
        "selected_model_modality": spec["modality"],
        "files": files_info,
        "downloaded_count": downloaded_count,
        "total_count": len(req_files),
        "all_downloaded": all_downloaded,
        "total_size_gb": round(total_size_gb, 2),
        "drive_persisted": DRIVE_MOUNTED,
        "ramdisk_enabled": ram_dir is not None,
        "ram_cached_count": ram_cached_count,
        "model_storage_path": ram_dir or drive_dir,
        "bf16_shards_count": downloaded_count if target_id in ("minimax_h3", "minimax_h3_ref2va") else 0,
    }

_CACHED_FILE_STATUS_MAP: Dict[str, Tuple[float, Dict[str, Any]]] = {}

def _get_cached_model_files_status(target_id: str) -> Dict[str, Any]:
    """对磁盘/云盘 FUSE 权重扫描做轻量缓存，避免高频状态轮询在视频生成期间阻塞 I/O"""
    now = time.time()
    ttl = 1.8 if MODEL_DOWNLOAD_STATE["is_downloading"] else 12.0
    with _TELEMETRY_LOCK:
        cached = _CACHED_FILE_STATUS_MAP.get(target_id)
        if cached and (now - cached[0]) < ttl:
            return dict(cached[1])
    fresh = check_model_files_status(target_id)
    with _TELEMETRY_LOCK:
        _CACHED_FILE_STATUS_MAP[target_id] = (now, dict(fresh))
    return fresh

# ==============================================================================
# 独立功能 1：多模型通用极速下载器 (8路并发 HTTP Range + 后台 Google Drive 备份)
# ==============================================================================
def _async_backup_to_drive(src_file: str, dst_dir: str, fname: str):
    """在后台线程中将内存盘/NVMe下载好的权重同步到 Google Drive，不阻塞用户加载模型"""
    if not DRIVE_MOUNTED:
        return
    try:
        dst_file = os.path.join(dst_dir, fname)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        if os.path.exists(dst_file) and os.path.getsize(dst_file) == os.path.getsize(src_file):
            return
        tmp_dst = dst_file + ".syncing"
        with open(src_file, "rb") as rf, open(tmp_dst, "wb") as wf:
            shutil.copyfileobj(rf, wf, length=32 * 1024 * 1024)
        os.replace(tmp_dst, dst_file)
        print(f"[后台云盘同步完成] {fname} 已持久化保存至 Google Drive")
    except Exception as e:
        print(f"[后台云盘同步跳过] {fname}: {e}")

def download_model_weights_core(model_id: Optional[str] = None, hf_token: str = "") -> Tuple[bool, str]:
    # (shared state on runtime_state)
    touch_activity()
    if MODEL_DOWNLOAD_STATE["is_downloading"]:
        return False, "模型正在下载中，请稍候..."

    norm_id = _normalize_builtin_model_alias(model_id)
    target_id = norm_id if (norm_id and norm_id in MODEL_CATALOG) else rs.SELECTED_MODEL_ID
    rs.SELECTED_MODEL_ID = target_id
    spec = MODEL_CATALOG[target_id]
    repo_id = spec["repo_id"]
    req_files = spec["files"]

    MODEL_DOWNLOAD_STATE["is_downloading"] = True
    MODEL_DOWNLOAD_STATE["progress"] = 0.02
    MODEL_DOWNLOAD_STATE["message"] = f"正在为 {spec['name']} 启动 8 路并发 HTTP Range 极速下载通道..."
    with CONSOLE_LOCK:
        ACTIVE_TQDM_BARS.clear()

    try:
        if hf_token and hf_token.strip():
            os.environ["HF_TOKEN"] = hf_token.strip()
            from huggingface_hub import login
            login(token=hf_token.strip(), add_to_git_credential=False)

        ram_dir, nvme_dir, drive_dir = _get_model_dirs(target_id)
        already_ready_gb = 0.0
        for fn_chk, _ in req_files:
            ex_p = _find_existing_model_file(target_id, fn_chk)
            if ex_p and os.path.exists(ex_p):
                already_ready_gb += os.path.getsize(ex_p) / (1024 ** 3)
        missing_approx_gb = max(0.5, float(spec["approx_size_gb"]) - already_ready_gb)

        drive_has_space = False
        if DRIVE_MOUNTED and os.path.exists("/content/drive/MyDrive"):
            try:
                drive_free_gb = shutil.disk_usage("/content/drive/MyDrive").free / (1024 ** 3)
                drive_has_space = drive_free_gb >= missing_approx_gb + 2.0
                if not drive_has_space:
                    print(
                        f"ℹ️ Google Drive 剩余空间 ({drive_free_gb:.1f} GB) 小于 {spec['name']} 待下载增量 "
                        f"({missing_approx_gb:.1f} GB)，自动改用实例本地高速 NVMe 磁盘 ({nvme_dir}) 存储！"
                    )
            except Exception:
                drive_has_space = False

        primary_dl_dir = ram_dir if ram_dir else (drive_dir if (DRIVE_MOUNTED and drive_has_space) else nvme_dir)
        os.makedirs(primary_dl_dir, exist_ok=True)
        total_files = len(req_files)

        for idx, (fname, desc) in enumerate(req_files):
            touch_activity()
            existing = _find_existing_model_file(target_id, fname)
            base_prog = idx / float(total_files + 1)
            MODEL_DOWNLOAD_STATE["progress"] = round(base_prog + 0.04, 2)

            if existing:
                # 若使用 Transformers / Diffusers 目录加载或分片 GGUF，确保同一目录内有软链接以便按目录直接加载
                pipeline_arch = spec.get("pipeline_arch") or target_id
                if pipeline_arch in ("qwen38_27b", "transformers", "diffusers_image", "gguf"):
                    target_link = os.path.join(primary_dl_dir, fname)
                    os.makedirs(os.path.dirname(target_link), exist_ok=True)
                    if os.path.abspath(existing) != os.path.abspath(target_link) and not os.path.exists(target_link):
                        try:
                            os.symlink(existing, target_link)
                        except Exception:
                            pass
                size_gb = os.path.getsize(existing) / (1024 ** 3)
                MODEL_DOWNLOAD_STATE["message"] = f"[{idx+1}/{total_files}] {fname} 已就绪 ({size_gb:.2f} GB)，跳过下载"
                print(MODEL_DOWNLOAD_STATE["message"])
                continue

            MODEL_DOWNLOAD_STATE["message"] = f"[{idx+1}/{total_files}] 正在 8 线程并发下载 {fname} ({desc})..."
            print(MODEL_DOWNLOAD_STATE["message"])
            dl_path = rs.resolve_callable("fast_parallel_hf_download", fast_parallel_hf_download)(
                repo_id=repo_id,
                filename=fname,
                dest_dir=primary_dl_dir,
                hf_token=hf_token,
                num_connections=8,
            )
            if DRIVE_MOUNTED and drive_has_space and primary_dl_dir != drive_dir:
                threading.Thread(
                    target=_async_backup_to_drive,
                    args=(dl_path, drive_dir, fname),
                    daemon=True,
                ).start()

        MODEL_DOWNLOAD_STATE["progress"] = 0.92
        pipeline_arch = spec.get("pipeline_arch") or target_id
        if pipeline_arch in ("qwen38_27b_gguf", "gguf"):
            MODEL_DOWNLOAD_STATE["message"] = "正在预装并校验 llama.cpp CUDA 高速推理引擎..."
            from .llm_runtime import _ensure_llama_server_binary
            _ensure_llama_server_binary()
        else:
            MODEL_DOWNLOAD_STATE["message"] = f"正在获取 {spec['name']} 的分词器与 Processor 配置文件..."
            from huggingface_hub import snapshot_download

            if pipeline_arch in ("minimax_h3", "minimax_h3_ref2va"):
                from diffsynth.pipelines.minimax_h3_audio_video import ModelConfig
                proc_prefix = "Ref2VA" if pipeline_arch == "minimax_h3_ref2va" else "FL2VA"
                proc_cfg = ModelConfig(
                    model_id=repo_id or "MiniMaxAI/MiniMax-H3",
                    origin_file_pattern=f"{proc_prefix}/processor/",
                    download_source="huggingface",
                )
                proc_cfg.download_if_necessary()
            elif pipeline_arch in ("qwen38_27b", "transformers", "diffusers_image"):
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=primary_dl_dir,
                    ignore_patterns=["*.safetensors", "*.bin", "*.pth", "*.gguf", "*.pt"],
                    token=(hf_token or os.environ.get("HF_TOKEN") or "").strip() or None,
                )
            elif pipeline_arch == "qwen_image_21":
                from diffsynth.pipelines.qwen_image_21 import ModelConfig
                proc_cfg = ModelConfig(
                    model_id=repo_id or "Qwen/Qwen-Image-2.1",
                    origin_file_pattern="processor/",
                    download_source="huggingface",
                )
                proc_cfg.download_if_necessary()
            elif pipeline_arch == "z_image_turbo":
                from diffsynth.pipelines.z_image import ModelConfig
                tok_cfg = ModelConfig(
                    model_id=repo_id or "Tongyi-MAI/Z-Image-Turbo",
                    origin_file_pattern="tokenizer/",
                    download_source="huggingface",
                )
                tok_cfg.download_if_necessary()
            elif pipeline_arch == "minimax_h3_image":
                MODEL_DOWNLOAD_STATE["message"] = "正在同步 MiniMax-H3-Image 模块化配置与 Diffusers>=0.43 运行时..."
                from .media_pipeline import ensure_minimax_h3_image_dependencies
                ensure_minimax_h3_image_dependencies()
                from .media_pipeline import prepare_minimax_h3_image_assembled_dir
                prepare_minimax_h3_image_assembled_dir(
                    target_id=target_id,
                    repo_id=repo_id or "MiniMaxAI/MiniMax-H3-Image",
                    hf_token=hf_token,
                    link_weights=False,
                )

        MODEL_DOWNLOAD_STATE["progress"] = 1.0
        status = check_model_files_status(target_id)
        msg = f"{spec['name']} 全部 {status['downloaded_count']}/{status['total_count']} 个权重文件已就绪（共 {status['total_size_gb']:.2f} GB），现在可点击「加载到显存」！"
        MODEL_DOWNLOAD_STATE["message"] = msg
        print(msg)
        return True, msg

    except Exception as e:
        err_msg = f"下载 {spec['name']} 权重出错: {str(e)}"
        MODEL_DOWNLOAD_STATE["message"] = err_msg
        print(err_msg)
        return False, err_msg
    finally:
        with _TELEMETRY_LOCK:
            _CACHED_FILE_STATUS_MAP.clear()
        MODEL_DOWNLOAD_STATE["is_downloading"] = False

def download_minimax_h3_weights_core(hf_token: str = "") -> Tuple[bool, str]:
    return rs.resolve_callable("download_model_weights_core", download_model_weights_core)("minimax_h3", hf_token)

# ==============================================================================
# 独立功能 2：多模型通用显存加载引擎 (自动卸载旧模型释放显存 + 零拷贝极速加载)
# ==============================================================================
def _resolve_and_stage_model_file(model_id: str, fname: str) -> str:
    """解析模型权重路径；若位于较慢的云盘且 /dev/shm 内存盘充裕，自动预热至内存盘"""
    ram_dir, nvme_dir, drive_dir = _get_model_dirs(model_id)
    existing = _find_existing_model_file(model_id, fname)
    if not existing:
        return os.path.join(drive_dir, fname)

    if ram_dir and not existing.startswith("/dev/shm/"):
        ram_target = os.path.join(ram_dir, fname)
        if os.path.exists(ram_target) and os.path.getsize(ram_target) == os.path.getsize(existing):
            return ram_target
        try:
            os.makedirs(os.path.dirname(ram_target), exist_ok=True)
            free_shm = _safe_shm_available_bytes(reserve_ram_gb=34.0)
            file_sz = os.path.getsize(existing)
            if free_shm > file_sz + (2 * 1024 ** 3):
                t0 = time.time()
                print(f"正在将 {fname} ({file_sz/(1024**3):.2f} GB) 高速预热至 /dev/shm 内存盘...")
                tmp_ram = ram_target + ".staging"
                with open(existing, "rb") as rf, open(tmp_ram, "wb") as wf:
                    shutil.copyfileobj(rf, wf, length=32 * 1024 * 1024)
                os.replace(tmp_ram, ram_target)
                dt = max(0.01, time.time() - t0)
                print(f"{fname} 内存盘预热完成 ({file_sz/(1024**2)/dt:.0f} MB/s)")
                return ram_target
        except Exception as e:
            print(f"内存盘预热跳过 ({e})，直接读取原路径")
    return existing

def _stage_model_files_to_local_nvme(
    model_id: str,
    paths: List[str],
    rel_names: List[str],
    group_label: str,
) -> List[str]:
    """当分片仍需走磁盘映射且原文件位于慢速云盘 FUSE (/content/drive) 时，若本地 NVMe 空间充裕则整组顺序缓存至本地高速 SSD"""
    if not paths or len(paths) != len(rel_names):
        return paths
    _, nvme_dir, _ = _get_model_dirs(model_id)
    if not any(p.startswith("/content/drive/") for p in paths):
        return paths

    try:
        os.makedirs(nvme_dir, exist_ok=True)
        free_nvme_bytes = shutil.disk_usage(nvme_dir).free
        needed_bytes = 0
        stage_plan: List[Tuple[int, str, str, int]] = []
        result_paths = list(paths)

        for idx, (src_p, rel_name) in enumerate(zip(paths, rel_names)):
            if not src_p.startswith("/content/drive/") or not os.path.exists(src_p):
                continue
            src_sz = os.path.getsize(src_p)
            dst_p = os.path.join(nvme_dir, rel_name)
            if os.path.exists(dst_p) and os.path.getsize(dst_p) == src_sz:
                result_paths[idx] = dst_p
                continue
            needed_bytes += src_sz
            stage_plan.append((idx, src_p, dst_p, src_sz))

        if not stage_plan:
            return result_paths

        # 保留至少 22GB 本地磁盘安全余量，避免挤占视频导出与系统空间
        if free_nvme_bytes <= needed_bytes + (22 * 1024 ** 3):
            return result_paths

        total_gb = needed_bytes / (1024 ** 3)
        print(f"⚡ [NVMe 高速缓存] 正在将 {group_label} ({len(stage_plan)} 个分片 / {total_gb:.1f} GB) 从云盘顺序预热至本地 NVMe SSD...")
        t_all = time.time()
        for step_i, (idx, src_p, dst_p, src_sz) in enumerate(stage_plan, start=1):
            os.makedirs(os.path.dirname(dst_p), exist_ok=True)
            MODEL_DOWNLOAD_STATE["message"] = (
                f"正在将 {group_label} 高速缓存至本地 SSD ({step_i}/{len(stage_plan)})，消除后续换入延迟..."
            )
            tmp_dst = dst_p + ".staging"
            with open(src_p, "rb") as rf, open(tmp_dst, "wb") as wf:
                shutil.copyfileobj(rf, wf, length=32 * 1024 * 1024)
            os.replace(tmp_dst, dst_p)
            result_paths[idx] = dst_p
        dt_all = max(0.05, time.time() - t_all)
        print(f"⚡ [NVMe 高速缓存] {group_label} 本地 SSD 预热完成 ({total_gb / dt_all * 1024:.0f} MB/s)")
        return result_paths
    except Exception as e:
        print(f"ℹ️ 本地 NVMe 预热跳过 ({e})，继续读取原路径")
        return paths

def get_history_videos(save_dir: str) -> List[str]:
    if not os.path.exists(save_dir):
        return []
    files = glob.glob(os.path.join(save_dir, "*.mp4"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:12]

TOYBOX_REPO_URL = "https://github.com/dseditor/minimax-h3-colab.git"

TOYBOX_ENHANCER_REPO_URL = "https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8.git"

TOYBOX_CUSTOM_NODES = [
    ("minimax-h3-audio-t8", "https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git"),
    ("comfyui-SelfLift", "https://github.com/facok/comfyui-SelfLift.git"),
    ("ComfyUI-H3-Motion-Context", "https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context.git"),
]

TOYBOX_CORE_MODELS = [
    ("diffusion_models", "Minimax-h3_Singularity_x_10Eros-Max_PRISM_int8.safetensors", "OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM", "Minimax-h3_Singularity_x_10Eros-Max_PRISM_int8.safetensors", 19.53),
    ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 14.61),
    ("vae", "minimax_h3_video_vae_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_int8_convrot.safetensors", 2.62),
    ("vae", "minimax_h3_audio_vae_fp32.safetensors", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", 0.56),
    ("loras", "pika_h3_flex_ref2va_v1_bf16.safetensors", "Comfy-Org/MiniMax-H3", "loras/pika_h3_flex_ref2va_v1_bf16.safetensors", 1.80),
    ("loras", "minimax_h3_turbo_v4_step600_ema_DasiwaREF2VAHybridV1_0_curveproj1025_compat_v001.safetensors", "dseditor/minimax-h3-studio-assets", "loras/minimax_h3_turbo_v4_step600_ema_DasiwaREF2VAHybridV1_0_curveproj1025_compat_v001.safetensors", 1.80),
    ("loras", "MysticXXX_MMH3-V2.safetensors", "dseditor/minimax-h3-studio-assets", "loras/MysticXXX_MMH3-V2.safetensors", 0.60),
    ("loras", "H3_Combat_V2.safetensors", "dseditor/minimax-h3-studio-assets", "loras/H3_Combat_V2.safetensors", 0.60),
    ("loras", "Motion_Repair.safetensors", "dseditor/minimax-h3-studio-assets", "loras/Motion_Repair.safetensors", 0.60),
    ("loras", "H3_SPEED_slider.safetensors", "dseditor/minimax-h3-studio-assets", "loras/H3_SPEED_slider.safetensors", 0.40),
    ("semantic_bridge", "t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors", "dseditor/minimax-h3-studio-assets", "semantic_bridge/t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors", 0.80),
    ("latent_upscale_models", "minimax_h3_latent_upscaler_3d_fp16_V2.safetensors", "dseditor/minimax-h3-studio-assets", "latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16_V2.safetensors", 0.50),
]

TOYBOX_REWRITER_MODELS = [
    ("text_encoders", "minimax_h3_rewriter_ref2va_qwen3.5_9b_w4a8.safetensors", "dseditor/minimax-h3-studio-assets", "text_encoders/minimax_h3_rewriter_ref2va_qwen3.5_9b_w4a8.safetensors", 7.00),
]

TOYBOX_IMAGE_MODELS = [
    ("loras", "p-h3-image_lora_bf16.safetensors", "PrunaAI/P-H3-Image-Comfy", "loras/p-h3-image_lora_bf16.safetensors", 2.00),
    ("text_encoders", "p-h3-image_text_encoder_int8_convrot.safetensors", "PrunaAI/P-H3-Image-Comfy", "text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors", 6.00),
    ("vae", "p-h3-image_vae_int8_convrot.safetensors", "PrunaAI/P-H3-Image-Comfy", "vae/p-h3-image_vae_int8_convrot.safetensors", 2.00),
]

def _normalize_hf_repo_id(raw_repo: str) -> str:
    """规范化 HuggingFace 仓库 ID。"""
    return _normalize_hf_repo_and_pattern(raw_repo)[0]
