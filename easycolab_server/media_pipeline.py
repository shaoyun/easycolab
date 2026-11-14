"""EasyColab 服务端模块: media_pipeline."""

from PIL import Image
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import base64
import gc
import gradio as gr
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from . import runtime_state as rs
from .runtime_state import torch
from .llm_runtime import _CausalLMProcessorAdapter, _is_llm_model_id, _run_qwen_llm_job, _start_qwen_gguf_server, _stop_gguf_server_bundle
from .model_catalog import DEFAULT_OUTPUT_DIR, MODEL_CATALOG, _async_backup_to_drive, _find_existing_model_file, _get_model_dirs, _normalize_builtin_model_alias, _resolve_and_stage_model_file, _stage_model_files_to_local_nvme, check_model_files_status, fast_parallel_hf_download, get_history_videos
from .qwen_kernels import GenerationAbortedError, _clamp_context_length, _ensure_qwen_fast_kernels, _ensure_torchaudio_compatible, _optimize_qwen35_for_fast_inference
from .runtime_state import ABORT_GENERATION_EVENT, ACTIVE_TQDM_BARS, COMFYUI_STATE, CONSOLE_LOCK, DRIVE_MOUNTED, GPU_LOCK, JOBS, LOADED_RUNTIMES, MODEL_DOWNLOAD_STATE, _JOB_CANCEL_EVENTS, _get_system_ram_gb, decode_base64_image, torch, touch_activity
from .schemas import GenerateVideoRequest

def build_structured_prompt(
    visual_prompt: str,
    soundscape_prompt: str,
    music_prompt: str,
) -> str:
    visual_prompt = (visual_prompt or "").strip()
    soundscape_prompt = (soundscape_prompt or "").strip()
    music_prompt = (music_prompt or "").strip()

    if not soundscape_prompt and not music_prompt:
        return visual_prompt

    parts = [visual_prompt]
    if soundscape_prompt:
        parts.append(f"Soundscape & Dialogue: {soundscape_prompt}")
    if music_prompt:
        parts.append(f"Background Music: {music_prompt}")
    return "\n".join(parts)

# ==============================================================================
# 专属独立引擎：MiniMax-H3-Image (33B · 6步原生4K ModularPipeline + 共享编码器组装)
# ==============================================================================
MINIMAX_H3_IMAGE_CONFIG_PATTERNS = [
    "modular_model_index.json",
    "pdd_config.json",
    "scheduler/*",
    "tokenizer/*",
    "processor/*",
    "text_encoder/*.json",
    "text_encoder/*.txt",
    "transformer_diffusers/*.json",
    "vae_diffusers/*.json",
]

MINIMAX_H3_IMAGE_REQUIRED_CONFIG_MARKERS = (
    "modular_model_index.json",
    "transformer_diffusers/config.json",
    "vae_diffusers/config.json",
    "text_encoder/config.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "processor/preprocessor_config.json",
)

def ensure_minimax_h3_image_dependencies() -> Tuple[Any, Any]:
    """
    确保当前环境已安装支持 MiniMaxH3ImageModularPipeline 的最新版 diffusers (>=0.43.0)。
    若 Colab 预装版本较旧，自动静默升级并热刷新 sys.modules。
    """
    try:
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.modular_pipelines.minimax_h3_image import MiniMaxH3ImageModularPipeline  # noqa: F401

        return ComponentsManager, ModularPipeline
    except Exception:
        print("[MiniMax-H3-Image] 检测到当前 diffusers 版本不含 MiniMaxH3ImageModularPipeline，正在升级 diffusers>=0.43.0 ...")
        MODEL_DOWNLOAD_STATE["message"] = "正在自动升级 diffusers>=0.43.0 以支持 MiniMax-H3-Image 模块化管线..."
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-U",
                "diffusers>=0.43.0",
                "transformers>=4.57.0",
                "accelerate",
            ]
        )
        import importlib

        for mod_key in [k for k in list(sys.modules.keys()) if k == "diffusers" or k.startswith("diffusers.")]:
            sys.modules.pop(mod_key, None)
        importlib.invalidate_caches()
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.modular_pipelines.minimax_h3_image import MiniMaxH3ImageModularPipeline  # noqa: F401

        print("[MiniMax-H3-Image] diffusers>=0.43.0 热升级校验通过！")
        return ComponentsManager, ModularPipeline

def prepare_minimax_h3_image_assembled_dir(
    target_id: str = "minimax_h3_image",
    repo_id: str = "MiniMaxAI/MiniMax-H3-Image",
    hf_token: str = "",
    link_weights: bool = True,
    stage_nvme_if_low_ram: bool = False,
) -> str:
    """
    在实例本地高速 ext4 文件系统构建 MiniMax-H3-Image 零拷贝软链组装目录：
    1. 仅拉取 ~35MB 轻量配置文件 (modular_model_index.json / processor / tokenizer / config.json)，彻底跳过仓库内重复的 68.6GB SGLang 权重；
    2. 在 Google Drive 挂载时双向缓存轻量配置文件，二次重启免联网校验；
    3. 将 27 个必需 .safetensors 分片（含自动复用的 MiniMax-H3 FL2VA/text_encoder 66.7GB 编码器分片）软链至本地组装目录，规避云盘 FUSE 不支持软链问题。
    """
    assembled_dir = os.path.join(os.getcwd(), "models", "_assembled", target_id)
    os.makedirs(assembled_dir, exist_ok=True)
    _, _, drive_dir = _get_model_dirs(target_id)
    drive_cfg_cache = os.path.join(drive_dir, "_modular_configs")

    def _configs_complete(base_d: str) -> bool:
        return all(os.path.exists(os.path.join(base_d, m)) for m in MINIMAX_H3_IMAGE_REQUIRED_CONFIG_MARKERS)

    if not _configs_complete(assembled_dir):
        if DRIVE_MOUNTED and _configs_complete(drive_cfg_cache):
            shutil.copytree(drive_cfg_cache, assembled_dir, dirs_exist_ok=True)
        if not _configs_complete(assembled_dir):
            from huggingface_hub import snapshot_download

            tok = (hf_token or os.environ.get("HF_TOKEN") or "").strip() or None
            snapshot_download(
                repo_id=repo_id or "MiniMaxAI/MiniMax-H3-Image",
                local_dir=assembled_dir,
                allow_patterns=MINIMAX_H3_IMAGE_CONFIG_PATTERNS,
                token=tok,
            )
            if DRIVE_MOUNTED and os.path.exists("/content/drive/MyDrive"):
                try:
                    shutil.copytree(
                        assembled_dir,
                        drive_cfg_cache,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.safetensors", ".cache*"),
                    )
                except Exception:
                    pass

    if not link_weights:
        return assembled_dir

    spec = MODEL_CATALOG.get(target_id, MODEL_CATALOG["minimax_h3_image"])
    req_rel_names = [fn for fn, _ in spec["files"]]
    resolved_paths: List[str] = []
    for rel_fn in req_rel_names:
        found = _find_existing_model_file(target_id, rel_fn)
        if not found:
            raise RuntimeError(f"未找到必需权重分片 {rel_fn}，请先点击「下载权重」。")
        resolved_paths.append(found)

    if stage_nvme_if_low_ram:
        resolved_paths = _stage_model_files_to_local_nvme(
            target_id, resolved_paths, req_rel_names, "MiniMax-H3-Image 模块化权重"
        )

    for rel_fn, src_path in zip(req_rel_names, resolved_paths):
        dst_link = os.path.join(assembled_dir, rel_fn)
        os.makedirs(os.path.dirname(dst_link), exist_ok=True)
        abs_src = os.path.abspath(src_path)
        if os.path.lexists(dst_link):
            try:
                if os.path.islink(dst_link) and os.path.abspath(os.path.realpath(dst_link)) == abs_src and os.path.exists(dst_link):
                    continue
                os.unlink(dst_link)
            except Exception:
                pass
        os.symlink(abs_src, dst_link)

    return assembled_dir

def load_minimax_h3_image_pipeline(
    target_id: str,
    spec: Dict[str, Any],
    total_vram_gb: float,
    free_before_gb: float,
    hf_token: str = "",
) -> Tuple[Any, str]:
    """
    专用加载方法：装载 MiniMax-H3-Image (33B DiT + 32B VLM 编码器) 6 步原生 4K 模块化管线。
    - 大显存 + 大内存机型 (如 G4 96GB / H100 80GB / A100 High-RAM，VRAM>=65GB 且可用 RAM>=132GB)：
      采用官方 ComponentsManager(enable_auto_cpu_offload) 在 DDR5 系统内存与显存间整块极速轮换；
    - 中等显存或标准内存机型 (如 A100 40GB / L4 24GB 或可用 RAM<132GB)：
      按组件顺序加载并挂载 Diffusers Group Offloading (leaf_level + 磁盘/内存分层卸载) + 4K VAE Tiling，严防 OOM。
    """
    if total_vram_gb < 22.0:
        raise RuntimeError(
            f"当前 GPU 总显存 ({total_vram_gb:.1f} GB) 不足以运行 MiniMax-H3-Image (33B)！"
            "请切换至 G4 (96GB，推荐)、H100 (80GB)、A100 或 L4 (24GB) 实例。"
        )

    MODEL_DOWNLOAD_STATE["progress"] = 0.22
    ComponentsManager, ModularPipeline = ensure_minimax_h3_image_dependencies()
    _trim_linux_cpu_heap()
    total_ram_gb, avail_ram_gb = _get_system_ram_gb()
    repo_id = spec.get("repo_id") or "MiniMaxAI/MiniMax-H3-Image"

    MODEL_DOWNLOAD_STATE["progress"] = 0.34
    MODEL_DOWNLOAD_STATE["message"] = "正在构建 MiniMax-H3-Image 本地零拷贝组装目录与共享编码器映射..."
    assembled_dir = prepare_minimax_h3_image_assembled_dir(
        target_id=target_id,
        repo_id=repo_id,
        hf_token=hf_token,
        link_weights=True,
        stage_nvme_if_low_ram=(avail_ram_gb < 132.0),
    )

    total_ram_gb, avail_ram_gb = _get_system_ram_gb()
    if total_vram_gb >= 65.0 and avail_ram_gb >= 132.0:
        MODEL_DOWNLOAD_STATE["progress"] = 0.56
        MODEL_DOWNLOAD_STATE["message"] = (
            f"正在通过 ComponentsManager 装载 MiniMax-H3-Image 至 {total_ram_gb:.0f}GB 内存热备池与 GPU..."
        )
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device="cuda")
        pipe = ModularPipeline.from_pretrained(assembled_dir, components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16)
        vae_mod = getattr(pipe, "vae", None)
        if vae_mod is not None and total_vram_gb < 88.0 and hasattr(vae_mod, "enable_tiling"):
            vae_mod.enable_tiling()
        mode_label = f"{spec['name']} · BF16 原生 · ComponentsManager 内存热备"
        print(
            f"⚡ [MiniMax-H3-Image] 已启用官方 ComponentsManager 自动显存/内存热备轮换 "
            f"(VRAM {total_vram_gb:.1f}GB / 可用 RAM {avail_ram_gb:.1f}GB)"
        )
        return pipe, mode_label

    # 回退分支：中等显存 (<65GB) 或标准主机内存 (<132GB) 分组件顺序装载 + Group Offloading
    from diffusers.hooks import apply_group_offloading

    use_disk_offload = avail_ram_gb < 132.0
    offload_root = os.path.join(os.getcwd(), "models", "_offload_cache", target_id)
    if use_disk_offload:
        os.makedirs(offload_root, exist_ok=True)

    pipe = ModularPipeline.from_pretrained(assembled_dir)
    MODEL_DOWNLOAD_STATE["progress"] = 0.42
    MODEL_DOWNLOAD_STATE["message"] = "正在加载 MiniMax-H3-Image 调度器与多模态 Processor..."
    pipe.load_components(names=["scheduler", "tokenizer", "processor"])

    MODEL_DOWNLOAD_STATE["progress"] = 0.54
    MODEL_DOWNLOAD_STATE["message"] = "正在分层装载 32B 视觉文本编码器 (1/3)..."
    pipe.load_components(names="text_encoder", dtype=torch.bfloat16)
    te_disk = os.path.join(offload_root, "text_encoder") if use_disk_offload else None
    if te_disk:
        os.makedirs(te_disk, exist_ok=True)
    apply_group_offloading(
        pipe.text_encoder,
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        use_stream=not use_disk_offload,
        low_cpu_mem_usage=True,
        offload_to_disk_path=te_disk,
    )
    _trim_linux_cpu_heap()

    MODEL_DOWNLOAD_STATE["progress"] = 0.74
    MODEL_DOWNLOAD_STATE["message"] = "正在分层装载 33B 图像 DiT 主干 (2/3)..."
    pipe.load_components(names="transformer", dtype=torch.bfloat16)
    dit_disk = os.path.join(offload_root, "transformer") if use_disk_offload else None
    if dit_disk:
        os.makedirs(dit_disk, exist_ok=True)
    apply_group_offloading(
        pipe.transformer,
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        use_stream=not use_disk_offload,
        low_cpu_mem_usage=True,
        offload_to_disk_path=dit_disk,
    )
    _trim_linux_cpu_heap()

    MODEL_DOWNLOAD_STATE["progress"] = 0.88
    MODEL_DOWNLOAD_STATE["message"] = "正在装载 4K 16MP 图像 VAE 并开启分块解码保护 (3/3)..."
    pipe.load_components(names="vae", dtype=torch.bfloat16)
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    apply_group_offloading(
        pipe.vae,
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        use_stream=False,
    )
    _trim_linux_cpu_heap()
    tag_suffix = "NVMe 分层卸载" if use_disk_offload else "分层流式卸载"
    return pipe, f"{spec['name']} · BF16 · {tag_suffix}"

def run_minimax_h3_image_inference(
    active_pipe: Any,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    ref_images: Optional[List[Any]] = None,
) -> Any:
    """
    专用推理方法：调用 MiniMaxH3ImageModularPipeline 执行 6 步 PDD 蒸馏原生 4K 文生图或多参考图编辑。
    注意：该管线固定 6 步输出头且已蒸馏 CFG，严禁传入 num_inference_steps / guidance_scale / negative_prompt，
    必须显式指定 output="images" 获取 PIL.Image 列表。
    """
    w = max(512, (int(width) // 32) * 32)
    h = max(512, (int(height) // 32) * 32)
    clean_prompt = (prompt or "").strip() or "Ultra-detailed 4K commercial photography, natural lighting, sharp focus."

    def _invoke_once(seed_val: int):
        gen = torch.Generator(device="cpu").manual_seed(int(seed_val))
        call_kw: Dict[str, Any] = {
            "prompt": clean_prompt,
            "height": h,
            "width": w,
            "generator": gen,
            "output": "images",
        }
        valid_refs = [im.convert("RGB") if hasattr(im, "convert") else im for im in (ref_images or []) if im is not None]
        if valid_refs:
            call_kw["image"] = valid_refs if len(valid_refs) > 1 else valid_refs[0]
        out = active_pipe(**call_kw)
        if isinstance(out, (list, tuple)):
            return out[0]
        if hasattr(out, "images") and out.images:
            return out.images[0]
        return out

    try:
        return _invoke_once(int(seed))
    except torch.cuda.OutOfMemoryError:
        print("[MiniMax-H3-Image] 检测到高分辨率解码显存峰值，自动开启 VAE Tiling 并清理瞬态缓存重试...")
        _reclaim_transient_gpu_memory()
        vae_mod = getattr(active_pipe, "vae", None)
        if vae_mod is not None and hasattr(vae_mod, "enable_tiling"):
            vae_mod.enable_tiling()
        return _invoke_once(int(seed))

def _sync_legacy_globals():
    """将多模型显存驻留表 LOADED_RUNTIMES 同步至兼容全局变量 PIPELINE / LLM_BUNDLE"""
    # (shared state on runtime_state)
    rs.LLM_BUNDLE = None
    if _is_llm_model_id(rs.SELECTED_MODEL_ID) and rs.SELECTED_MODEL_ID in LOADED_RUNTIMES:
        rs.LLM_BUNDLE = LOADED_RUNTIMES[rs.SELECTED_MODEL_ID].get("llm_bundle")
    if rs.LLM_BUNDLE is None:
        for rt in reversed(list(LOADED_RUNTIMES.values())):
            if rt.get("llm_bundle") is not None:
                rs.LLM_BUNDLE = rt["llm_bundle"]
                break

    sel_rt = LOADED_RUNTIMES.get(rs.SELECTED_MODEL_ID)
    if sel_rt and sel_rt.get("pipeline") is not None:
        rs.PIPELINE = sel_rt["pipeline"]
    else:
        rs.PIPELINE = None
        for rt in reversed(list(LOADED_RUNTIMES.values())):
            if rt.get("pipeline") is not None:
                rs.PIPELINE = rt["pipeline"]
                break

    if rs.SELECTED_MODEL_ID in LOADED_RUNTIMES:
        rs.LOADED_MODEL_ID = rs.SELECTED_MODEL_ID
    elif LOADED_RUNTIMES:
        rs.LOADED_MODEL_ID = next(reversed(LOADED_RUNTIMES))
    else:
        rs.LOADED_MODEL_ID = None

    if LOADED_RUNTIMES:
        rs.CURRENT_MODEL_MODE = " + ".join(rt["mode_label"] for rt in LOADED_RUNTIMES.values())
    else:
        rs.CURRENT_MODEL_MODE = None

def _trim_linux_cpu_heap():
    """在释放超大规模 CPU BF16 权重张量后主动触发 glibc malloc_trim(0)，将物理页立即归还给系统 MemAvailable"""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

def _release_pipeline_object(pipe_obj: Any):
    """主动解除 ModularPipeline / ComponentsManager / Accelerate Hook 的循环引用，确保 125GB+ CPU/GPU 权重立即释放"""
    if pipe_obj is None:
        return
    try:
        cm = getattr(pipe_obj, "components_manager", None)
        if cm is not None:
            for attr_dict_name in ("components", "_components", "_offload_hooks"):
                d = getattr(cm, attr_dict_name, None)
                if isinstance(d, dict):
                    d.clear()
        for sub_attr in ("transformer", "text_encoder", "vae", "dit", "video_vae", "audio_vae"):
            sub_mod = getattr(pipe_obj, sub_attr, None)
            if sub_mod is not None:
                try:
                    from diffusers.hooks import remove_group_offloading
                    remove_group_offloading(sub_mod)
                except Exception:
                    pass
                try:
                    setattr(pipe_obj, sub_attr, None)
                except Exception:
                    pass
    except Exception:
        pass

def _unload_model_locked(model_id: Optional[str] = None) -> List[str]:
    """在已持有 GPU_LOCK 的前提下卸载指定模型（或全部模型），返回被卸载的模型名称列表"""
    # (shared state on runtime_state)
    norm_id = _normalize_builtin_model_alias(model_id) if (model_id and model_id != "all") else model_id
    removed_names: List[str] = []
    if not norm_id or norm_id == "all":
        for mid in list(LOADED_RUNTIMES.keys()):
            entry = LOADED_RUNTIMES.pop(mid, None)
            if entry:
                _stop_gguf_server_bundle(entry.get("llm_bundle"))
                _release_pipeline_object(entry.get("pipeline"))
                removed_names.append(entry.get("name", mid))
                entry.clear()
        _stop_gguf_server_bundle(rs.LLM_BUNDLE)
        _release_pipeline_object(rs.PIPELINE)
        rs.PIPELINE = None
        rs.LLM_BUNDLE = None
        rs.LOADED_MODEL_ID = None
        rs.CURRENT_MODEL_MODE = None
    elif norm_id in LOADED_RUNTIMES:
        entry = LOADED_RUNTIMES.pop(norm_id, None)
        if entry:
            _stop_gguf_server_bundle(entry.get("llm_bundle"))
            if rs.PIPELINE is entry.get("pipeline"):
                rs.PIPELINE = None
            if rs.LLM_BUNDLE is entry.get("llm_bundle"):
                rs.LLM_BUNDLE = None
            _release_pipeline_object(entry.get("pipeline"))
            removed_names.append(entry.get("name", norm_id))
            entry.clear()
        _sync_legacy_globals()

    _trim_linux_cpu_heap()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return removed_names

def _unload_all_gpu_models():
    """彻底释放全部已加载模型的 GPU 显存"""
    _unload_model_locked("all")

def unload_model_from_vram_core(model_id: Optional[str] = None) -> Tuple[bool, str]:
    """供客户端按需独立卸载单个或全部显存驻留模型（立即中断正在进行的推理任务以避免死锁）"""
    touch_activity()
    ABORT_GENERATION_EVENT.set()
    for jid, j in list(JOBS.items()):
        if j.get("status") in ("queued", "running") and (not model_id or model_id == "all" or j.get("model_id") == model_id):
            ev = _JOB_CANCEL_EVENTS.get(jid)
            if ev is not None:
                ev.set()
            j["status"] = "failed"
            j["progress"] = 0.0
            j["message"] = "已终止推理并从显存卸载该模型"
    with CONSOLE_LOCK:
        ACTIVE_TQDM_BARS.clear()
    with GPU_LOCK:
        try:
            removed = _unload_model_locked(model_id)
            free_gb = (torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)) if torch.cuda.is_available() else 0.0
            if not removed:
                msg = "当前指定模型未驻留显存，无需卸载。"
            else:
                remain_cnt = len(LOADED_RUNTIMES)
                remain_str = f"（显存中仍保留 {remain_cnt} 个常驻模型）" if remain_cnt > 0 else "（显存已完全清空）"
                msg = f"已从显存卸载 {', '.join(removed)}，当前可用显存 {free_gb:.1f} GB {remain_str}"
            MODEL_DOWNLOAD_STATE["message"] = msg
            print(f"[显存管理] {msg}")
            return True, msg
        finally:
            ABORT_GENERATION_EVENT.clear()

def _ensure_vram_for_new_model_locked(target_id: str, required_free_gb: float) -> List[str]:
    """
    多模型共驻智能显存与系统内存守卫 (LRU Eviction)：
    - 若目标模型自身已在显存中，先卸载旧实例以便重载；
    - 若已有 ComponentsManager / 内存热备型 33B 巨型管线驻留，按其前向峰值 (~64GB VRAM) 与系统内存余量核算可用空间，
      使 Qwen3.8-GGUF (~21GB) 在 G4 96GB 上可与其安全共驻，而切换到另一巨型模型时自动先腾退旧实例释放 125GB+ 内存！
    - 仅当剩余可用空间不足以容纳新模型时，按最早加载顺序 (LRU) 自动腾退旧模型，防止 CUDA / Cgroup OOM。
    """
    evicted: List[str] = []
    if target_id in LOADED_RUNTIMES:
        _unload_model_locked(target_id)

    if not torch.cuda.is_available():
        return evicted

    def _effective_free_vram_gb() -> float:
        free_b, total_b = torch.cuda.mem_get_info("cuda")
        raw_free = free_b / (1024 ** 3)
        total_v = total_b / (1024 ** 3)
        used_v = max(0.0, total_v - raw_free)
        # ComponentsManager 在空闲期将 62GB 主干暂存 CPU 内存（当前显存占用接近 0），推理期需瞬时占用 ~64GB 显存
        has_cpu_offloaded_giant = any(
            mid in ("minimax_h3", "minimax_h3_ref2va", "minimax_h3_image")
            or "ComponentsManager" in str(rt.get("mode_label", ""))
            or "内存热备" in str(rt.get("mode_label", ""))
            for mid, rt in LOADED_RUNTIMES.items()
        )
        if has_cpu_offloaded_giant and used_v < 58.0:
            reserved_peak = 64.0
            return min(raw_free, max(0.0, total_v - max(used_v, reserved_peak)))
        return raw_free

    target_spec = MODEL_CATALOG.get(target_id, {})
    target_arch = target_spec.get("pipeline_arch") or target_id
    needs_heavy_cpu_ram = target_arch in ("minimax_h3", "minimax_h3_ref2va", "minimax_h3_image")

    while LOADED_RUNTIMES:
        eff_free = _effective_free_vram_gb()
        _, avail_ram = _get_system_ram_gb()
        ram_short = needs_heavy_cpu_ram and avail_ram < 132.0 and any(
            "内存热备" in str(rt.get("mode_label", "")) or "ComponentsManager" in str(rt.get("mode_label", ""))
            for rt in LOADED_RUNTIMES.values()
        )
        if eff_free >= required_free_gb and not ram_short:
            break
        oldest_id = next(iter(LOADED_RUNTIMES))
        names = _unload_model_locked(oldest_id)
        evicted.extend(names)
    return evicted

def load_model_to_vram_core(
    model_id: Optional[str] = None,
    precision_mode: str = "bf16",
    offload_mode: str = "disk",
    hf_token: str = "",
    context_length: Optional[int] = None,
) -> Tuple[bool, str]:
    # (shared state on runtime_state)
    touch_activity()

    if context_length is not None:
        rs.QWEN_CONTEXT_LENGTH = _clamp_context_length(context_length)
    ctx_len = rs.QWEN_CONTEXT_LENGTH

    if MODEL_DOWNLOAD_STATE["is_loading_vram"]:
        return False, "模型正在加载进显存中，请稍候..."

    if not torch.cuda.is_available():
        return False, "未检测到 CUDA GPU！请在 Colab 中切换为 G4 / A100 / L4 GPU。"

    norm_id = _normalize_builtin_model_alias(model_id)
    target_id = norm_id if (norm_id and norm_id in MODEL_CATALOG) else rs.SELECTED_MODEL_ID
    rs.SELECTED_MODEL_ID = target_id
    spec = MODEL_CATALOG[target_id]
    pipeline_arch = spec.get("pipeline_arch") or target_id
    file_status = check_model_files_status(target_id)
    auto_switched_notice = ""

    if target_id == "qwen38_27b_gguf":
        if not (
            _find_existing_model_file("qwen38_27b_gguf", "Qwen3.8-27B-UD-Q4_K_M.gguf")
            and _find_existing_model_file("qwen38_27b_gguf", "mmproj-F16.gguf")
        ):
            return False, f"{spec['name']} 权重尚未下载完整 ({file_status['downloaded_count']}/{file_status['total_count']})！请先点击「下载权重」。"
    elif pipeline_arch == "gguf":
        main_ggufs = [
            fn for fn, _ in spec.get("files", [])
            if fn.lower().endswith(".gguf") and "mmproj" not in fn.lower() and "/mtp" not in fn.lower() and not os.path.basename(fn.lower()).startswith("mtp")
        ]
        if not main_ggufs or not all(_find_existing_model_file(target_id, fn) for fn in main_ggufs):
            return False, f"{spec['name']} GGUF 主权重尚未下载完整 ({file_status['downloaded_count']}/{file_status['total_count']})！请先点击「下载权重」。"
    else:
        if not file_status["all_downloaded"]:
            return False, f"{spec['name']} 权重尚未下载完整 ({file_status['downloaded_count']}/{file_status['total_count']})！请先点击「下载权重」。"

    t_start = time.time()
    MODEL_DOWNLOAD_STATE["is_loading_vram"] = True
    MODEL_DOWNLOAD_STATE["progress"] = 0.12
    MODEL_DOWNLOAD_STATE["message"] = f"正在评估显存余量并装载 {spec['name']} 至 GPU..."
    with CONSOLE_LOCK:
        ACTIVE_TQDM_BARS.clear()

    with GPU_LOCK:
        try:
            total_vram_gb = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
            vram_limit = max(8.0, total_vram_gb - 2.5)
            approx_gb = float(spec.get("approx_size_gb") or 20.0)

            # 计算新模型所需最小空闲显存阈值：余量充足时自动与已加载模型共驻显存！
            if target_id == "qwen38_27b":
                required_free_gb = 53.0
            elif pipeline_arch in ("qwen38_27b_gguf", "gguf"):
                q4_kv_gb = (ctx_len / 262144.0) * 4.6
                required_free_gb = min(max(12.0, approx_gb + 2.0 + q4_kv_gb), max(12.0, total_vram_gb - 1.2))
            elif pipeline_arch in ("minimax_h3", "minimax_h3_ref2va", "minimax_h3_image"):
                # BF16 单阶段峰值约 62.5GB；在 G4 (96GB) 上可与 Qwen3.8-GGUF (~21GB) 同时共驻显存
                required_free_gb = min(65.0, max(24.0, total_vram_gb - 5.0))
            elif pipeline_arch == "transformers":
                required_free_gb = min(max(12.0, approx_gb + 3.0), max(12.0, total_vram_gb - 1.5))
            else:
                required_free_gb = 24.0

            if torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3) < required_free_gb and COMFYUI_STATE.get("is_running"):
                from .comfyui_service import free_comfyui_vram_core
                rs.resolve_callable("free_comfyui_vram_core", free_comfyui_vram_core)()
            evicted_names = _ensure_vram_for_new_model_locked(target_id, required_free_gb)
            free_before_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)

            vram_config = {
                "offload_dtype": "disk",
                "offload_device": "disk",
                "onload_dtype": "disk",
                "onload_device": "disk",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cuda",
            }

            new_pipe = None
            new_llm = None
            mode_label = spec["name"]
            repo_id = spec.get("repo_id", "")

            if pipeline_arch in ("minimax_h3", "minimax_h3_ref2va"):
                from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

                _trim_linux_cpu_heap()
                total_ram_gb, avail_ram_gb = _get_system_ram_gb()
                cpu_fast_vram_config = {
                    "offload_dtype": torch.bfloat16,
                    "offload_device": "cpu",
                    "onload_dtype": torch.bfloat16,
                    "onload_device": "cpu",
                    "preparing_dtype": torch.bfloat16,
                    "preparing_device": "cuda",
                    "computation_dtype": torch.bfloat16,
                    "computation_device": "cuda",
                }

                part_prefix = "Ref2VA" if pipeline_arch == "minimax_h3_ref2va" else "FL2VA"
                te_rel_names = [f"{part_prefix}/text_encoder/model-{i:05d}-of-00014.safetensors" for i in range(1, 15)]
                dit_rel_names = [f"{part_prefix}/transformer/model-{i:05d}-of-00013.safetensors" for i in range(1, 14)]

                MODEL_DOWNLOAD_STATE["progress"] = 0.22
                MODEL_DOWNLOAD_STATE["message"] = f"正在映射 {spec['name']} 多模态编码器权重 (14 分片)..."
                te_paths = [_resolve_and_stage_model_file(target_id, fn) for fn in te_rel_names]

                MODEL_DOWNLOAD_STATE["progress"] = 0.42
                MODEL_DOWNLOAD_STATE["message"] = f"正在映射 {spec['name']} 33B DiT 主干权重 (13 分片)..."
                dit_paths = [_resolve_and_stage_model_file(target_id, fn) for fn in dit_rel_names]

                MODEL_DOWNLOAD_STATE["progress"] = 0.58
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 3D 视频 VAE 与 32kHz 立体声 VAE 权重..."
                vvae_path = _resolve_and_stage_model_file(
                    target_id, f"{part_prefix}/video_vae/source/model.safetensors"
                )
                avae_path = _resolve_and_stage_model_file(
                    target_id, f"{part_prefix}/audio_vae/model.safetensors"
                )

                # 基于分片映射完成后的真实可用系统内存自适应选择热备策略（严守 >=24GB CPU 内存安全余量，杜绝 /dev/shm + 匿名内存叠加触发 Cgroup OOM）：
                # - Tier 1 (如 G4 176GB RAM, 可用 >=148GB)：text_encoder(62.1GB) + dit(61.7GB) 双主干常驻系统内存，PCIe 极速互换 (~4-6s)！
                # - Tier 2 (可用 88~148GB RAM)：text_encoder(62.1GB) 常驻系统内存 (~4-6s 极速解析提示词)，dit 顺序缓存至本地高速 NVMe！
                # - Tier 3 (低内存机型 <88GB RAM)：保持磁盘按需映射，若本地 NVMe 空间充裕则脱离慢速云盘 FUSE。
                total_ram_gb, avail_ram_gb = _get_system_ram_gb()
                ram_residency_tag = ""
                if avail_ram_gb >= 148.0:
                    te_cfg = cpu_fast_vram_config
                    dit_cfg = cpu_fast_vram_config
                    vae_cfg = cpu_fast_vram_config
                    ram_residency_tag = "双主干内存热备"
                    print(
                        f"⚡ [G4 大内存热备] 检测到充裕系统内存 ({avail_ram_gb:.1f}/{total_ram_gb:.1f} GB)，"
                        f"已启用 {spec['name']} 32B 文本编码器 + 33B DiT 双主干 DDR5 内存常驻热备（阶段切换仅需 ~5 秒）！"
                    )
                    MODEL_DOWNLOAD_STATE["progress"] = 0.68
                    MODEL_DOWNLOAD_STATE["message"] = (
                        f"正在将 {spec['name']} 预装载至 {total_ram_gb:.0f}GB 高速系统内存热备池（首次载入后换提示词仅需 ~5 秒）..."
                    )
                elif avail_ram_gb >= 88.0:
                    te_cfg = cpu_fast_vram_config
                    dit_paths = _stage_model_files_to_local_nvme(
                        target_id, dit_paths, dit_rel_names, f"{spec['name']} 33B DiT 主干"
                    )
                    dit_cfg = vram_config
                    vae_cfg = cpu_fast_vram_config
                    ram_residency_tag = "编码器内存热备"
                    print(
                        f"⚡ [混合内存热备] 系统可用内存 {avail_ram_gb:.1f}/{total_ram_gb:.1f} GB，"
                        f"已将 {spec['name']} 32B 文本编码器锁入系统内存热备（换提示词 ~5 秒极速换入），DiT 走本地高速存储！"
                    )
                    MODEL_DOWNLOAD_STATE["progress"] = 0.68
                    MODEL_DOWNLOAD_STATE["message"] = (
                        "正在将 32B 文本编码器装载至系统内存热备池（后续解析新提示词仅需 ~5 秒）..."
                    )
                else:
                    te_paths = _stage_model_files_to_local_nvme(
                        target_id, te_paths, te_rel_names, f"{spec['name']} 多模态编码器"
                    )
                    dit_paths = _stage_model_files_to_local_nvme(
                        target_id, dit_paths, dit_rel_names, f"{spec['name']} 33B DiT 主干"
                    )
                    te_cfg = vram_config
                    dit_cfg = vram_config
                    vae_cfg = vram_config
                    MODEL_DOWNLOAD_STATE["progress"] = 0.72
                    MODEL_DOWNLOAD_STATE["message"] = f"正在构建 {spec['name']} BF16 原生 GPU 计算图与 Processor..."

                new_pipe = MiniMaxH3Pipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(path=te_paths, **te_cfg),
                        ModelConfig(path=dit_paths, **dit_cfg),
                        ModelConfig(path=vvae_path, **vae_cfg),
                        ModelConfig(path=avae_path, **vae_cfg),
                    ],
                    processor_config=ModelConfig(
                        model_id=repo_id or "MiniMaxAI/MiniMax-H3",
                        origin_file_pattern=f"{part_prefix}/processor/",
                        download_source="huggingface",
                    ),
                    vram_limit=vram_limit,
                )
                mode_label = f"{spec['name']} · BF16 原生"
                residency_tag = _enable_high_vram_optimizations(
                    new_pipe,
                    total_vram_gb=total_vram_gb,
                    preload_vram=True,
                    skip_preload_attrs=("dit",) if free_before_gb >= 76.0 else ("text_encoder", "dit"),
                )
                tags = [t for t in (ram_residency_tag, residency_tag) if t]
                if tags:
                    mode_label = f"{mode_label} · {' / '.join(tags)}"

            elif pipeline_arch == "qwen_image_21":
                from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
                MODEL_DOWNLOAD_STATE["progress"] = 0.45
                MODEL_DOWNLOAD_STATE["message"] = f"正在映射 {spec['name']} 权重并构建 GPU 管线..."
                dit_paths = [
                    _resolve_and_stage_model_file(target_id, f"transformer/diffusion_pytorch_model-{i:05d}-of-00002.safetensors")
                    for i in range(1, 3)
                ]
                te_paths = [
                    _resolve_and_stage_model_file(target_id, f"text_encoder/model-{i:05d}-of-00004.safetensors")
                    for i in range(1, 5)
                ]
                vae_path = _resolve_and_stage_model_file(target_id, "vae/diffusion_pytorch_model.safetensors")
                new_pipe = QwenImage21Pipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(path=dit_paths, **vram_config),
                        ModelConfig(path=te_paths, **vram_config),
                        ModelConfig(path=vae_path, **vram_config),
                    ],
                    processor_config=ModelConfig(
                        model_id=repo_id or "Qwen/Qwen-Image-2.1",
                        origin_file_pattern="processor/",
                        download_source="huggingface",
                    ),
                    vram_limit=vram_limit,
                )
                _enable_high_vram_optimizations(new_pipe, total_vram_gb=total_vram_gb, preload_vram=True)
                mode_label = f"{spec['name']} · BF16"

            elif pipeline_arch == "z_image_turbo":
                from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
                MODEL_DOWNLOAD_STATE["progress"] = 0.45
                MODEL_DOWNLOAD_STATE["message"] = f"正在映射 {spec['name']} 权重并构建 8 步极速生图管线..."
                dit_paths = [
                    _resolve_and_stage_model_file(target_id, f"transformer/diffusion_pytorch_model-{i:05d}-of-00003.safetensors")
                    for i in range(1, 4)
                ]
                te_paths = [
                    _resolve_and_stage_model_file(target_id, f"text_encoder/model-{i:05d}-of-00003.safetensors")
                    for i in range(1, 4)
                ]
                vae_path = _resolve_and_stage_model_file(target_id, "vae/diffusion_pytorch_model.safetensors")
                new_pipe = ZImagePipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(path=dit_paths, **vram_config),
                        ModelConfig(path=te_paths, **vram_config),
                        ModelConfig(path=vae_path, **vram_config),
                    ],
                    tokenizer_config=ModelConfig(
                        model_id=repo_id or "Tongyi-MAI/Z-Image-Turbo",
                        origin_file_pattern="tokenizer/",
                        download_source="huggingface",
                    ),
                    vram_limit=vram_limit,
                )
                _enable_high_vram_optimizations(new_pipe, total_vram_gb=total_vram_gb, preload_vram=True)
                mode_label = f"{spec['name']} · 8步极速 BF16"

            elif pipeline_arch == "minimax_h3_image":
                new_pipe, mode_label = load_minimax_h3_image_pipeline(
                    target_id=target_id,
                    spec=spec,
                    total_vram_gb=total_vram_gb,
                    free_before_gb=free_before_gb,
                    hf_token=hf_token,
                )

            elif pipeline_arch == "diffusers_image":
                MODEL_DOWNLOAD_STATE["progress"] = 0.45
                MODEL_DOWNLOAD_STATE["message"] = f"正在使用 Diffusers 装载自定义图像模型 {spec['name']}..."
                first_rel = spec["files"][0][0] if spec.get("files") else ""
                first_existing = _find_existing_model_file(target_id, first_rel) if first_rel else None
                model_dir = os.path.dirname(first_existing) if first_existing else _get_model_dirs(target_id)[2]
                if "/" in first_rel and first_existing:
                    # 若分片在子目录 (如 transformer/xxx.safetensors)，取顶层模型目录
                    sub_depth = len(first_rel.split("/")) - 1
                    for _ in range(sub_depth):
                        model_dir = os.path.dirname(model_dir)
                from diffusers import AutoPipelineForText2Image, DiffusionPipeline
                try:
                    new_pipe = AutoPipelineForText2Image.from_pretrained(
                        model_dir if os.path.exists(os.path.join(model_dir, "model_index.json")) else repo_id,
                        torch_dtype=torch.bfloat16,
                    ).to("cuda")
                except Exception:
                    new_pipe = DiffusionPipeline.from_pretrained(
                        model_dir if os.path.exists(os.path.join(model_dir, "model_index.json")) else repo_id,
                        torch_dtype=torch.bfloat16,
                    ).to("cuda")
                mode_label = f"{spec['name']} · Diffusers BF16"

            elif pipeline_arch in ("qwen38_27b", "transformers"):
                if target_id == "qwen38_27b" and free_before_gb < 49.0:
                    raise RuntimeError(
                        f"当前 GPU 可用显存 ({free_before_gb:.1f} GB / 总计 {total_vram_gb:.1f} GB) 不足以加载 "
                        f"Qwen3.8-27B BF16 原生权重（需约 52 GB 显存，推荐 A100 80GB）。"
                        f"若在 L4 (24GB) 或中低显存运行，请在模型中心选择独立的「Qwen3.8-27B-GGUF (UD-Q4_K_M)」量化版！"
                    )

                MODEL_DOWNLOAD_STATE["progress"] = 0.30
                MODEL_DOWNLOAD_STATE["message"] = (
                    f"正在装载 {spec['name']} Transformers 原生权重 (上下文上限 {ctx_len // 1024}K)..."
                )
                first_rel = spec["files"][0][0] if spec.get("files") else "model-00001-of-00018.safetensors"
                first_shard = _find_existing_model_file(target_id, first_rel)
                model_dir = os.path.dirname(first_shard) if first_shard else _get_model_dirs(target_id)[2]

                # 确保配置文件在同一目录中完备
                if not os.path.exists(os.path.join(model_dir, "config.json")):
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=model_dir,
                        ignore_patterns=["*.safetensors", "*.bin", "*.pth", "*.gguf", "*.pt"],
                    )

                if not os.path.exists(os.path.join(model_dir, "config.json")):
                    raise RuntimeError(
                        f"仓库 {repo_id} 未包含标准 Transformers 配置文件 (config.json / tokenizer.json)，"
                        f"仅包含独立权重分片（如 3D 重建 / 专用 SDK / ComfyUI 单体组件），无法作为对话或视觉大模型直接加载。"
                        f"请选用 GGUF 量化模型、含 config.json 的标准 Transformers 模型，或在 ComfyUI 工作台中使用单体权重。"
                    )

                def _load_transformers_llm():
                    _ensure_torchaudio_compatible()
                    _ensure_qwen_fast_kernels()
                    from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM

                    def _load_tok():
                        try:
                            return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                        except Exception:
                            return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)

                    try:
                        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
                        if not hasattr(processor, "tokenizer"):
                            tok = _load_tok()
                            processor = _CausalLMProcessorAdapter(tok)
                    except Exception:
                        tok = _load_tok()
                        processor = _CausalLMProcessorAdapter(tok)

                    try:
                        mdl = AutoModelForImageTextToText.from_pretrained(
                            model_dir,
                            dtype=torch.bfloat16,
                            attn_implementation="sdpa",
                            device_map="cuda",
                            trust_remote_code=True,
                        )
                    except Exception:
                        mdl = AutoModelForCausalLM.from_pretrained(
                            model_dir,
                            torch_dtype=torch.bfloat16,
                            attn_implementation="sdpa",
                            device_map="auto",
                            trust_remote_code=True,
                        )
                    mode_tag = f"{spec['name']} · BF16 原生 · {ctx_len // 1024}K"
                    mdl.eval()
                    try:
                        _optimize_qwen35_for_fast_inference(mdl, max_context_len=ctx_len)
                    except Exception:
                        pass
                    return mdl, processor, mode_tag

                try:
                    mdl, proc, mode_tag = _load_transformers_llm()
                except (KeyError, ValueError, ImportError) as up_e:
                    print(f"检测到分词器或 transformers 组件缺失 ({up_e})，正在自动升级 transformers 及分词器依赖...")
                    subprocess.check_call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "-q",
                            "-U",
                            "transformers",
                            "accelerate",
                            "qwen-vl-utils",
                            "sentencepiece",
                            "tiktoken",
                            "protobuf",
                            "blobfile",
                        ]
                    )
                    import importlib
                    import transformers
                    importlib.reload(transformers)
                    mdl, proc, mode_tag = _load_transformers_llm()

                new_llm = {
                    "backend": "transformers",
                    "model": mdl,
                    "processor": proc,
                    "model_id": target_id,
                    "model_name": spec["name"],
                    "context_length": ctx_len,
                }
                mode_label = mode_tag

            elif pipeline_arch in ("qwen38_27b_gguf", "gguf"):
                MODEL_DOWNLOAD_STATE["progress"] = 0.25
                MODEL_DOWNLOAD_STATE["message"] = f"正在预热 {spec['name']} GGUF 权重并启动 llama.cpp CUDA 引擎..."
                if target_id == "qwen38_27b_gguf":
                    gguf_path = _resolve_and_stage_model_file("qwen38_27b_gguf", "Qwen3.8-27B-UD-Q4_K_M.gguf")
                    mmproj_path = _resolve_and_stage_model_file("qwen38_27b_gguf", "mmproj-F16.gguf")

                    mtp_rel = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
                    mtp_path: Optional[str] = None
                    try:
                        if not _find_existing_model_file("qwen38_27b_gguf", mtp_rel):
                            MODEL_DOWNLOAD_STATE["progress"] = 0.38
                            MODEL_DOWNLOAD_STATE["message"] = "正在自动补齐 Unsloth MTP 投机解码加速头 (1.31 GB)..."
                            print("[GGUF 引擎] 检测到尚未缓存 MTP 投机解码权重，正在 8 路并发极速补齐...")
                            ram_d, nvme_d, drive_d = _get_model_dirs("qwen38_27b_gguf")
                            dl_dir = ram_d if ram_d else (drive_d if DRIVE_MOUNTED else nvme_d)
                            os.makedirs(dl_dir, exist_ok=True)
                            dl_mtp = rs.resolve_callable("fast_parallel_hf_download", fast_parallel_hf_download)(
                                repo_id="unsloth/Qwen3.8-27B-GGUF",
                                filename=mtp_rel,
                                dest_dir=dl_dir,
                                hf_token=os.environ.get("HF_TOKEN", ""),
                                num_connections=8,
                            )
                            if DRIVE_MOUNTED and dl_dir != drive_d:
                                threading.Thread(
                                    target=_async_backup_to_drive,
                                    args=(dl_mtp, drive_d, mtp_rel),
                                    daemon=True,
                                ).start()
                        mtp_path = _resolve_and_stage_model_file("qwen38_27b_gguf", mtp_rel)
                    except Exception as mtp_e:
                        print(f"[GGUF 引擎] MTP 投机权重获取跳过，将以标准模式启动: {mtp_e}")
                        mtp_path = None
                else:
                    all_gguf_rels = [fn for fn, _ in spec.get("files", []) if fn.lower().endswith(".gguf")]
                    mmproj_rels = [fn for fn in all_gguf_rels if "mmproj" in fn.lower()]
                    mtp_rels = [fn for fn in all_gguf_rels if ("/mtp" in fn.lower() or os.path.basename(fn.lower()).startswith("mtp"))]
                    main_rels = sorted([fn for fn in all_gguf_rels if fn not in mmproj_rels and fn not in mtp_rels])
                    if not main_rels:
                        raise RuntimeError(f"未在 {spec['name']} 中找到主 GGUF 权重文件")
                    staged_mains = [_resolve_and_stage_model_file(target_id, fn) for fn in main_rels]
                    gguf_path = staged_mains[0]
                    mmproj_path = _resolve_and_stage_model_file(target_id, mmproj_rels[0]) if mmproj_rels else None
                    mtp_path = _resolve_and_stage_model_file(target_id, mtp_rels[0]) if mtp_rels else None

                new_llm = _start_qwen_gguf_server(
                    gguf_path=gguf_path,
                    mmproj_path=mmproj_path,
                    ctx_len=ctx_len,
                    cur_free_gb=free_before_gb,
                    mtp_path=mtp_path,
                    model_id=target_id,
                    model_name=spec["name"],
                    total_weights_gb=approx_gb,
                )
                mode_label = new_llm["mode_label"]

            free_after_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
            vram_delta_gb = round(max(0.5, free_before_gb - free_after_gb), 1)

            LOADED_RUNTIMES[target_id] = {
                "model_id": target_id,
                "name": spec["name"],
                "modality": spec["modality"],
                "mode_label": mode_label,
                "pipeline": new_pipe,
                "llm_bundle": new_llm,
                "vram_gb": vram_delta_gb,
                "loaded_at": time.time(),
            }
            _sync_legacy_globals()

            elapsed_s = round(time.time() - t_start, 1)
            MODEL_DOWNLOAD_STATE["progress"] = 1.0
            co_count = len(LOADED_RUNTIMES)
            evict_tip = f"（因显存上限已自动腾退 {', '.join(evicted_names)}）" if evicted_names else ""
            co_tip = f" | 当前共 {co_count} 个模型同时常驻显存，切换 0 等待" if co_count > 1 else ""
            msg = f"{spec['name']} 已装载至显存 ({mode_label}，约 {vram_delta_gb}GB，耗时 {elapsed_s}s){co_tip}！{evict_tip}{auto_switched_notice}"
            MODEL_DOWNLOAD_STATE["message"] = msg
            print(msg)
            return True, msg

        except Exception as e:
            _sync_legacy_globals()
            err = f"加载 {spec['name']} 至显存失败: {str(e)}"
            MODEL_DOWNLOAD_STATE["message"] = err
            print(err)
            return False, err
        finally:
            MODEL_DOWNLOAD_STATE["is_loading_vram"] = False
            touch_activity()

def load_minimax_h3_to_vram_core(
    precision_mode: str = "bf16",
    offload_mode: str = "fast_mmap",
    hf_token: str = "",
) -> Tuple[bool, str]:
    return rs.resolve_callable("load_model_to_vram_core", load_model_to_vram_core)("minimax_h3", "bf16", offload_mode, hf_token)

def _enable_high_vram_optimizations(
    pipe,
    total_vram_gb: float,
    preload_vram: bool = True,
    skip_preload_attrs: Tuple[str, ...] = (),
) -> str:
    """
    针对 G4 (96GB) / A100 (80GB) 等大显存实例 (>45GB VRAM) 的深度加速（完美兼容多模型显存共驻与 BF16 超大模型）：
    1. 拦截 DiffSynth 默认在每个阶段切换时强制触发的 module.offload()，基于实时剩余显存智能判断：
       - 对于 MiniMax-H3 BF16 (text_encoder 62.1GB + dit 61.7GB > 96GB)，在 text_encoder 与 dit 互斥切换时精准腾退，
         而在 dit -> video_vae (9.7GB) -> audio_vae (0.56GB) 阶段只要余量 >= 25GB 即全量常驻，连抽换种子 0 重载！
    2. 为 Qwen3-VL 32B PromptEmbedder 挂载零触发 LRU 缓存：命中缓存时连 load_models_to_device(["text_encoder"]) 亦直接跳过，
       保证 61.7GB DiT 主干在 G4 96GB 显存中全程锁死不卸载！
    """
    import types

    is_minimax_bf16 = pipe.__class__.__name__ == "MiniMaxH3Pipeline"

    # 1. 挂载 PromptEmbedder 编码缓存（连抽多条或同 Prompt 换 Seed 时 0ms 瞬时跳过 32B Qwen3-VL 推理及显存换入）
    try:
        for unit in getattr(pipe, "units", []):
            if unit.__class__.__name__ == "MiniMaxH3Unit_PromptEmbedder" and not hasattr(
                unit, "_orig_process"
            ):
                unit._orig_process = unit.process
                unit._orig_onload_names = getattr(unit, "onload_model_names", ("text_encoder",))
                # 置空外层自动 onload，仅在缓存未命中时按需加载 text_encoder，避免同提示词连抽时把常驻 DiT 挤出显存
                unit.onload_model_names = ()
                unit._embed_cache = {}

                def _cached_prompt_process(
                    self_unit,
                    pipe_obj,
                    prompt,
                    keyframes=None,
                    ref_blocks=None,
                    height=None,
                    width=None,
                    text_embedding=None,
                ):
                    if ref_blocks is None and text_embedding is None and prompt is not None:
                        kf_sig = None
                        if keyframes:
                            try:
                                kf_sig = tuple(
                                    (img.size, img.mode, img.tobytes()[:512]) for img in keyframes
                                )
                            except Exception:
                                kf_sig = id(keyframes)
                        cache_key = (prompt, height, width, kf_sig)
                        cached = self_unit._embed_cache.get(cache_key)
                        if cached is not None:
                            print("⚡ [PromptCache] 命中 Qwen3-VL 32B 多模态编码缓存，0ms 瞬时跳过文本编码阶段！")
                            return {
                                "prompt_embeds": cached["prompt_embeds"].to(
                                    pipe_obj.device, pipe_obj.torch_dtype
                                ),
                                "text_token_tags": cached["text_token_tags"].to(
                                    pipe_obj.device, torch.long
                                ),
                            }
                        t_pe = time.time()
                        pipe_obj.load_models_to_device(self_unit._orig_onload_names)
                        res = self_unit._orig_process(
                            pipe_obj,
                            prompt,
                            keyframes=keyframes,
                            ref_blocks=ref_blocks,
                            height=height,
                            width=width,
                            text_embedding=text_embedding,
                        )
                        self_unit._embed_cache.clear()
                        self_unit._embed_cache[cache_key] = {
                            "prompt_embeds": res["prompt_embeds"].detach().clone(),
                            "text_token_tags": res["text_token_tags"].detach().clone(),
                        }
                        print(
                            f"⚡ [PromptEmbedder] 多模态提示词解析完成，总耗时 {time.time() - t_pe:.1f}s"
                            "（已写入缓存，同提示词换种子后续 0ms）"
                        )
                        return res
                    t_pe = time.time()
                    pipe_obj.load_models_to_device(self_unit._orig_onload_names)
                    res = self_unit._orig_process(
                        pipe_obj,
                        prompt,
                        keyframes=keyframes,
                        ref_blocks=ref_blocks,
                        height=height,
                        width=width,
                        text_embedding=text_embedding,
                    )
                    print(f"⚡ [PromptEmbedder] 多模态提示词解析完成，总耗时 {time.time() - t_pe:.1f}s")
                    return res

                unit.process = types.MethodType(_cached_prompt_process, unit)
    except Exception as e:
        print(f"ℹ️ Prompt 缓存挂载跳过: {e}")

    if total_vram_gb < 45.0:
        return ""

    def _is_submodel_fully_prepared_on_cuda(mdl) -> bool:
        if mdl is None:
            return False
        found_stateful = False
        for mod in mdl.modules():
            if hasattr(mod, "state"):
                found_stateful = True
                if getattr(mod, "state", 0) != 2:
                    return False
        if found_stateful:
            return True
        p = next(mdl.parameters(), None)
        return bool(p is not None and p.device.type == "cuda")

    def _offload_submodel_to_host(mdl) -> bool:
        if mdl is None or not getattr(mdl, "vram_management_enabled", False):
            return False
        has_active = any(
            hasattr(mod, "state") and getattr(mod, "state", 0) != 0
            for mod in mdl.modules()
        )
        if not has_active:
            return False
        if hasattr(mdl, "offload"):
            mdl.offload()
        else:
            for mod in mdl.modules():
                if hasattr(mod, "offload"):
                    mod.offload()
        return True

    # 2. 替换 load_models_to_device：智能阶段显存驻留 + 整块 PCIe 极速换入
    def _smart_load_models_to_device(self_pipe, model_names):
        if not getattr(self_pipe, "vram_management_enabled", False):
            return
        target_set = set(model_names or ())
        if not target_set:
            return

        # 快速路径：若请求的所有目标子模块均已 100% 处于 GPU 显存 (state=2)，0ms 直接放行，绝不误卸其他常驻模块
        target_children = [
            (name, mdl)
            for name, mdl in self_pipe.named_children()
            if name in target_set and mdl is not None
        ]
        if target_children and all(_is_submodel_fully_prepared_on_cuda(mdl) for _, mdl in target_children):
            return

        def _get_free_vram_gb() -> float:
            try:
                return torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
            except Exception:
                return 0.0

        free_gb = _get_free_vram_gb()
        need_heavy_backbone = is_minimax_bf16 and (("dit" in target_set) or ("text_encoder" in target_set))
        req_free_gb = 66.5 if need_heavy_backbone else 24.0

        # 第一级腾退：切换 MiniMax-H3 两大 62GB 主干时，优先仅腾退对侧闲置大主干，让轻量的 video_vae / audio_vae 在 G4 96GB 中全程钉死显存
        if need_heavy_backbone and free_gb < req_free_gb:
            offloaded_heavy = False
            for name, model in self_pipe.named_children():
                if name not in target_set and name in ("dit", "text_encoder") and model is not None:
                    if _offload_submodel_to_host(model):
                        offloaded_heavy = True
            if offloaded_heavy:
                gc.collect()
                torch.cuda.empty_cache()
                free_gb = _get_free_vram_gb()

        # 第二级腾退：仅当腾退大主干后显存仍不足 (例如与 21GB Qwen-GGUF 共驻显存时)，再按需腾退其余非目标子模块
        if free_gb < req_free_gb:
            offloaded_rest = False
            for name, model in self_pipe.named_children():
                if name not in target_set and model is not None:
                    if _offload_submodel_to_host(model):
                        offloaded_rest = True
            if offloaded_rest:
                gc.collect()
                torch.cuda.empty_cache()
                free_gb = _get_free_vram_gb()

        # 目标模块整块换入：显存充裕时直接一次性执行 preparing() 锁入 GPU (state=2)，彻底消灭推理时 800+ 层逐层同步钩子开销
        for name, model in target_children:
            if _is_submodel_fully_prepared_on_cuda(model):
                continue
            if getattr(model, "vram_management_enabled", False):
                t_sw = time.time()
                if hasattr(model, "onload"):
                    model.onload()
                else:
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()

                cur_free_gb = _get_free_vram_gb()
                bulk_need_gb = 66.0 if (is_minimax_bf16 and name in ("dit", "text_encoder")) else 22.0
                if cur_free_gb >= bulk_need_gb:
                    for module in model.modules():
                        if hasattr(module, "preparing") and getattr(module, "state", 2) != 2:
                            try:
                                module.preparing()
                            except Exception:
                                pass
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    dt_sw = time.time() - t_sw
                    if dt_sw >= 0.3:
                        print(f"⚡ [极速显存换入] 子模块 {name} 已整块载入 GPU 显存，耗时 {dt_sw:.1f}s")

    pipe.load_models_to_device = types.MethodType(_smart_load_models_to_device, pipe)
    pipe._high_vram_fast_mode = True

    # 3. 预热指定子模块至 GPU 显存 (state=2)，保留 >=22GB 安全激活余量
    if preload_vram:
        children = [
            (name, mdl)
            for name, mdl in pipe.named_children()
            if mdl is not None and name not in skip_preload_attrs
        ]
        for idx, (attr_name, model) in enumerate(children):
            free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
            if free_gb < 22.0:
                break
            prog = round(0.80 + 0.18 * ((idx + 1) / max(1, len(children))), 2)
            MODEL_DOWNLOAD_STATE["progress"] = prog
            MODEL_DOWNLOAD_STATE["message"] = f"[显存常驻加速] 正在将子模块 {attr_name} 预热锁入 GPU 显存..."
            for module in model.modules():
                if hasattr(module, "preparing") and getattr(module, "state", 2) != 2:
                    try:
                        module.preparing()
                    except Exception:
                        pass
        _trim_linux_cpu_heap()
        used_gb = (
            torch.cuda.mem_get_info("cuda")[1] - torch.cuda.mem_get_info("cuda")[0]
        ) / (1024 ** 3)
        print(
            f"[大显存常驻已激活] 管线子模块已预热至显存 (当前总驻留 {used_gb:.1f} GB / {total_vram_gb:.1f} GB)"
        )
        return "大显存常驻"

    return "大显存常驻加速"

def _reclaim_transient_gpu_memory():
    """在连续视频/图像生成前后主动回收 Python 引用与 CUDA 缓存碎片，杜绝第 2 条视频中途 OOM 崩溃"""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

def _run_pipeline_with_smart_tiling(pipe, call_kwargs: dict):
    """仅当实时空闲显存 >= 46GB 时启用整帧 VAE 解码 (tiled=False)；超长帧序列 (如 15s / 362帧) 触顶时自动开启梯度检查点兜底杜绝 OOM"""
    # 若生成高分辨率超长视频 (如 720P + >=260帧/11~15秒)，采样前若 video_vae 驻留显存且总空闲偏紧，先释放 VAE 留足 33GB+ 激活空间
    num_f = int(call_kwargs.get("num_frames") or 124)
    pix = int(call_kwargs.get("height") or 480) * int(call_kwargs.get("width") or 832)
    if pipe.__class__.__name__ == "MiniMaxH3Pipeline" and num_f >= 260 and pix >= 1280 * 700:
        try:
            vvae = getattr(pipe, "video_vae", None)
            if vvae is not None and hasattr(vvae, "offload"):
                vvae.offload()
                _reclaim_transient_gpu_memory()
            elif vvae is not None:
                for mod in vvae.modules():
                    if hasattr(mod, "offload"):
                        mod.offload()
                _reclaim_transient_gpu_memory()
        except Exception:
            pass

    def _invoke_pipe_with_ref_compat(p, kw: Dict[str, Any]):
        try:
            return p(**kw)
        except TypeError as te:
            # 兼容尚未升级 references 参数的旧版 DiffSynth MiniMaxH3Pipeline（自动平滑降级为首尾帧 keyframes）
            if "references" in kw and "unexpected keyword argument" in str(te):
                refs = kw.get("references") or []
                imgs = [r.get("image") for r in refs if isinstance(r, dict) and r.get("image") is not None]
                if not imgs:
                    for r in refs:
                        if isinstance(r, dict) and isinstance(r.get("video"), list) and r["video"]:
                            imgs.append(r["video"][0])
                            if len(r["video"]) > 1:
                                imgs.append(r["video"][-1])
                            break
                fb_kw = {k: v for k, v in kw.items() if k != "references"}
                if imgs:
                    fb_kw["keyframes"] = imgs[:2]
                    fb_kw["keyframe_indices"] = [0] if len(imgs) == 1 else [0, -1]
                return p(**fb_kw)
            raise

    if getattr(pipe, "_high_vram_fast_mode", False) and "tiled" not in call_kwargs:
        try:
            free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
        except Exception:
            free_gb = 0.0
        if free_gb >= 46.0:
            fast_kwargs = dict(call_kwargs)
            fast_kwargs["tiled"] = False
            try:
                return _invoke_pipe_with_ref_compat(pipe, fast_kwargs)
            except torch.cuda.OutOfMemoryError:
                print("⚠️ 整帧 VAE 解码显存触顶，正在清理缓存并自动切换为 Tiled 分块安全解码...")
                _reclaim_transient_gpu_memory()
    safe_kwargs = dict(call_kwargs)
    safe_kwargs.setdefault("tiled", True)
    try:
        return _invoke_pipe_with_ref_compat(pipe, safe_kwargs)
    except torch.cuda.OutOfMemoryError:
        if pipe.__class__.__name__ == "MiniMaxH3Pipeline" and not safe_kwargs.get("use_gradient_checkpointing"):
            print("⚠️ 超长序列注意力显存触顶，正在自动释放闲置子模块并启用梯度检查点安全重试...")
            for sub_name in ("video_vae", "audio_vae", "text_encoder"):
                sub_m = getattr(pipe, sub_name, None)
                if sub_m is not None:
                    if hasattr(sub_m, "offload"):
                        sub_m.offload()
                    else:
                        for mod in sub_m.modules():
                            if hasattr(mod, "offload"):
                                mod.offload()
            _reclaim_transient_gpu_memory()
            gc_kwargs = dict(safe_kwargs)
            gc_kwargs["use_gradient_checkpointing"] = True
            return _invoke_pipe_with_ref_compat(pipe, gc_kwargs)
        raise

# ==============================================================================
# ==============================================================================
# 视频生成核心函数与 MiniMax-H3 Ref2VA 多模态参考规范
# 官方上限：最多 9 张参考图 (<Picture 1>~<Picture 9>) + 最多 3 段参考视频 (<Video 1>~<Video 3>)
#         + 最多 3 段参考音频 (<Audio 1>~<Audio 3>)，单次总参考模态资产上限 12 个
# ==============================================================================
MAX_REF2VA_IMAGES = 9

MAX_REF2VA_VIDEOS = 3

MAX_REF2VA_AUDIOS = 3

MAX_REF2VA_TOTAL_ASSETS = 12

def _align_minimax_h3_frame_count(frame_count: int) -> int:
    """将帧数向上对齐至 MiniMax-H3 DiT 要求的 17k + 5 时间网格 (如 5, 22, 39, 56, 73, 90, 107, 124)"""
    current = max(int(frame_count), 5)
    while current % 17 != 5:
        current += 1
    return current

def load_reference_video_frames(
    video_source: Any,
    width: int = 832,
    height: int = 480,
    target_fps: int = 24,
    max_frames: int = 124,
) -> Optional[List[Image.Image]]:
    """
    将参考视频（文件路径、base64 字符串、字节流或已解码帧列表）重采样至 24fps 并对齐到 17k+5 帧数网格，
    返回 RGB PIL.Image 列表供 MiniMaxH3Pipeline references=[{"type": "video", "video": frames}] 使用。
    """
    if video_source is None:
        return None

    # 1. 若已是帧列表 (如单元测试传入或预处理完成的 PIL.Image 列表)
    if isinstance(video_source, (list, tuple)):
        raw_frames = list(video_source)
        if not raw_frames:
            return None
        num_out = min(max_frames, _align_minimax_h3_frame_count(len(raw_frames)))
        out_list: List[Image.Image] = []
        for idx in range(num_out):
            frm = raw_frames[min(idx, len(raw_frames) - 1)]
            if isinstance(frm, Image.Image):
                im = frm.convert("RGB")
                if im.size != (width, height):
                    im = im.resize((width, height))
                out_list.append(im)
        return out_list if out_list else None

    # 2. 解析文件路径或 base64 / bytes 临时文件
    temp_video_path: Optional[str] = None
    video_path: Optional[str] = None
    try:
        if isinstance(video_source, str):
            s_val = video_source.strip()
            if not s_val:
                return None
            if os.path.isfile(s_val):
                video_path = s_val
            else:
                if "," in s_val and s_val.split(",", 1)[0].lower().startswith("data:"):
                    s_val = s_val.split(",", 1)[1]
                raw_bytes = base64.b64decode(s_val)
                fd, temp_video_path = tempfile.mkstemp(suffix=".mp4", prefix="easycolab_refvid_")
                with os.fdopen(fd, "wb") as tf:
                    tf.write(raw_bytes)
                video_path = temp_video_path
        elif isinstance(video_source, (bytes, bytearray)):
            fd, temp_video_path = tempfile.mkstemp(suffix=".mp4", prefix="easycolab_refvid_")
            with os.fdopen(fd, "wb") as tf:
                tf.write(bytes(video_source))
            video_path = temp_video_path
        elif hasattr(video_source, "name") and isinstance(getattr(video_source, "name"), str) and os.path.isfile(video_source.name):
            video_path = video_source.name

        if not video_path or not os.path.isfile(video_path):
            return None

        raw_frames_loaded: List[Image.Image] = []
        src_fps = float(target_fps)

        # 优先使用 DiffSynth 官方 VideoData 读取器
        try:
            from diffsynth.utils.data import VideoData
            vd = VideoData(video_path, height=height, width=width)
            raw_frames_loaded = [
                (f.convert("RGB") if isinstance(f, Image.Image) else Image.fromarray(f).convert("RGB"))
                for f in vd.raw_data()
            ]
            meta_fps = vd.data.reader.get_meta_data().get("fps")
            if meta_fps and float(meta_fps) > 0:
                src_fps = float(meta_fps)
        except Exception:
            pass

        # 回退方案 1: PyAV (av)
        if not raw_frames_loaded:
            try:
                import av
                with av.open(video_path) as container:
                    if container.streams.video:
                        vstream = container.streams.video[0]
                        if vstream.average_rate:
                            src_fps = float(vstream.average_rate)
                        for packet_frame in container.decode(vstream):
                            pil_f = packet_frame.to_image().convert("RGB")
                            if pil_f.size != (width, height):
                                pil_f = pil_f.resize((width, height))
                            raw_frames_loaded.append(pil_f)
                            if len(raw_frames_loaded) >= max_frames * 3:
                                break
            except Exception:
                pass

        # 回退方案 2: imageio
        if not raw_frames_loaded:
            try:
                import imageio
                reader = imageio.get_reader(video_path)
                try:
                    meta = reader.get_meta_data()
                    if meta and meta.get("fps"):
                        src_fps = float(meta["fps"])
                except Exception:
                    pass
                for arr in reader:
                    pil_f = Image.fromarray(arr).convert("RGB")
                    if pil_f.size != (width, height):
                        pil_f = pil_f.resize((width, height))
                    raw_frames_loaded.append(pil_f)
                    if len(raw_frames_loaded) >= max_frames * 3:
                        break
                reader.close()
            except Exception:
                pass

        if not raw_frames_loaded:
            return None

        est_target_count = max(1, int(round(len(raw_frames_loaded) * float(target_fps) / max(1.0, src_fps))))
        num_out_frames = min(int(max_frames), _align_minimax_h3_frame_count(est_target_count))
        resampled: List[Image.Image] = []
        for k in range(num_out_frames):
            idx = int(round(k * src_fps / float(target_fps)))
            if idx >= len(raw_frames_loaded):
                idx = len(raw_frames_loaded) - 1
            frm = raw_frames_loaded[idx]
            if frm.size != (width, height):
                frm = frm.resize((width, height))
            resampled.append(frm)
        return resampled if resampled else None
    except Exception as e:
        print(f"[Ref2VA 视频参考解码提示] 解析参考视频跳过: {e}")
        return None
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

def decode_base64_video_frames(
    b64_str: Optional[str],
    width: int = 832,
    height: int = 480,
    target_fps: int = 24,
    max_frames: int = 124,
) -> Optional[List[Image.Image]]:
    if not b64_str or not str(b64_str).strip():
        return None
    return load_reference_video_frames(
        b64_str,
        width=width,
        height=height,
        target_fps=target_fps,
        max_frames=max_frames,
    )

def load_reference_audio_item(
    audio_source: Any,
    target_sr: int = 32000,
) -> Optional[Dict[str, Any]]:
    """
    将参考音频（文件路径、base64 字符串、字节流、Tensor 或 (waveform, sr) 元组）解码为
    MiniMaxH3Pipeline references 所需的 {"type": "audio", "audio": waveform_tensor, "sample_rate": sr} 字典。
    对应提示词中的 <Audio 1> ~ <Audio 3> 标签（按 1 起始顺序号严格对应）。
    """
    if audio_source is None:
        return None

    # 1. 若已是预处理好的字典 / 元组 / Tensor
    if isinstance(audio_source, dict) and audio_source.get("audio") is not None:
        return {
            "type": "audio",
            "audio": audio_source["audio"],
            "sample_rate": int(audio_source.get("sample_rate") or target_sr),
        }
    if isinstance(audio_source, tuple) and len(audio_source) == 2:
        wf, sr = audio_source
        if not isinstance(wf, torch.Tensor):
            wf = torch.as_tensor(wf, dtype=torch.float32)
        return {"type": "audio", "audio": wf, "sample_rate": int(sr or target_sr)}
    if isinstance(audio_source, torch.Tensor):
        return {"type": "audio", "audio": audio_source, "sample_rate": int(target_sr)}

    temp_audio_path: Optional[str] = None
    audio_path: Optional[str] = None
    try:
        if isinstance(audio_source, str):
            s_val = audio_source.strip()
            if not s_val:
                return None
            if os.path.isfile(s_val):
                audio_path = s_val
            else:
                if "," in s_val and s_val.split(",", 1)[0].lower().startswith("data:"):
                    s_val = s_val.split(",", 1)[1]
                raw_bytes = base64.b64decode(s_val)
                fd, temp_audio_path = tempfile.mkstemp(suffix=".wav", prefix="easycolab_refaud_")
                with os.fdopen(fd, "wb") as tf:
                    tf.write(raw_bytes)
                audio_path = temp_audio_path
        elif isinstance(audio_source, (bytes, bytearray)):
            fd, temp_audio_path = tempfile.mkstemp(suffix=".wav", prefix="easycolab_refaud_")
            with os.fdopen(fd, "wb") as tf:
                tf.write(bytes(audio_source))
            audio_path = temp_audio_path
        elif hasattr(audio_source, "name") and isinstance(getattr(audio_source, "name"), str) and os.path.isfile(audio_source.name):
            audio_path = audio_source.name

        if not audio_path or not os.path.isfile(audio_path):
            return None

        # 优先使用 DiffSynth 官方 read_audio
        try:
            from diffsynth.utils.data.audio import read_audio
            res = read_audio(audio_path)
            if isinstance(res, tuple) and len(res) >= 2:
                wf, sr = res[0], int(res[1])
                if not isinstance(wf, torch.Tensor):
                    wf = torch.as_tensor(wf, dtype=torch.float32)
                return {"type": "audio", "audio": wf, "sample_rate": sr}
        except Exception:
            pass

        # 回退方案 1: torchaudio
        try:
            import torchaudio
            wf, sr = torchaudio.load(audio_path)
            return {"type": "audio", "audio": wf.to(torch.float32), "sample_rate": int(sr)}
        except Exception:
            pass

        # 回退方案 2: soundfile
        try:
            import soundfile as sf
            data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            wf = torch.from_numpy(data.T).to(torch.float32)
            return {"type": "audio", "audio": wf, "sample_rate": int(sr)}
        except Exception:
            pass

        # 回退方案 3: Python 标准库 wave (支持 PCM WAV)
        try:
            import wave
            import array
            with wave.open(audio_path, "rb") as wf_reader:
                sr = wf_reader.getframerate() or target_sr
                n_channels = max(1, wf_reader.getnchannels())
                sampwidth = wf_reader.getsampwidth()
                n_frames = wf_reader.getnframes()
                raw_pcm = wf_reader.readframes(n_frames)
                if sampwidth == 2:
                    samples = array.array("h", raw_pcm)
                    t_pcm = torch.tensor(samples, dtype=torch.float32) / 32768.0
                elif sampwidth == 1:
                    samples = array.array("B", raw_pcm)
                    t_pcm = (torch.tensor(samples, dtype=torch.float32) - 128.0) / 128.0
                else:
                    t_pcm = torch.zeros((n_channels, max(1, n_frames)), dtype=torch.float32)
                if t_pcm.ndim == 1:
                    t_pcm = t_pcm.view(-1, n_channels).transpose(0, 1).contiguous()
                return {"type": "audio", "audio": t_pcm, "sample_rate": int(sr)}
        except Exception:
            pass

        return None
    except Exception as e:
        print(f"[Ref2VA 音频参考解码提示] 解析参考音频跳过: {e}")
        return None
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass

def build_minimax_h3_references(
    ref_images: Optional[Sequence[Any]] = None,
    ref_videos: Optional[Sequence[Any]] = None,
    ref_audios: Optional[Sequence[Any]] = None,
    width: int = 832,
    height: int = 480,
    fps: int = 24,
    max_video_frames: int = 124,
) -> List[Dict[str, Any]]:
    """
    按 MiniMax-H3 Ref2VA 官方模型上限构建 references 多模态列表（各模态内部严格按 1 起始顺序号对应提示词标签）：
    - 参考图片 (type="image"): 最多 9 张 (<Picture 1> ~ <Picture 9> / <Subject 1> ~ <Subject 9>)
    - 参考视频 (type="video"): 最多 3 段 (<Video 1> ~ <Video 3>)
    - 参考音频 (type="audio"): 最多 3 段 (<Audio 1> ~ <Audio 3>)
    - 总参考资产上限: 12 个
    """
    refs: List[Dict[str, Any]] = []

    if ref_images:
        img_count = 0
        for item in ref_images:
            if img_count >= MAX_REF2VA_IMAGES or len(refs) >= MAX_REF2VA_TOTAL_ASSETS:
                break
            if item is None:
                continue
            pil_im: Optional[Image.Image] = None
            if isinstance(item, Image.Image):
                pil_im = item.convert("RGB")
            elif isinstance(item, str):
                s_item = item.strip()
                if not s_item:
                    continue
                if os.path.isfile(s_item):
                    try:
                        pil_im = Image.open(s_item).convert("RGB")
                    except Exception:
                        pil_im = None
                else:
                    try:
                        pil_im = decode_base64_image(s_item)
                    except Exception:
                        pil_im = None
            elif hasattr(item, "name") and isinstance(getattr(item, "name"), str) and os.path.isfile(item.name):
                try:
                    pil_im = Image.open(item.name).convert("RGB")
                except Exception:
                    pil_im = None
            if pil_im is not None:
                refs.append({"type": "image", "image": pil_im})
                img_count += 1

    if ref_videos:
        vid_count = 0
        for v_item in ref_videos:
            if vid_count >= MAX_REF2VA_VIDEOS or len(refs) >= MAX_REF2VA_TOTAL_ASSETS:
                break
            if v_item is None:
                continue
            v_frames = load_reference_video_frames(
                v_item,
                width=int(width),
                height=int(height),
                target_fps=int(fps),
                max_frames=int(max_video_frames),
            )
            if v_frames:
                refs.append({"type": "video", "video": v_frames})
                vid_count += 1

    if ref_audios:
        aud_count = 0
        for a_item in ref_audios:
            if aud_count >= MAX_REF2VA_AUDIOS or len(refs) >= MAX_REF2VA_TOTAL_ASSETS:
                break
            if a_item is None:
                continue
            aud_entry = load_reference_audio_item(a_item)
            if aud_entry is not None:
                refs.append(aud_entry)
                aud_count += 1

    return refs

RESOLUTION_MAP = {
    "832 × 480 (16:9 横屏标准 - 推荐)": (832, 480),
    "480 × 832 (9:16 竖屏短视频)": (480, 832),
    "1280 × 720 (16:9 720P 高清)": (1280, 720),
    "720 × 1280 (9:16 720P 竖屏高清)": (720, 1280),
    "640 × 640 (1:1 方形)": (640, 640),
}

def generate_video(
    task_mode: str,
    visual_prompt: str,
    soundscape_prompt: str,
    music_prompt: str,
    first_frame_img: Optional[Image.Image],
    last_frame_img: Optional[Image.Image],
    resolution_label: str,
    num_frames: int,
    num_inference_steps: int,
    fps: int,
    seed: int,
    randomize_seed: bool,
    save_dir: str,
    ref_images_files: Optional[List[Any]] = None,
    ref_videos_files: Optional[List[Any]] = None,
    ref_audios_files: Optional[List[Any]] = None,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], str, int, List[str]]:
    # (shared state on runtime_state)
    touch_activity()

    if not visual_prompt or not visual_prompt.strip():
        return None, "⚠️ 请输入画面描述提示词 (Prompt)！", seed, get_history_videos(save_dir)

    if rs.PIPELINE is None:
        return (
            None,
            "⚠️ 模型尚未加载到 GPU 显存！请先在左侧点击「1️⃣ 下载模型权重」与「2️⃣ 加载模型至显存」。",
            seed,
            get_history_videos(save_dir),
        )

    if randomize_seed or seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    width, height = RESOLUTION_MAP.get(resolution_label, (832, 480))
    full_prompt = build_structured_prompt(visual_prompt, soundscape_prompt, music_prompt)

    keyframes = []
    keyframe_indices = []
    references = []
    if task_mode == "fl2va":
        if first_frame_img is not None:
            keyframes.append(first_frame_img.convert("RGB"))
            keyframe_indices.append(0)
        if last_frame_img is not None:
            keyframes.append(last_frame_img.convert("RGB"))
            keyframe_indices.append(-1)
        if not keyframes:
            return (
                None,
                "⚠️ 在「首尾帧图生视频」模式下，请至少上传一张【首帧图片】或【尾帧图片】！",
                seed,
                get_history_videos(save_dir),
            )
    elif task_mode == "ref2va":
        candidate_imgs: List[Any] = [im for im in (first_frame_img, last_frame_img) if im is not None]
        if ref_images_files:
            candidate_imgs.extend(ref_images_files if isinstance(ref_images_files, list) else [ref_images_files])
        candidate_vids: List[Any] = []
        if ref_videos_files:
            candidate_vids.extend(ref_videos_files if isinstance(ref_videos_files, list) else [ref_videos_files])
        candidate_auds: List[Any] = []
        if ref_audios_files:
            candidate_auds.extend(ref_audios_files if isinstance(ref_audios_files, list) else [ref_audios_files])
        references = build_minimax_h3_references(
            ref_images=candidate_imgs[:MAX_REF2VA_IMAGES],
            ref_videos=candidate_vids[:MAX_REF2VA_VIDEOS],
            ref_audios=candidate_auds[:MAX_REF2VA_AUDIOS],
            width=width,
            height=height,
            fps=int(fps),
            max_video_frames=int(num_frames),
        )
        if not references:
            return (
                None,
                "⚠️ 在「主体参考生音视频 (Ref2VA)」模式下，请至少上传 1 张参考图片（`<Picture 1>`~`<Picture 9>`）、1 段参考视频（`<Video 1>`~`<Video 3>`）或 1 段参考音频（`<Audio 1>`~`<Audio 3>`）！",
                seed,
                get_history_videos(save_dir),
            )

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"minimax_h3_{task_mode}_{timestamp}_seed{seed}.mp4"
    output_path = os.path.join(save_dir, filename)

    start_time = time.time()
    with GPU_LOCK:
        try:
            from diffsynth.utils.data.audio_video import write_video_audio

            if progress is not None:
                progress(0.05, desc="正在准备潜空间与多模态编码...")

            call_kwargs = dict(
                prompt=full_prompt,
                height=height,
                width=width,
                num_frames=int(num_frames),
                num_inference_steps=int(num_inference_steps),
                seed=int(seed),
            )
            if keyframes:
                call_kwargs["keyframes"] = keyframes
                call_kwargs["keyframe_indices"] = keyframe_indices
            elif references:
                call_kwargs["references"] = references

            _reclaim_transient_gpu_memory()
            video, audio = _run_pipeline_with_smart_tiling(rs.PIPELINE, call_kwargs)

            if progress is not None:
                progress(0.92, desc="正在合成 32kHz 立体声音轨与 MP4 视频...")

            write_video_audio(
                video=video,
                audio=audio,
                output_path=output_path,
                fps=int(fps),
                audio_sample_rate=32000,
            )
            video = None
            audio = None
            _reclaim_transient_gpu_memory()

            elapsed = time.time() - start_time
            status_msg = (
                f"✅ 生成完成！耗时: {elapsed:.1f} 秒 | 分辨率: {width}×{height} | "
                f"帧数: {num_frames} ({num_frames/fps:.1f}s) | Seed: {seed}\n"
                f"📁 已保存至: {output_path}"
            )
            return output_path, status_msg, seed, get_history_videos(save_dir)

        except Exception as e:
            _reclaim_transient_gpu_memory()
            return None, f"❌ 生成过程中出错: {str(e)}", seed, get_history_videos(save_dir)
        finally:
            touch_activity()

def _fmt_mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"

def _pipeline_job_execution_guard(
    pipe,
    job_id: str,
    job_cancel_event: threading.Event,
    start_time: float,
    base_prog: float,
    span_prog: float,
    prefix: str,
    model_name: str,
    media_type: str,
):
    """
    为所有 DiffSynth 视频/生图管线注入全链路真实进度与毫秒级 GPU 中断守卫：
    1. 挂载层块级 forward_pre_hook：点击停止推理后在当前 Transformer Block 结束时 (<0.1s) 立即抛出 GenerationAbortedError，
       瞬时释放 GPU_LOCK，彻底杜绝僵尸任务阻塞新任务。
    2. 接管 progress_bar_cmd 与阶段切换 load_models_to_device：
       - 0.04 ~ 0.08: 多模态提示词编码与潜空间初始化
       - 0.08 ~ 0.82 (图像 0.90): DiT 逐步去噪采样（每步实时刷新百分比与剩余时间，单行刷新不堆叠）
       - 0.84 ~ 0.93: 3D 视频 VAE 与 32kHz 立体声 VAE 解码（明确区分采样完成与后处理解码，绝不提前报 100%）
       - 0.94 ~ 0.99: MP4 音轨合成与落盘
    """
    # (shared state on runtime_state)
    import types

    dit_end = 0.90 if media_type == "image" else 0.82
    hook_handles = []
    orig_load_fn = getattr(pipe, "load_models_to_device", None)

    def _check_abort():
        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            raise GenerationAbortedError("已手动停止当前推理任务")

    def _abort_pre_hook(_mod, _inp):
        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            raise GenerationAbortedError("已手动停止当前推理任务")

    # 为管线主干及内部 Transformer Blocks 挂载轻量级中断钩子，使长耗时单步也能在 ~100ms 内立刻中止
    try:
        seen_mods = set()
        for _, child in pipe.named_children():
            if child is None or not isinstance(child, torch.nn.Module):
                continue
            if id(child) not in seen_mods:
                seen_mods.add(id(child))
                hook_handles.append(child.register_forward_pre_hook(_abort_pre_hook))
            for sub_name, sub_mod in child.named_modules():
                if isinstance(sub_mod, torch.nn.ModuleList) and any(
                    k in sub_name for k in ("blocks", "layers")
                ):
                    for blk in sub_mod:
                        if isinstance(blk, torch.nn.Module) and id(blk) not in seen_mods:
                            seen_mods.add(id(blk))
                            hook_handles.append(blk.register_forward_pre_hook(_abort_pre_hook))
    except Exception:
        pass

    if orig_load_fn is not None:
        def _stage_aware_load_models(self_pipe, model_names):
            # (shared state on runtime_state)
            _check_abort()
            names_set = set(model_names or ())
            if "text_encoder" in names_set:
                rs.CURRENT_STAGE_BAR_LABEL = "多模态提示词编码"
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + 0.05 * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = f"{prefix}{model_name} 正在使用多模态编码器解析提示词 (5%)..."
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = "提示词编码:   5%|#---------| 正在解析画面、运镜与声音提示词..."
            elif "video_vae" in names_set or "vae" in names_set:
                v_frac = 0.92 if media_type == "image" else 0.85
                lbl = "高清图像 VAE 解码" if media_type == "image" else "3D 视频 VAE 解码"
                rs.CURRENT_STAGE_BAR_LABEL = lbl
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + v_frac * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = f"{prefix}DiT 采样已完成，正在执行{lbl} ({int(v_frac * 100)}%)..."
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = f"{lbl}: {int(v_frac * 100):3d}%|########--| 正在将潜空间张量解码为高清像素..."
            elif "audio_vae" in names_set:
                rs.CURRENT_STAGE_BAR_LABEL = "32kHz 立体声解码"
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + 0.92 * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = f"{prefix}视频帧解码完成，正在执行 32kHz 立体声音轨解码 (92%)..."
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = "32kHz 立体声解码:  92%|#########-| 正在解码双声道环境音与配乐波形..."
            res = orig_load_fn(model_names)
            _check_abort()
            touch_activity()
            return res

        pipe.load_models_to_device = types.MethodType(_stage_aware_load_models, pipe)

    def _job_progress_bar(iterable, *_args, **_kwargs):
        # (shared state on runtime_state)
        _check_abort()
        items = list(iterable)
        total = max(1, len(items))
        loop_t0 = time.time()
        rs.CURRENT_STAGE_BAR_LABEL = None

        if job_id in JOBS:
            JOBS[job_id]["progress"] = round(base_prog + 0.08 * span_prog, 3)
            JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
            JOBS[job_id]["message"] = f"{prefix}{model_name} 正在执行去噪采样第 1/{total} 步 (8%)..."
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS["sampling"] = f"DiT 去噪采样:   0%|----------| 0/{total} [正在执行第 1/{total} 步...]"

        for idx, item in enumerate(items):
            _check_abort()
            yield item
            _check_abort()
            done = idx + 1
            step_frac = done / float(total)
            dt = max(0.01, time.time() - loop_t0)
            avg_s = dt / float(done)
            eta_s = max(0.0, avg_s * (total - done))
            overall_frac = 0.08 + (dit_end - 0.08) * step_frac
            fill = max(0, min(10, int(round(10 * step_frac))))
            touch_activity()

            if done < total:
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + overall_frac * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = (
                        f"{prefix}{model_name} 正在采样 {done}/{total} 步 "
                        f"(总进度 {int(overall_frac * 100)}% · {avg_s:.1f}s/步 · 采样剩余约 {int(round(eta_s))}s)..."
                    )
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = (
                        f"DiT 去噪采样: {int(step_frac * 100):3d}%|{'#' * fill}{'-' * (10 - fill)}| "
                        f"{done}/{total} [{_fmt_mmss(dt)}<{_fmt_mmss(eta_s)}, {avg_s:.2f}s/step] · 总进度 {int(overall_frac * 100)}%"
                    )
            else:
                post_frac = dit_end + 0.02
                next_lbl = "高清图像 VAE 解码" if media_type == "image" else "3D 视频与音频 VAE 解码"
                rs.CURRENT_STAGE_BAR_LABEL = next_lbl
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + post_frac * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = (
                        f"{prefix}DiT 采样已完成 ({total}/{total})，正在进入{next_lbl} ({int(post_frac * 100)}%)..."
                    )
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = (
                        f"{next_lbl}: {int(post_frac * 100):3d}%|########--| "
                        f"DiT 采样已完成 ({total}/{total}，耗时 {_fmt_mmss(dt)}) -> 正在解码输出..."
                    )
                print(f"[{model_name}] DiT 采样 {total}/{total} 步完成 ({dt:.1f}s)，开始执行{next_lbl}...")

    try:
        yield _job_progress_bar
    finally:
        rs.CURRENT_STAGE_BAR_LABEL = None
        if orig_load_fn is not None:
            pipe.load_models_to_device = orig_load_fn
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass

def _run_async_generation_job(job_id: str, req: GenerateVideoRequest):
    touch_activity()
    job_cancel_event = _JOB_CANCEL_EVENTS.setdefault(job_id, threading.Event())
    active_model_id = (
        req.model_id
        if (req.model_id and req.model_id in MODEL_CATALOG)
        else (rs.SELECTED_MODEL_ID if rs.SELECTED_MODEL_ID in LOADED_RUNTIMES else (rs.LOADED_MODEL_ID or rs.SELECTED_MODEL_ID))
    )
    batch_count = max(1, min(8, int(getattr(req, "batch_count", 1) or 1)))

    JOBS[job_id]["model_id"] = active_model_id
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["progress"] = 0.01
    JOBS[job_id]["batch_count"] = batch_count
    JOBS[job_id]["completed_count"] = 0
    JOBS[job_id]["completed_videos"] = []
    start_time = time.time()
    JOBS[job_id]["started_at"] = start_time

    try:
        if job_cancel_event.is_set():
            raise GenerationAbortedError("已手动停止当前推理任务")

        # 分支 A：Qwen3.8-27B / Qwen3.8-27B-GGUF 视觉理解与深度推理对话
        if _is_llm_model_id(active_model_id):
            _run_qwen_llm_job(job_id, req, target_llm_id=active_model_id)
            return

        # 智能路由 MiniMax-H3 与 MiniMax-H3-Ref2VA：
        # 1. 若用户在 MiniMax-H3 下选择 ref2va 模式且 minimax_h3_ref2va 已在显存，优先命中 Ref2VA 专用主干；
        # 2. 若请求的模型未加载但配对兄弟模型已加载，自动复用已驻留的 MiniMaxH3Pipeline 管线免重复加载报错。
        if active_model_id == "minimax_h3" and req.task_mode == "ref2va" and "minimax_h3_ref2va" in LOADED_RUNTIMES:
            active_model_id = "minimax_h3_ref2va"
            JOBS[job_id]["model_id"] = active_model_id
        elif active_model_id not in LOADED_RUNTIMES:
            if active_model_id == "minimax_h3_ref2va" and "minimax_h3" in LOADED_RUNTIMES:
                active_model_id = "minimax_h3"
            elif active_model_id == "minimax_h3" and "minimax_h3_ref2va" in LOADED_RUNTIMES:
                active_model_id = "minimax_h3_ref2va"

        rt = LOADED_RUNTIMES.get(active_model_id)
        active_pipe = rt.get("pipeline") if rt else (rs.PIPELINE if rs.LOADED_MODEL_ID == active_model_id else None)
        if active_pipe is None:
            req_name = MODEL_CATALOG.get(active_model_id, {}).get("name", active_model_id)
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = f"{req_name} 尚未加载到 GPU 显存！请先在左侧点击「加载到显存」（显存充裕时可与现有模型同时常驻）。"
            return

        spec = MODEL_CATALOG.get(active_model_id, MODEL_CATALOG["minimax_h3"])
        pipeline_arch = spec.get("pipeline_arch") or active_model_id
        modality = spec["modality"]
        media_type = "image" if modality == "image" else "video"
        JOBS[job_id]["media_type"] = media_type
        JOBS[job_id]["progress"] = 0.02
        JOBS[job_id]["message"] = (
            f"正在准备 {spec['name']} 推理管线 (共 {batch_count} 项)..."
            if batch_count > 1
            else f"正在准备 {spec['name']} 推理管线..."
        )

        base_seed = req.seed
        if req.randomize_seed or base_seed < 0:
            base_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        JOBS[job_id]["seed"] = base_seed

        first_img = decode_base64_image(req.first_frame_base64)
        last_img = decode_base64_image(req.last_frame_base64)

        # 按官方上限（最多 9 张参考图 + 最多 3 段参考视频，去重保序）收集 Ref2VA 多模态参考资产
        raw_ref_img_b64s: List[str] = []
        explicit_ref_b64s = [s.strip() for s in (req.ref_images_base64 or []) if s and s.strip()]
        for legacy_b64 in (req.first_frame_base64, req.last_frame_base64):
            if legacy_b64 and legacy_b64.strip() and legacy_b64.strip() not in explicit_ref_b64s:
                raw_ref_img_b64s.append(legacy_b64.strip())
        raw_ref_img_b64s.extend(explicit_ref_b64s)
        raw_ref_img_b64s = raw_ref_img_b64s[:MAX_REF2VA_IMAGES]

        raw_ref_vid_b64s: List[str] = [
            v.strip() for v in (req.ref_videos_base64 or []) if v and str(v).strip()
        ][:MAX_REF2VA_VIDEOS]
        raw_ref_aud_b64s: List[str] = [
            a.strip() for a in (req.ref_audios_base64 or []) if a and str(a).strip()
        ][:MAX_REF2VA_AUDIOS]

        keyframes = []
        keyframe_indices = []
        references = []
        if pipeline_arch in ("minimax_h3", "minimax_h3_ref2va"):
            if req.task_mode == "fl2va":
                if first_img is not None:
                    keyframes.append(first_img)
                    keyframe_indices.append(0)
                if last_img is not None:
                    keyframes.append(last_img)
                    keyframe_indices.append(-1)
                if not keyframes:
                    JOBS[job_id]["status"] = "failed"
                    JOBS[job_id]["message"] = "首尾帧模式下请至少提供首帧或尾帧图片"
                    return
            elif req.task_mode == "ref2va" or (
                pipeline_arch == "minimax_h3_ref2va" and (raw_ref_img_b64s or raw_ref_vid_b64s or raw_ref_aud_b64s)
            ):
                references = build_minimax_h3_references(
                    ref_images=raw_ref_img_b64s,
                    ref_videos=raw_ref_vid_b64s,
                    ref_audios=raw_ref_aud_b64s,
                    width=int(req.width),
                    height=int(req.height),
                    fps=int(req.fps),
                    max_video_frames=int(req.num_frames),
                )
                if req.task_mode == "ref2va" and not references:
                    JOBS[job_id]["status"] = "failed"
                    JOBS[job_id]["message"] = (
                        "主体参考生视频 (Ref2VA) 模式下请至少提供 1 张参考图 (<Picture 1>~<Picture 9>)、1 段参考视频 (<Video 1>~<Video 3>) 或 1 段参考音频 (<Audio 1>~<Audio 3>)"
                    )
                    return
        elif modality == "image" and req.task_mode == "i2i_edit" and first_img is None and last_img is None:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = "图像编辑/重绘模式下请至少拖入一张输入图片"
            return

        os.makedirs(rs.DEFAULT_OUTPUT_DIR, exist_ok=True)
        last_filename = None
        last_seed = base_seed

        for batch_idx in range(batch_count):
            if job_cancel_event.is_set():
                raise GenerationAbortedError("已手动停止当前推理任务")

            if batch_idx == 0:
                current_seed = base_seed
            elif req.randomize_seed:
                current_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            else:
                current_seed = base_seed + batch_idx

            last_seed = current_seed
            JOBS[job_id]["seed"] = current_seed
            prefix = f"[{batch_idx + 1}/{batch_count}] " if batch_count > 1 else ""
            base_prog = batch_idx / float(batch_count)
            span_prog = 1.0 / float(batch_count)

            if GPU_LOCK.locked():
                JOBS[job_id]["progress"] = round(base_prog + 0.01 * span_prog, 3)
                JOBS[job_id]["message"] = f"{prefix}正在等待上一任务释放 GPU 显存锁..."

            with GPU_LOCK:
                if job_cancel_event.is_set():
                    raise GenerationAbortedError("已手动停止当前推理任务")
                ABORT_GENERATION_EVENT.clear()

                # 获取 GPU 锁后重置本轮起始计时并清空所有旧进度条，确保新任务严格从 0% 与 0.0s 起步
                sub_start = time.time()
                if batch_idx == 0:
                    start_time = sub_start
                    JOBS[job_id]["started_at"] = start_time
                    JOBS[job_id]["elapsed"] = 0.0
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS.clear()

                JOBS[job_id]["progress"] = round(base_prog + 0.04 * span_prog, 3)
                JOBS[job_id]["message"] = (
                    f"{prefix}{spec['name']} 正在初始化潜空间并准备 {req.num_inference_steps} 步推理 (Seed: {current_seed})..."
                )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                touch_activity()
                _reclaim_transient_gpu_memory()

                with _pipeline_job_execution_guard(
                    pipe=active_pipe,
                    job_id=job_id,
                    job_cancel_event=job_cancel_event,
                    start_time=start_time,
                    base_prog=base_prog,
                    span_prog=span_prog,
                    prefix=prefix,
                    model_name=spec["name"],
                    media_type=media_type,
                ) as step_progress_bar:
                    if pipeline_arch in ("minimax_h3", "minimax_h3_ref2va"):
                        from diffsynth.utils.data.audio_video import write_video_audio
                        full_prompt = build_structured_prompt(
                            req.visual_prompt, req.soundscape_prompt, req.music_prompt
                        )
                        filename = f"minimax_h3_{req.task_mode}_{timestamp}_seed{current_seed}.mp4"
                        output_path = os.path.join(rs.DEFAULT_OUTPUT_DIR, filename)
                        call_kwargs = dict(
                            prompt=full_prompt,
                            height=int(req.height),
                            width=int(req.width),
                            num_frames=int(req.num_frames),
                            num_inference_steps=int(req.num_inference_steps),
                            seed=int(current_seed),
                            progress_bar_cmd=step_progress_bar,
                        )
                        if keyframes:
                            call_kwargs["keyframes"] = keyframes
                            call_kwargs["keyframe_indices"] = keyframe_indices
                        elif references:
                            call_kwargs["references"] = references
                        video, audio = _run_pipeline_with_smart_tiling(active_pipe, call_kwargs)
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.95 * span_prog, 3)
                        JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                        JOBS[job_id]["message"] = f"{prefix}VAE 解码完成，正在合成 32kHz 立体声音轨并封装 MP4 (95%)..."
                        with CONSOLE_LOCK:
                            ACTIVE_TQDM_BARS["sampling"] = "音视频封装:  95%|#########-| 正在编码 H.264 视频与 32kHz 立体声 MP4..."
                        write_video_audio(
                            video=video,
                            audio=audio,
                            output_path=output_path,
                            fps=int(req.fps),
                            audio_sample_rate=32000,
                        )
                        video = None
                        audio = None

                    elif pipeline_arch == "qwen_image_21":
                        filename = f"qwen_image21_{req.task_mode}_{timestamp}_seed{current_seed}.png"
                        output_path = os.path.join(rs.DEFAULT_OUTPUT_DIR, filename)
                        edit_imgs = [im for im in (first_img, last_img) if im is not None] if req.task_mode == "i2i_edit" else None
                        img_out = active_pipe(
                            prompt=req.visual_prompt,
                            negative_prompt=req.negative_prompt.strip() or " ",
                            edit_image=edit_imgs if edit_imgs else None,
                            height=int(req.height),
                            width=int(req.width),
                            num_inference_steps=int(req.num_inference_steps),
                            seed=int(current_seed),
                            progress_bar_cmd=step_progress_bar,
                        )
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.96 * span_prog, 3)
                        JOBS[job_id]["message"] = f"{prefix}正在写入无损 PNG 图像文件 (96%)..."
                        img_out.save(output_path)
                        img_out = None
                        edit_imgs = None

                    elif pipeline_arch == "z_image_turbo":
                        filename = f"z_image_turbo_{req.task_mode}_{timestamp}_seed{current_seed}.png"
                        output_path = os.path.join(rs.DEFAULT_OUTPUT_DIR, filename)
                        w, h = int(req.width), int(req.height)
                        call_kw = dict(
                            prompt=req.visual_prompt,
                            negative_prompt=req.negative_prompt or "",
                            height=h,
                            width=w,
                            num_inference_steps=int(req.num_inference_steps),
                            seed=int(current_seed),
                            rand_device="cuda",
                            progress_bar_cmd=step_progress_bar,
                        )
                        if req.task_mode == "i2i_edit" and first_img is not None:
                            call_kw["input_image"] = first_img.resize((w, h))
                            call_kw["denoising_strength"] = max(0.15, min(1.0, float(req.denoising_strength or 0.70)))
                        img_out = active_pipe(**call_kw)
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.96 * span_prog, 3)
                        JOBS[job_id]["message"] = f"{prefix}正在写入无损 PNG 图像文件 (96%)..."
                        img_out.save(output_path)
                        img_out = None
                        call_kw.clear()

                    elif pipeline_arch == "minimax_h3_image":
                        filename = f"minimax_h3_image_{req.task_mode}_{timestamp}_seed{current_seed}.png"
                        output_path = os.path.join(rs.DEFAULT_OUTPUT_DIR, filename)
                        w, h = int(req.width), int(req.height)
                        ref_imgs = [im for im in (first_img, last_img) if im is not None] if req.task_mode == "i2i_edit" else None
                        JOBS[job_id]["progress"] = round(base_prog + 0.18 * span_prog, 3)
                        JOBS[job_id]["message"] = f"{prefix}正在执行 MiniMax-H3-Image 6 步 PDD 蒸馏原生超清采样 ({w}×{h})..."
                        with CONSOLE_LOCK:
                            ACTIVE_TQDM_BARS["sampling"] = (
                                f"MiniMax-H3-Image 采样:  25%|##5       | 6 步 PDD 蒸馏推理中 ({w}×{h})..."
                            )
                        img_out = rs.resolve_callable("run_minimax_h3_image_inference", run_minimax_h3_image_inference)(
                            active_pipe=active_pipe,
                            prompt=req.visual_prompt,
                            width=w,
                            height=h,
                            seed=int(current_seed),
                            ref_images=ref_imgs,
                        )
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.96 * span_prog, 3)
                        JOBS[job_id]["message"] = f"{prefix}正在写入无损 PNG 图像文件 (96%)..."
                        img_out.save(output_path)
                        img_out = None
                        ref_imgs = None

                    elif pipeline_arch == "diffusers_image":
                        filename = f"{active_model_id}_{req.task_mode}_{timestamp}_seed{current_seed}.png"
                        output_path = os.path.join(rs.DEFAULT_OUTPUT_DIR, filename)
                        w, h = int(req.width), int(req.height)
                        gen = torch.Generator(device="cuda").manual_seed(int(current_seed))
                        out_res = active_pipe(
                            prompt=req.visual_prompt,
                            height=h,
                            width=w,
                            num_inference_steps=int(req.num_inference_steps),
                            generator=gen,
                        )
                        img_out = out_res.images[0] if hasattr(out_res, "images") else out_res
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.96 * span_prog, 3)
                        JOBS[job_id]["message"] = f"{prefix}正在写入无损 PNG 图像文件 (96%)..."
                        img_out.save(output_path)
                        img_out = None

                    else:
                        raise ValueError(f"未知模型 ID: {active_model_id}")

                if not os.path.exists(output_path) or os.path.getsize(output_path) < 256:
                    raise RuntimeError("输出媒体文件未成功生成或写入为空")

                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS.clear()
                _reclaim_transient_gpu_memory()

            sub_elapsed = round(time.time() - sub_start, 1)
            last_filename = filename
            kind_cn = "图像" if media_type == "image" else "视频"
            print(f"[{spec['name']}] {kind_cn}生成完成 | 耗时 {sub_elapsed}s | Seed: {current_seed} | 文件: {filename}")
            media_item = {
                "index": batch_idx + 1,
                "seed": current_seed,
                "filename": filename,
                "video_url": f"/minimax_api/video/{filename}",
                "media_type": media_type,
                "elapsed": sub_elapsed,
            }
            JOBS[job_id]["completed_videos"] = list(JOBS[job_id]["completed_videos"]) + [media_item]
            JOBS[job_id]["completed_count"] = batch_idx + 1
            JOBS[job_id]["filename"] = filename
            JOBS[job_id]["video_url"] = f"/minimax_api/video/{filename}"
            JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
            if batch_count > 1 and batch_idx + 1 < batch_count:
                JOBS[job_id]["progress"] = round((batch_idx + 1) / float(batch_count), 3)
                JOBS[job_id]["message"] = (
                    f"第 {batch_idx + 1}/{batch_count} 项已完成 ({sub_elapsed}s)，已同步回传 Mac！继续生成第 {batch_idx + 2} 项..."
                )
            touch_activity()

        elapsed = round(time.time() - start_time, 1)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 1.0
        JOBS[job_id]["elapsed"] = elapsed
        JOBS[job_id]["filename"] = last_filename
        JOBS[job_id]["video_url"] = f"/minimax_api/video/{last_filename}" if last_filename else None
        if batch_count > 1:
            JOBS[job_id]["message"] = (
                f"批量生成 {batch_count} 项全部完成！总耗时 {elapsed}s (平均 {round(elapsed / batch_count, 1)}s/项)"
            )
        else:
            JOBS[job_id]["message"] = f"生成完成！耗时 {elapsed}s (Seed: {last_seed})"

    except GenerationAbortedError as abort_err:
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()
        _reclaim_transient_gpu_memory()
        if job_id in JOBS:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["progress"] = 0.0
            JOBS[job_id]["message"] = str(abort_err) or "已手动停止当前推理任务"
        print(f"[任务控制] 任务 {job_id} 已中断并释放 GPU 显存锁。")
    except Exception as e:
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()
        _reclaim_transient_gpu_memory()
        if job_id in JOBS:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = f"生成出错: {str(e)}"
    finally:
        cleanup_paths = JOBS.get(job_id, {}).pop("_temp_cleanup_paths", None)
        if cleanup_paths:
            for p in cleanup_paths:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
        _JOB_CANCEL_EVENTS.pop(job_id, None)
        touch_activity()
