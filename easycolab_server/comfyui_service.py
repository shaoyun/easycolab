"""EasyColab 服务端模块: comfyui_service."""

from PIL import Image
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

from . import runtime_state as rs
from .media_pipeline import _reclaim_transient_gpu_memory, unload_model_from_vram_core
from .model_catalog import TOYBOX_CORE_MODELS, TOYBOX_IMAGE_MODELS, TOYBOX_REWRITER_MODELS, fast_parallel_hf_download
from .runtime_state import COMFYUI_LOCK, COMFYUI_STATE, COMFY_BACKEND, COMFY_INGRESS, _StripAuthRedirectHandler, _has_rclone_drive_remote, touch_activity

# ==============================================================================
# ComfyUI 云端节点工作台 + Google Drive 自动软链与双向持久化引擎
# ==============================================================================
COMFY_STANDARD_MODEL_SUBDIRS = [
    "checkpoints",
    "diffusion_models",
    "unet",
    "vae",
    "loras",
    "text_encoders",
    "clip",
    "clip_vision",
    "controlnet",
    "upscale_models",
    "latent_upscale_models",
    "audio_encoders",
    "embeddings",
    "style_models",
    "gguf",
    "semantic_bridge",
]

_COMFY_PROTECTED_PIP_PKGS = re.compile(
    r"^\s*(torch|torchvision|torchaudio|xformers)($|[<>=!~;\s\[])",
    re.IGNORECASE,
)

_COMFY_BG_SYNC_STARTED = False

_COMFY_ADOPT_CHECKED = False

_COMFY_STATE_FILE = "/tmp/.easycolab_comfy_state.json"

_COMFY_STATUS_CACHE: Dict[str, Any] = {"ts": 0.0, "counts": (0, 0, 0), "outputs": [], "packs": []}

COMFY_TEMPLATE_RAW_BASE = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/"

# 100% 免审核 (Gated=False) 官方原生节点预置图生视频 / 首尾帧运镜工作流包
COMFY_PRESET_WORKFLOW_PACKS: List[Dict[str, Any]] = [
    {
        "id": "wan22_i2v_flf2v",
        "title": "万相 Wan2.2 14B · 图生视频 + 首尾帧双工作流",
        "badge": "新手首选 · 4步极速",
        "scenario": "含「单图动起来」与「首尾双图丝滑过渡」2 套中文画布，G4 约 25 秒出片，纯官方内置节点零红框报错",
        "approx_size_gb": 35.4,
        "workflows": [
            "01_Wan2.2_极速单图生视频(4步傻瓜版).json",
            "02_Wan2.2_首尾帧丝滑过渡(双图4步极速).json",
        ],
        "models": [
            ("text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6.27),
            ("vae", "wan_2.1_vae.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors", 0.24),
            ("diffusion_models", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 13.31),
            ("diffusion_models", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 13.31),
            ("loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", 1.14),
            ("loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", 1.14),
        ],
    },
    {
        "id": "minimax_h3_i2v_flf2v",
        "title": "MiniMax-H3 Turbo · 图生有声视频 + 首尾帧双工作流",
        "badge": "自带立体声音效 · 8步加速",
        "scenario": "含「单图生有声视频」与「首尾双图有声过渡」2 套极简画布，一次生成高清动态画面 + 同步环境音效/配乐/对白",
        "approx_size_gb": 39.1,
        "workflows": [
            "03_MiniMax-H3_极速图生有声视频(单图带立体声).json",
            "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json",
        ],
        "models": [
            ("vae", "minimax_h3_video_vae_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_int8_convrot.safetensors", 2.62),
            ("vae", "minimax_h3_audio_vae_fp32.safetensors", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", 0.56),
            ("diffusion_models", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", 19.53),
            ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 14.61),
            ("loras", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "lightx2v/Minimax-h3-Turbo", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 1.82),
        ],
    },
    {
        "id": "ltx23_flf2v",
        "title": "LTX-2.3 (22B) · 首尾帧电影级运镜有声工作流",
        "badge": "电影推拉摇移 · 仅 2 个权重",
        "scenario": "专为首尾关键帧大幅度镜头运动优化（如高空俯拍平滑推镜至低角度特写 + 自动合成配乐音轨）",
        "approx_size_gb": 36.3,
        "workflows": [
            "05_LTX-2.3_首尾帧电影运镜(双图极速带音效).json",
        ],
        "models": [
            ("checkpoints", "ltx-2.3-22b-distilled-fp8.safetensors", "Lightricks/LTX-2.3-fp8", "ltx-2.3-22b-distilled-fp8.safetensors", 27.50),
            ("text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors", "Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", 8.80),
        ],
    },
    {
        "id": "hunyuan15_i2v_1080p",
        "title": "混元 HunyuanVideo 1.5 · 单图直出 1080P 超分工作流",
        "badge": "720P + 1080P 双路超清",
        "scenario": "已默认激活潜空间超分管线，传 1 张图一键自动渲染 720P 原片与 1920×1080 超高清成片",
        "approx_size_gb": 43.5,
        "workflows": [
            "06_Hunyuan1.5_单图直出1080P超清视频.json",
        ],
        "models": [
            ("clip_vision", "sigclip_vision_patch14_384.safetensors", "Comfy-Org/sigclip_vision_384", "sigclip_vision_patch14_384.safetensors", 0.80),
            ("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", 8.74),
            ("text_encoders", "byt5_small_glyphxl_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors", 0.41),
            ("latent_upscale_models", "hunyuanvideo15_latent_upsampler_1080p.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/latent_upscale_models/hunyuanvideo15_latent_upsampler_1080p.safetensors", 0.45),
            ("diffusion_models", "hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors", 15.20),
            ("vae", "hunyuanvideo15_vae_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/vae/hunyuanvideo15_vae_fp16.safetensors", 2.35),
            ("diffusion_models", "hunyuanvideo1.5_720p_i2v_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/hunyuanvideo1.5_720p_i2v_fp16.safetensors", 15.51),
        ],
    },
    {
        "id": "minimax_h3_toybox",
        "title": "H3 Toybox 全能创作台 · Ref2VA / 首尾帧 / 延伸 / P-H3 生图",
        "badge": "独立Web创作台 · 共用Comfy引擎",
        "scenario": "复用 Qwen3-VL / H3 VAE 共享底座，补齐 PRISM 主模型、语义桥、SelfLift、延伸气锁与 P-H3 生图套件",
        "approx_size_gb": 54.4,
        "workflows": [
            "07_MiniMax-H3_Toybox标准全功能工作流(WF).json",
        ],
        "models": list(TOYBOX_CORE_MODELS) + list(TOYBOX_REWRITER_MODELS) + list(TOYBOX_IMAGE_MODELS),
    },
]

def _ensure_comfyui_beginner_assets(comfy_root: str, persist_root: str):
    """
    为新手免折腾准备：
    1. 默认将 ComfyUI 界面语言设为简体中文 (Comfy.Locale = zh)；
    2. 生成 1280x720 预设示例首帧/尾帧图片至 input/ 目录并建立软链，即使未传图直接点「运行」也零报错。
    """
    try:
        settings_dir = os.path.join(persist_root, "user", "default")
        os.makedirs(settings_dir, exist_ok=True)
        settings_path = os.path.join(settings_dir, "comfy.settings.json")
        cfg_obj: Dict[str, Any] = {}
        if os.path.isfile(settings_path):
            try:
                cfg_obj = json.load(open(settings_path, "r", encoding="utf-8"))
            except Exception:
                cfg_obj = {}
        if not cfg_obj.get("Comfy.Locale"):
            cfg_obj["Comfy.Locale"] = "zh"
            with open(settings_path, "w", encoding="utf-8") as sf:
                json.dump(cfg_obj, sf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    try:
        from PIL import ImageDraw
        input_persist = os.path.join(persist_root, "input")
        os.makedirs(input_persist, exist_ok=True)
        if os.path.isdir(comfy_root):
            _safe_symlink_dir(input_persist, os.path.join(comfy_root, "input"))

        start_img_p = os.path.join(input_persist, "easycolab_sample_start.png")
        end_img_p = os.path.join(input_persist, "easycolab_sample_end.png")

        if not os.path.isfile(start_img_p):
            w, h = 1280, 720
            im = Image.new("RGB", (w, h), (16, 24, 48))
            dr = ImageDraw.Draw(im)
            for y in range(h):
                t = y / float(h - 1)
                r = int(18 * (1 - t) + 64 * t)
                g = int(28 * (1 - t) + 108 * t)
                b = int(68 * (1 - t) + 168 * t)
                dr.line([(0, y), (w, y)], fill=(r, g, b))
            # 太阳光晕与海平面剪影（首帧：朝阳初升左侧）
            dr.ellipse([260, 210, 440, 390], fill=(255, 206, 126))
            dr.ellipse([285, 235, 415, 365], fill=(255, 240, 196))
            dr.polygon([(0, 520), (240, 430), (520, 495), (860, 410), (1140, 485), (1280, 450), (1280, 720), (0, 720)], fill=(18, 28, 46))
            dr.rectangle([0, 565, w, h], fill=(12, 22, 38))
            im.save(start_img_p, format="PNG")

        if not os.path.isfile(end_img_p):
            w, h = 1280, 720
            im = Image.new("RGB", (w, h), (32, 18, 52))
            dr = ImageDraw.Draw(im)
            for y in range(h):
                t = y / float(h - 1)
                r = int(42 * (1 - t) + 232 * t)
                g = int(24 * (1 - t) + 118 * t)
                b = int(76 * (1 - t) + 88 * t)
                dr.line([(0, y), (w, y)], fill=(r, g, b))
            # 尾帧：晚霞金日移至右侧高空，测试首尾帧运镜与光影平滑过渡
            dr.ellipse([800, 130, 1000, 330], fill=(255, 168, 92))
            dr.ellipse([830, 160, 970, 300], fill=(255, 232, 178))
            dr.polygon([(0, 540), (210, 460), (490, 510), (820, 395), (1090, 470), (1280, 430), (1280, 720), (0, 720)], fill=(24, 16, 36))
            dr.rectangle([0, 575, w, h], fill=(16, 12, 28))
            im.save(end_img_p, format="PNG")
    except Exception as e:
        print(f"[ComfyUI] 生成默认占位参考图提示: {e}")

def _fetch_official_comfy_template(template_filename: str) -> Dict[str, Any]:
    url = COMFY_TEMPLATE_RAW_BASE + template_filename
    req = urllib.request.Request(url, headers={"User-Agent": "EasyColab/2.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _make_zh_guide_note(node_id: int, pos: List[float], title: str, steps_md: str) -> Dict[str, Any]:
    return {
        "id": node_id,
        "type": "MarkdownNote",
        "pos": pos,
        "size": [430, 340],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "title": title,
        "properties": {},
        "widgets_values": [steps_md],
        "widgets_values_named": {"text": steps_md},
        "color": "#1e3a5f",
        "bgcolor": "#13263e",
    }

def _ensure_easycolab_comfy_preset_helper(comfy_root: str, persist_root: str):
    """
    预置轻量前端扩展：每次通过全新 *.trycloudflare.com 域名首次打开 ComfyUI 时，
    由于新域名 localStorage 为空，ComfyUI 默认会加载内置的 Z-Image Turbo 文生图模板，
    并把缺失的模型自动替换成目录里字母序第一个文件（例如把 minimax_h3_audio_vae 塞进图像 VAEDecode 导致维度 24!=32 报错）。
    本扩展会在检测到默认 Z-Image Turbo 占位图时，自动切换为用户已安装的 EasyColab 中文预置工作流。
    """
    if not os.path.isdir(comfy_root):
        return
    try:
        helper_dir = os.path.join(comfy_root, "custom_nodes", "easycolab_preset_helper")
        js_dir = os.path.join(helper_dir, "js")
        os.makedirs(js_dir, exist_ok=True)

        init_py = '''import os
import json
from aiohttp import web
import folder_paths
from server import PromptServer

WEB_DIRECTORY = "./js"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    if getattr(PromptServer, "instance", None) is not None:
        @PromptServer.instance.routes.get("/easycolab/default_workflow")
        async def _easycolab_get_default_workflow(request):
            try:
                get_ud = getattr(folder_paths, "get_user_directory", None)
                user_dir = get_ud() if callable(get_ud) else os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "user",
                )
                default_dir = os.path.join(user_dir, "default")
                wf_dir = os.path.join(default_dir, "workflows")
                target_fn = ""
                ptr_path = os.path.join(default_dir, "easycolab_active_preset.json")
                if os.path.isfile(ptr_path):
                    try:
                        with open(ptr_path, "r", encoding="utf-8") as pf:
                            ptr = json.load(pf)
                        cand = str(ptr.get("workflow_filename") or "").strip()
                        if cand and os.path.isfile(os.path.join(wf_dir, cand)):
                            target_fn = cand
                    except Exception:
                        pass
                if not target_fn and os.path.isdir(wf_dir):
                    cands = sorted(
                        f for f in os.listdir(wf_dir)
                        if f.lower().endswith(".json") and not f.startswith(".")
                    )
                    if cands:
                        target_fn = cands[0]
                if target_fn:
                    wf_path = os.path.join(wf_dir, target_fn)
                    with open(wf_path, "r", encoding="utf-8") as wf:
                        wf_obj = json.load(wf)
                    return web.json_response({"ok": True, "filename": target_fn, "workflow": wf_obj})
            except Exception as exc:
                return web.json_response({"ok": False, "error": str(exc)})
            return web.json_response({"ok": False})
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
'''

        autoload_js = '''import { app } from "../../scripts/app.js";

function isBuiltInZImageStarterGraph(graph) {
    if (!graph || !Array.isArray(graph._nodes)) return false;
    if (graph._nodes.length === 0) return false;
    return graph._nodes.some((n) => {
        const models = n?.properties?.models;
        return Array.isArray(models) && models.some((m) => String(m?.url || "").includes("z_image_turbo"));
    });
}

app.registerExtension({
    name: "EasyColab.PresetWorkflowAutoLoader",
    async setup() {
        if (sessionStorage.getItem("easycolab_preset_autoloaded_v1") === "1") {
            return;
        }
        const tryAutoLoad = async (allowEmpty) => {
            if (sessionStorage.getItem("easycolab_preset_autoloaded_v1") === "1") return true;
            const g = app.graph;
            const isStarter = isBuiltInZImageStarterGraph(g);
            const isEmpty = allowEmpty && g && Array.isArray(g._nodes) && g._nodes.length === 0;
            if (!isStarter && !isEmpty) return false;

            sessionStorage.setItem("easycolab_preset_autoloaded_v1", "1");
            try {
                const resp = await fetch("/easycolab/default_workflow", { cache: "no-store" });
                if (!resp.ok) return true;
                const data = await resp.json();
                if (data && data.ok && data.workflow) {
                    const wfTitle = String(data.filename || "EasyColab_Preset").replace(/\\.json$/i, "");
                    await app.loadGraphData(data.workflow, true, true, wfTitle);
                    console.log("[EasyColab] 已自动载入预置中文工作流:", data.filename);
                }
            } catch (err) {
                console.warn("[EasyColab] 自动载入预置工作流跳过:", err);
            }
            return true;
        };

        setTimeout(() => { tryAutoLoad(false); }, 350);
        setTimeout(() => { tryAutoLoad(true); }, 1100);
    },
});
'''
        with open(os.path.join(helper_dir, "__init__.py"), "w", encoding="utf-8") as f_py:
            f_py.write(init_py)
        with open(os.path.join(js_dir, "autoload.js"), "w", encoding="utf-8") as f_js:
            f_js.write(autoload_js)
    except Exception as e:
        print(f"[ComfyUI] 写入 EasyColab 默认工作流引导扩展提示: {e}")

def _prune_workflow_dangling_links(wf_data: Dict[str, Any]):
    """确保每个节点的 inputs/outputs 链接表与顶层 links 完全一致，彻底杜绝 LiteGraph 断链警告"""
    node_ids = {n.get("id") for n in wf_data.get("nodes", [])}
    valid_links = [
        lk for lk in wf_data.get("links", [])
        if isinstance(lk, list) and len(lk) >= 5 and lk[1] in node_ids and lk[3] in node_ids
    ]
    wf_data["links"] = valid_links
    valid_lids = {lk[0] for lk in valid_links}
    for n in wf_data.get("nodes", []):
        for inp in n.get("inputs", []) or []:
            if inp.get("link") is not None and inp.get("link") not in valid_lids:
                inp["link"] = None
        for out in n.get("outputs", []) or []:
            if isinstance(out.get("links"), list):
                out["links"] = [lid for lid in out["links"] if lid in valid_lids]

def _build_and_install_comfy_preset_workflows(
    persist_root: str,
    pack_ids: Optional[List[str]] = None,
) -> List[str]:
    """拉取官方最新模板并转换为全中文标题、内置三步傻瓜注释、G4 高清参数与零缺图报错的开箱即用工作流"""
    wf_dir = os.path.join(persist_root, "user", "default", "workflows")
    os.makedirs(wf_dir, exist_ok=True)
    target_set = set(pack_ids) if pack_ids else {p["id"] for p in COMFY_PRESET_WORKFLOW_PACKS}
    installed_files: List[str] = []

    # 1. 万相 Wan2.2 14B (单图 I2V + 首尾帧 FLF2V 4步极速版)
    if "wan22_i2v_flf2v" in target_set:
        try:
            d_i2v = _fetch_official_comfy_template("video_wan2_2_14B_i2v.json")
            kept_nodes = []
            for n in d_i2v.get("nodes", []):
                ntype = n.get("type")
                if ntype == "MarkdownNote":
                    continue
                if n.get("id") == 97 and ntype == "LoadImage":
                    n["title"] = "① 点击此处选择起始图片 (Choose file to upload)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif n.get("id") == 129:
                    n["title"] = "② 填写画面动作与镜头描述 (G4 已预设 1280×720 · 4步极速)"
                    wv = list(n.get("widgets_values") or [])
                    if len(wv) >= 4:
                        wv[0] = "镜头缓慢向前推进，晨曦阳光洒在海面与远山上，云层轻柔流动，水面泛起细腻金光涟漪，电影级光影质感。"
                        wv[1] = 1280
                        wv[2] = 720
                    n["widgets_values"] = wv
                    n.pop("widgets_values_named", None)
                elif ntype == "SaveVideo":
                    n["title"] = "③ 生成结果预览与自动保存 (右键可下载视频)"
                    n["widgets_values"] = ["video/Wan2.2_单图生视频", "auto", "auto"]
                    n.pop("widgets_values_named", None)
                kept_nodes.append(n)
            kept_nodes.append(_make_zh_guide_note(
                501,
                [-520, -420],
                "【EasyColab 三步傻瓜出片指南 · 单图生视频】",
                "## 三步直接出片（无需连线！）\n\n"
                "1. **传图**：点击右侧 **「① 点击此处选择起始图片」** 节点底部的上传按钮，选一张你的图片。\n"
                "2. **写动作**：在中间 **「② 填写画面动作与镜头描述」** 第一个文本框里用中文或英文写希望画面怎么动（如：镜头推进、人物转头微笑、发丝随风飘动）。\n"
                "3. **点运行**：点击网页右上角蓝色的 **「运行 (Run)」** 按钮，G4 显卡约 **25 秒** 即可在右侧直接预览成片！\n\n"
                "> 提示：所有生成的视频都会自动同步至你的 Google Drive，也可回到 EasyColab 客户端一键拉取到 Mac 本地。",
            ))
            d_i2v["nodes"] = kept_nodes
            d_i2v["last_node_id"] = max(int(d_i2v.get("last_node_id") or 200), 505)
            d_i2v.setdefault("extra", {})["ds"] = {"scale": 0.72, "offset": [580, 560]}
            _prune_workflow_dangling_links(d_i2v)
            out_1 = os.path.join(wf_dir, "01_Wan2.2_极速单图生视频(4步傻瓜版).json")
            with open(out_1, "w", encoding="utf-8") as wf:
                json.dump(d_i2v, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_1))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] Wan2.2 I2V 模板处理异常: {e}")

        try:
            d_flf = _fetch_official_comfy_template("video_wan2_2_14B_flf2v.json")
            # 仅提取官方模板中 4-Step LightX2V LoRA 极速分支连通分量 (原 mode=4)，彻底剔除慢速重复节点并自动激活 (mode=0)
            active_ids = {n["id"] for n in d_flf.get("nodes", []) if n.get("mode") == 4}
            flf_nodes = []
            for n in d_flf.get("nodes", []):
                nid = n.get("id")
                if nid not in active_ids or n.get("type") == "Note":
                    continue
                n["mode"] = 0
                if isinstance(n.get("pos"), list) and len(n["pos"]) >= 2:
                    n["pos"] = [n["pos"][0], n["pos"][1] - 1600]
                if nid == 68 and n.get("type") == "LoadImage":
                    # link 157 -> node 67 slot 5 (start_image)
                    n["title"] = "① 点击上传起始首帧图片 (Start Frame)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 62 and n.get("type") == "LoadImage":
                    # link 158 -> node 67 slot 6 (end_image)
                    n["title"] = "② 点击上传结束尾帧图片 (End Frame)"
                    n["widgets_values"] = ["easycolab_sample_end.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 6 and n.get("type") == "CLIPTextEncode":
                    n["title"] = "③ 输入首尾帧过渡动作与运镜描述 (正向提示词)"
                    n["widgets_values"] = [
                        "清晨海面的朝阳缓缓升起并化作绚丽的傍晚漫天晚霞，镜头平稳向右推进，云层与海浪自然流动过渡，光影由冷蓝渐变为温暖金红，画面连贯丝滑。"
                    ]
                    n.pop("widgets_values_named", None)
                elif nid == 7 and n.get("type") == "CLIPTextEncode":
                    n["title"] = "反向提示词 (保持默认即可)"
                elif nid == 67 and n.get("type") == "WanFirstLastFrameToVideo":
                    n["title"] = "④ 视频尺寸与总帧数 (81帧≈5秒 · 1280×720)"
                    n["widgets_values"] = [1280, 720, 81, 1]
                    n.pop("widgets_values_named", None)
                elif nid == 61 and n.get("type") == "SaveVideo":
                    n["title"] = "⑤ 首尾帧过渡成片预览与自动保存"
                    n["widgets_values"] = ["video/Wan2.2_首尾帧过渡", "auto", "auto"]
                    n.pop("widgets_values_named", None)
                flf_nodes.append(n)

            flf_nodes.append(_make_zh_guide_note(
                502,
                [-430, 490],
                "【EasyColab 三步傻瓜出片指南 · 首尾帧过渡】",
                "## 首尾两张图丝滑变形/运镜（4步极速已开启）\n\n"
                "1. **传首尾两张图**：在右侧 **「① 起始首帧图片」** 和 **「② 结束尾帧图片」** 分别上传开头和结尾画面。\n"
                "2. **写过渡过程**：在 **「③ 输入首尾帧过渡动作与运镜描述」** 写清楚从图 1 怎么变化/运镜到图 2。\n"
                "3. **点右上角运行**：无需按任何快捷键，4 步加速 LoRA 已自动配好，约 **25~35 秒** 直接生成丝滑过渡视频！",
            ))
            d_flf["nodes"] = flf_nodes
            d_flf["links"] = [
                lk for lk in d_flf.get("links", [])
                if lk[1] in active_ids and lk[3] in active_ids
            ]
            zh_group_titles = {
                1: "自动模型加载区 (无需改动)",
                2: "第一步：上传起始首帧图片",
                5: "第二步：上传结束尾帧图片",
                3: "第三步：输入画面过渡描述",
                4: "第四步：视频分辨率与帧数 (默认 1280×720)",
            }
            kept_groups = []
            for g in d_flf.get("groups", []):
                gid = g.get("id")
                if gid in zh_group_titles:
                    b = list(g.get("bounding", [0, 0, 400, 400]))
                    if len(b) >= 2:
                        b[1] = b[1] - 1600
                    g["bounding"] = b
                    g["title"] = zh_group_titles[gid]
                    kept_groups.append(g)
            d_flf["groups"] = kept_groups
            d_flf["last_node_id"] = max(int(d_flf.get("last_node_id") or 200), 505)
            d_flf.setdefault("extra", {})["ds"] = {"scale": 0.62, "offset": [480, 120]}
            _prune_workflow_dangling_links(d_flf)
            out_2 = os.path.join(wf_dir, "02_Wan2.2_首尾帧丝滑过渡(双图4步极速).json")
            with open(out_2, "w", encoding="utf-8") as wf:
                json.dump(d_flf, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_2))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] Wan2.2 FLF2V 模板处理异常: {e}")

    # 2. MiniMax-H3 Turbo (单图有声视频 + 首尾帧有声视频 8步极速双工作流)
    if "minimax_h3_i2v_flf2v" in target_set:
        try:
            import copy
            d_h3_raw = _fetch_official_comfy_template("video_minimax_h3_i2v.json")

            def _customize_h3(is_dual_frame: bool) -> Dict[str, Any]:
                d_h3 = copy.deepcopy(d_h3_raw)
                nodes_out = []
                for n in d_h3.get("nodes", []):
                    nid = n.get("id")
                    ntype = n.get("type")
                    if ntype == "MarkdownNote" or nid in (119, 120):
                        continue
                    if nid == 114 and ntype == "LoadImage":
                        n["title"] = "① 点击上传起始首帧图片 (First Frame)"
                        n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                        n["widgets_values_named"] = {"image": "easycolab_sample_start.png", "upload": "image"}
                        n["pos"] = [-1320, 4820]
                    elif nid == 115 and ntype == "ResolutionSelector":
                        n["title"] = "画幅比例与清晰度 (0.6≈1056×608 · 0.98=1344×768)"
                        n["widgets_values"] = ["16:9 (Widescreen)", 0.6, 32]
                        n["widgets_values_named"] = {
                            "aspect_ratio": "16:9 (Widescreen)",
                            "megapixels": 0.6,
                            "multiple": 32,
                        }
                        n["pos"] = [-1320, 5660] if is_dual_frame else [-1320, 5260]
                    elif nid == 105:
                        n["title"] = (
                            "③ 填写过渡动作 + 音效配乐描述 (8步 Turbo 极速已开启)"
                            if is_dual_frame
                            else "② 填写画面动作 + 音效配乐描述 (8步 Turbo 极速已开启)"
                        )
                        zh_prompt = (
                            "电影级航拍镜头：画面从清晨宁静的蔚蓝海岸平稳推进并过渡至漫天金红晚霞的黄昏海湾，海浪轻拍礁石，光影自然流转。\n"
                            "Audio: 轻柔的海浪拍岸声、温暖的海风吹拂声，伴随舒缓治愈的钢琴与大提琴电影配乐。"
                            if is_dual_frame
                            else "电影级运镜缓慢向前推进，清晨阳光穿透薄雾洒在海面，金光波光粼粼，飞鸟掠过天际。\n"
                                 "Audio: 清脆的海浪声、微风声与远处海鸥鸣叫，搭配温柔空灵的钢琴环境配乐。"
                        )
                        wv = list(n.get("widgets_values") or [])
                        if len(wv) >= 10:
                            wv[0] = zh_prompt
                            wv[9] = True  # 默认启用 8-step Turbo LoRA 极速出片
                        n["widgets_values"] = wv
                        if isinstance(n.get("widgets_values_named"), dict):
                            n["widgets_values_named"]["prompt"] = zh_prompt
                            n["widgets_values_named"]["value"] = True
                    elif nid == 92 and ntype == "SaveVideo":
                        n["title"] = "④ 有声视频预览与自动保存 (请打开播放器音量!)"
                        vid_prefix = "video/MiniMax_H3_首尾帧有声" if is_dual_frame else "video/MiniMax_H3_图生有声"
                        n["widgets_values"] = [vid_prefix, "auto", "auto"]
                        n["widgets_values_named"] = {
                            "filename_prefix": vid_prefix,
                            "format": "auto",
                            "codec": "auto",
                        }
                    nodes_out.append(n)

                links_out = [lk for lk in d_h3.get("links", []) if lk[0] != 228]
                if is_dual_frame:
                    end_node_id = 201
                    new_link_id = 250
                    nodes_out.append({
                        "id": end_node_id,
                        "type": "LoadImage",
                        "pos": [-1320, 5240],
                        "size": [315, 314],
                        "flags": {},
                        "order": 1,
                        "mode": 0,
                        "inputs": [],
                        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [new_link_id]}, {"name": "MASK", "type": "MASK", "links": None}],
                        "title": "② 点击上传结束尾帧图片 (Last Frame)",
                        "properties": {"Node name for S&R": "LoadImage"},
                        "widgets_values": ["easycolab_sample_end.png", "image"],
                        "widgets_values_named": {"image": "easycolab_sample_end.png", "upload": "image"},
                    })
                    for n in nodes_out:
                        if n.get("id") == 105:
                            for inp in n.get("inputs", []):
                                if inp.get("name") == "last_frame":
                                    inp["link"] = new_link_id
                    links_out.append([new_link_id, end_node_id, 0, 105, 1, "IMAGE"])
                    d_h3["last_link_id"] = max(int(d_h3.get("last_link_id") or 230), new_link_id + 1)

                nodes_out.append(_make_zh_guide_note(
                    503,
                    [-1800, 4820],
                    "【EasyColab 三步出片指南 · MiniMax-H3 原生有声视频】",
                    "## 画面 + 立体声音效/配乐一次生成！\n\n"
                    + (
                        "1. **传首尾两张图**：在 **「① 起始首帧」** 与 **「② 结束尾帧」** 上传起止图片。\n"
                        "2. **写画面 + 声音**：在中间 **「③ 过渡动作 + 音效配乐描述」** 写清画面变化，并在 `Audio:` 后写想要的环境音/台词/音乐。\n"
                        "3. **点运行**：已默认开启 **8 步 Turbo 加速**，生成后记得打开右侧播放器喇叭图标听立体声！"
                        if is_dual_frame
                        else "1. **传首帧图**：在 **「① 起始首帧」** 点击上传你的参考图。\n"
                             "2. **写画面 + 声音**：在中间 **「② 画面动作 + 音效配乐描述」** 描述动作，并在 `Audio:` 后写音效与配乐风格。\n"
                             "3. **点运行**：已默认开启 **8 步 Turbo 加速**，生成后打开右侧视频喇叭即可试听同步立体声！"
                    ),
                ))
                d_h3["nodes"] = nodes_out
                d_h3["links"] = links_out
                d_h3["last_node_id"] = max(int(d_h3.get("last_node_id") or 210), 505)
                d_h3.setdefault("extra", {})["ds"] = {"scale": 0.68, "offset": [1860, -4740]}
                _prune_workflow_dangling_links(d_h3)
                return d_h3

            out_3 = os.path.join(wf_dir, "03_MiniMax-H3_极速图生有声视频(单图带立体声).json")
            with open(out_3, "w", encoding="utf-8") as wf:
                json.dump(_customize_h3(is_dual_frame=False), wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_3))

            out_4 = os.path.join(wf_dir, "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json")
            with open(out_4, "w", encoding="utf-8") as wf:
                json.dump(_customize_h3(is_dual_frame=True), wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_4))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] MiniMax-H3 模板处理异常: {e}")

    # 3. LTX-2.3 22B Distilled (首尾帧电影运镜 + 音轨)
    if "ltx23_flf2v" in target_set:
        try:
            d_ltx = _fetch_official_comfy_template("video_ltx2_3_flf2v.json")
            ltx_nodes = []
            for n in d_ltx.get("nodes", []):
                nid = n.get("id")
                ntype = n.get("type")
                if ntype == "MarkdownNote":
                    continue
                if nid == 31 and ntype == "LoadImage":
                    n["title"] = "① 点击上传起始首帧图片 (First Frame)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 39 and ntype == "LoadImage":
                    n["title"] = "② 点击上传结束尾帧图片 (Last Frame)"
                    n["widgets_values"] = ["easycolab_sample_end.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 129:
                    n["title"] = "③ 首尾帧电影运镜与音轨参数 (LTX-2.3 22B Distilled)"
                elif nid == 68 and ntype == "SaveVideo":
                    n["title"] = "④ 电影运镜成片预览与自动保存"
                    n["widgets_values"] = ["video/LTX2.3_首尾帧运镜", "auto", "auto"]
                    n.pop("widgets_values_named", None)
                ltx_nodes.append(n)

            ltx_nodes.append(_make_zh_guide_note(
                504,
                [-490, 2250],
                "【EasyColab 三步傻瓜出片指南 · LTX-2.3 首尾帧运镜】",
                "## 电影级大幅度推拉摇移运镜（自带音效配乐）\n\n"
                "1. **传首尾帧**：在右侧 **「① 起始首帧」** 与 **「② 结束尾帧」** 上传两张不同机位或状态的画面。\n"
                "2. **写运镜轨迹**：在中间 **「③ 首尾帧电影运镜与音轨参数」** 文本框描述镜头怎么移动（建议英文描述运镜 + `Music:` 描述音乐）。\n"
                "3. **点运行**：采用 Distilled 极速蒸馏权重，一键直出高帧率带音轨视频！",
            ))
            d_ltx["nodes"] = ltx_nodes
            d_ltx["last_node_id"] = max(int(d_ltx.get("last_node_id") or 200), 506)
            d_ltx.setdefault("extra", {})["ds"] = {"scale": 0.66, "offset": [560, -2160]}
            _prune_workflow_dangling_links(d_ltx)
            out_5 = os.path.join(wf_dir, "05_LTX-2.3_首尾帧电影运镜(双图极速带音效).json")
            with open(out_5, "w", encoding="utf-8") as wf:
                json.dump(d_ltx, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_5))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] LTX-2.3 FLF2V 模板处理异常: {e}")

    # 4. 混元 HunyuanVideo 1.5 (单图 720P -> 1080P 自动潜空间超分)
    if "hunyuan15_i2v_1080p" in target_set:
        try:
            d_hy = _fetch_official_comfy_template("video_hunyuan_video_1.5_720p_i2v.json")
            sr_enable_ids = {108, 111, 109, 113, 114, 117, 118, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141}
            drop_ids = {104, 120, 112, 121, 122, 123, 124, 136}
            hy_nodes = []
            for n in d_hy.get("nodes", []):
                nid = n.get("id")
                if nid in drop_ids:
                    continue
                if nid in sr_enable_ids:
                    n["mode"] = 0
                if nid == 80 and n.get("type") == "LoadImage":
                    n["title"] = "① 点击上传起始首帧图片 (720P → 1080P 超分)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 44 and n.get("type") == "CLIPTextEncode":
                    n["title"] = "② 输入画面动作与光影描述 (中英文皆可)"
                    n["widgets_values"] = [
                        "清晨阳光洒在壮阔的海岸山脉上，镜头平稳向前推进，海面波光粼粼，云雾在山谷间缓慢流动，极致写实电影摄影质感。"
                    ]
                    n.pop("widgets_values_named", None)
                elif nid in (127, 131) and n.get("type") == "RandomNoise":
                    wv = list(n.get("widgets_values") or [42, "randomize"])
                    if len(wv) >= 2:
                        wv[1] = "randomize"
                    n["widgets_values"] = wv
                    n.pop("widgets_values_named", None)
                elif nid == 102 and n.get("type") == "SaveVideo":
                    n["title"] = "③ 720P 基础成片预览"
                elif nid == 118 and n.get("type") == "SaveVideo":
                    n["title"] = "④ 1920×1080 超高清成片预览与保存"
                hy_nodes.append(n)

            hy_nodes.append(_make_zh_guide_note(
                506,
                [-1120, 480],
                "【EasyColab 三步傻瓜出片指南 · 混元 1.5 直出 1080P】",
                "## 单图直出 1920×1080 超高清视频（超分管线已激活）\n\n"
                "1. **传图片**：在右侧 **「① 点击上传起始首帧图片」** 选择你的原图。\n"
                "2. **改提示词**：在上方 **「② 输入画面动作与光影描述」** 输入动态描述。\n"
                "3. **点运行**：无需按 `Ctrl+B`，工作流会先生成 720P 动态原片，再自动接力潜空间超分输出 **1920×1080** 超清成片！",
            ))
            d_hy["nodes"] = hy_nodes
            d_hy["links"] = [
                lk for lk in d_hy.get("links", [])
                if lk[1] not in drop_ids and lk[3] not in drop_ids
            ]
            for g in d_hy.get("groups", []):
                if g.get("id") == 3:
                    g["title"] = "第一步：上传起始图片"
                elif g.get("id") == 2:
                    g["title"] = "第二步：输入画面动作描述"
                elif g.get("id") == 8:
                    g["title"] = "1080P 超高清潜空间放大管线 (已默认激活)"
            d_hy["last_node_id"] = max(int(d_hy.get("last_node_id") or 200), 508)
            d_hy.setdefault("extra", {})["ds"] = {"scale": 0.56, "offset": [1180, 120]}
            _prune_workflow_dangling_links(d_hy)
            out_6 = os.path.join(wf_dir, "06_Hunyuan1.5_单图直出1080P超清视频.json")
            with open(out_6, "w", encoding="utf-8") as wf:
                json.dump(d_hy, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_6))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] Hunyuan 1.5 模板处理异常: {e}")

    if "minimax_h3_toybox" in target_set:
        try:
            from .toybox_service import _resolve_toybox_runtime_paths
            paths = _resolve_toybox_runtime_paths()
            wf_src = os.path.join(paths["app_dir"], "WF.json")
            out_7 = os.path.join(wf_dir, "07_MiniMax-H3_Toybox标准全功能工作流(WF).json")
            if os.path.isfile(wf_src):
                shutil.copy2(wf_src, out_7)
            else:
                req = urllib.request.Request(
                    "https://raw.githubusercontent.com/dseditor/minimax-h3-colab/main/toybox/WF.json",
                    headers={"User-Agent": "EasyColab/2.0"},
                )
                with urllib.request.urlopen(req, timeout=20.0) as resp:
                    wf_bytes = resp.read()
                if wf_bytes:
                    with open(out_7, "wb") as wf:
                        wf.write(wf_bytes)
            if os.path.isfile(out_7):
                installed_files.append(os.path.basename(out_7))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] H3 Toybox WF.json 复制提示: {e}")

    if installed_files:
        try:
            with open(os.path.join(wf_dir, ".easycolab_wf_schema_v2"), "w", encoding="utf-8") as sf:
                sf.write("v2")
            ptr_path = os.path.join(persist_root, "user", "default", "easycolab_active_preset.json")
            if (pack_ids and len(pack_ids) == 1) or not os.path.isfile(ptr_path):
                with open(ptr_path, "w", encoding="utf-8") as pf:
                    json.dump({"workflow_filename": installed_files[0]}, pf, ensure_ascii=False, indent=2)
        except Exception:
            pass

    _COMFY_STATUS_CACHE["ts"] = 0.0
    return installed_files

def _get_comfy_preset_packs_status(persist_root: str) -> List[Dict[str, Any]]:
    wf_dir = os.path.join(persist_root, "user", "default", "workflows")
    results: List[Dict[str, Any]] = []
    for pack in COMFY_PRESET_WORKFLOW_PACKS:
        models = pack["models"]
        installed_m = 0
        for sub, fname, _, _, _ in models:
            p = os.path.join(persist_root, "models", sub, fname)
            if os.path.isfile(p) and os.path.getsize(p) > 1024 * 1024:
                installed_m += 1
        wfs = pack["workflows"]
        installed_wf = sum(1 for w in wfs if os.path.isfile(os.path.join(wf_dir, w)))
        ready = (installed_m == len(models)) and (installed_wf == len(wfs))
        results.append({
            "id": pack["id"],
            "title": pack["title"],
            "badge": pack["badge"],
            "scenario": pack["scenario"],
            "approx_size_gb": pack["approx_size_gb"],
            "workflows": wfs,
            "installed_models": installed_m,
            "total_models": len(models),
            "workflows_installed": installed_wf == len(wfs),
            "ready": ready,
        })
    return results

def _resolve_comfyui_persistence_paths() -> Tuple[str, str, str, str, bool]:
    comfy_root = "/content/ComfyUI" if os.path.exists("/content") else os.path.join(os.getcwd(), "ComfyUI")
    if os.path.exists("/content/drive/MyDrive"):
        persist_root = "/content/drive/MyDrive/EasyColab_ComfyUI"
        return comfy_root, persist_root, "drive_symlink", "Google Drive 实时软链直连", True
    if _has_rclone_drive_remote():
        persist_root = "/content/drive_mirror/EasyColab_ComfyUI"
        return comfy_root, persist_root, "rclone_drive_sync", "Google Drive 自动软链持久化", True
    persist_root = (
        "/content/EasyColab_ComfyUI_Data"
        if os.path.exists("/content")
        else os.path.join(os.getcwd(), "EasyColab_ComfyUI_Data")
    )
    return comfy_root, persist_root, "local_only", "实例本地高速存储", False

def _safe_symlink_dir(target_dir: str, link_path: str):
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    if os.path.islink(link_path):
        cur_target = os.readlink(link_path)
        if os.path.abspath(cur_target) == os.path.abspath(target_dir):
            return
        os.unlink(link_path)
    elif os.path.isdir(link_path):
        for entry in os.listdir(link_path):
            src_item = os.path.join(link_path, entry)
            dst_item = os.path.join(target_dir, entry)
            if not os.path.exists(dst_item):
                try:
                    shutil.move(src_item, dst_item)
                except Exception:
                    pass
        shutil.rmtree(link_path, ignore_errors=True)
    elif os.path.exists(link_path):
        os.remove(link_path)
    os.symlink(target_dir, link_path)

def _scan_comfyui_counts_and_outputs() -> Tuple[Tuple[int, int, int], List[Dict[str, Any]]]:
    now = time.time()
    if (now - _COMFY_STATUS_CACHE["ts"]) < 6.0:
        return _COMFY_STATUS_CACHE["counts"], list(_COMFY_STATUS_CACHE["outputs"])

    comfy_root, persist_root, _, _, _ = rs.resolve_callable("_resolve_comfyui_persistence_paths", _resolve_comfyui_persistence_paths)()
    cn_count = 0
    cn_base = os.path.join(comfy_root, "custom_nodes")
    if not os.path.isdir(cn_base):
        cn_base = os.path.join(persist_root, "custom_nodes")
    if os.path.isdir(cn_base):
        try:
            for d in os.listdir(cn_base):
                if d.startswith(".") or d == "__pycache__":
                    continue
                if os.path.isdir(os.path.join(cn_base, d)):
                    cn_count += 1
        except Exception:
            pass

    model_exts = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}
    m_count = 0
    m_base = os.path.join(persist_root, "models")
    if os.path.isdir(m_base):
        try:
            for root_d, _, fnames in os.walk(m_base):
                for fn in fnames:
                    if os.path.splitext(fn)[1].lower() in model_exts:
                        m_count += 1
        except Exception:
            pass

    wf_count = 0
    wf_dir = os.path.join(persist_root, "user", "default", "workflows")
    if os.path.isdir(wf_dir):
        try:
            wf_count = sum(1 for fn in os.listdir(wf_dir) if fn.lower().endswith(".json"))
        except Exception:
            pass

    recent_outputs: List[Dict[str, Any]] = []
    out_dir = os.path.join(comfy_root, "output")
    if not os.path.isdir(out_dir):
        out_dir = os.path.join(persist_root, "output")
    valid_media = {".mp4", ".webm", ".mov", ".webp", ".gif", ".png", ".jpg", ".jpeg"}
    if os.path.isdir(out_dir):
        collected = []
        try:
            for root_d, _, fnames in os.walk(out_dir):
                for fn in fnames:
                    if fn.startswith("."):
                        continue
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in valid_media:
                        continue
                    full_p = os.path.join(root_d, fn)
                    try:
                        st = os.stat(full_p)
                        if st.st_size < 512:
                            continue
                        rel_p = os.path.relpath(full_p, out_dir).replace("\\", "/")
                        mtype = "video" if ext in {".mp4", ".webm", ".mov"} else "image"
                        collected.append((st.st_mtime, {
                            "rel_path": rel_p,
                            "filename": fn,
                            "media_type": mtype,
                            "size_mb": round(st.st_size / (1024 ** 2), 2),
                            "mtime": int(st.st_mtime),
                        }))
                    except Exception:
                        pass
            collected.sort(key=lambda x: x[0], reverse=True)
            recent_outputs = [item for _, item in collected[:16]]
        except Exception:
            pass

    _COMFY_STATUS_CACHE["ts"] = now
    _COMFY_STATUS_CACHE["counts"] = (cn_count, m_count, wf_count)
    _COMFY_STATUS_CACHE["outputs"] = recent_outputs
    return (cn_count, m_count, wf_count), recent_outputs

def _try_adopt_running_comfyui():
    """若 app.py 热更新重启前实例上已有在线 ComfyUI 与 cloudflared 隧道，自动无缝接管，不断开用户已打开的画布"""
    global _COMFY_ADOPT_CHECKED, _COMFY_BG_SYNC_STARTED
    if _COMFY_ADOPT_CHECKED or COMFYUI_STATE.get("is_running") or COMFYUI_STATE.get("is_starting"):
        return
    _COMFY_ADOPT_CHECKED = True

    port = int(COMFYUI_STATE.get("local_port") or COMFY_BACKEND)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/system_stats", timeout=1.0) as resp:
            if resp.status != 200:
                return
    except Exception:
        return

    pub_url = ""
    if os.path.isfile(_COMFY_STATE_FILE):
        try:
            st_obj = json.load(open(_COMFY_STATE_FILE, "r", encoding="utf-8"))
            pub_url = str(st_obj.get("public_url") or "").strip()
        except Exception:
            pub_url = ""

    if not pub_url:
        cf_re = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
        for m_port in range(20241, 20247):
            for ep in ("/quicktunnel", "/metrics"):
                try:
                    with opener.open(f"http://127.0.0.1:{m_port}{ep}", timeout=0.6) as mr:
                        body = mr.read().decode("utf-8", errors="ignore")
                    m = cf_re.search(body)
                    if m:
                        pub_url = m.group(0)
                        break
                    m_host = re.search(r'"hostname"\s*:\s*"([a-zA-Z0-9-]+\.trycloudflare\.com)"', body)
                    if m_host:
                        pub_url = "https://" + m_host.group(1)
                        break
                except Exception:
                    pass
            if pub_url:
                break

    if not pub_url:
        try:
            cf_bin = _ensure_cloudflared_binary()
            tunnel_proc = subprocess.Popen(
                [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{COMFY_INGRESS}", "--no-autoupdate"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            COMFYUI_STATE["tunnel_proc"] = tunnel_proc
            cf_re = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
            if tunnel_proc.stdout is not None:
                t_end = time.time() + 12.0
                while time.time() < t_end:
                    ln = tunnel_proc.stdout.readline()
                    if not ln:
                        break
                    m = cf_re.search(ln)
                    if m:
                        pub_url = m.group(0)
                        break
        except Exception:
            pass

    comfy_root, persist_root, _, label, _ = rs.resolve_callable("_resolve_comfyui_persistence_paths", _resolve_comfyui_persistence_paths)()
    _ensure_comfyui_beginner_assets(comfy_root, persist_root)
    COMFYUI_STATE["is_running"] = True
    COMFYUI_STATE["public_url"] = pub_url
    COMFYUI_STATE["progress"] = 1.0
    COMFYUI_STATE["status_message"] = f"已无缝接管在线 ComfyUI 画布（{label}）"
    if pub_url:
        try:
            with open(_COMFY_STATE_FILE, "w", encoding="utf-8") as sf:
                json.dump({"public_url": pub_url, "port": port}, sf)
        except Exception:
            pass
    if not _COMFY_BG_SYNC_STARTED:
        _COMFY_BG_SYNC_STARTED = True
        threading.Thread(target=_comfyui_periodic_drive_sync_loop, daemon=True).start()
    print(f"[ComfyUI] 已自动接管实例上正在运行的 ComfyUI 服务: {pub_url or f'127.0.0.1:{port}'}")

def _poll_local_comfyui_runtime_quiet():
    proc = COMFYUI_STATE.get("comfy_proc")
    if proc is not None and proc.poll() is not None:
        COMFYUI_STATE["is_running"] = False
        COMFYUI_STATE["queue_running"] = 0
        COMFYUI_STATE["queue_pending"] = 0
        if not COMFYUI_STATE["is_starting"]:
            COMFYUI_STATE["status_message"] = f"ComfyUI 进程已退出 (code={proc.returncode})，可点击重新启动"
        COMFYUI_STATE["comfy_proc"] = None
        return

    if not COMFYUI_STATE.get("is_running"):
        _try_adopt_running_comfyui()
        if not COMFYUI_STATE.get("is_running"):
            return

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    port = int(COMFYUI_STATE.get("local_port") or COMFY_BACKEND)
    try:
        with opener.open(f"http://127.0.0.1:{port}/queue", timeout=1.2) as r:
            q_data = json.loads(r.read().decode("utf-8", errors="replace"))
        q_run = len(q_data.get("queue_running") or [])
        q_pend = len(q_data.get("queue_pending") or [])
        COMFYUI_STATE["queue_running"] = q_run
        COMFYUI_STATE["queue_pending"] = q_pend
        if q_run > 0 or q_pend > 0:
            touch_activity()
    except Exception:
        pass

    try:
        with opener.open(f"http://127.0.0.1:{port}/history?max_items=5", timeout=1.2) as r:
            h_data = json.loads(r.read().decode("utf-8", errors="replace"))
        if isinstance(h_data, dict):
            h_sig = hash(tuple(sorted(h_data.keys())))
            if h_sig != COMFYUI_STATE.get("last_history_count", 0) and COMFYUI_STATE.get("last_history_count", 0) != 0:
                touch_activity()
                _COMFY_STATUS_CACHE["ts"] = 0.0
            COMFYUI_STATE["last_history_count"] = h_sig
    except Exception:
        pass

def get_comfyui_status_dict() -> Dict[str, Any]:
    _, persist_root, mode, label, is_drive = rs.resolve_callable("_resolve_comfyui_persistence_paths", _resolve_comfyui_persistence_paths)()
    (cn_count, m_count, wf_count), recent_outputs = _scan_comfyui_counts_and_outputs()
    preset_packs = _get_comfy_preset_packs_status(persist_root)
    COMFYUI_STATE["persistence_mode"] = mode
    COMFYUI_STATE["persistence_label"] = label
    COMFYUI_STATE["drive_persisted"] = is_drive
    COMFYUI_STATE["custom_nodes_count"] = cn_count
    COMFYUI_STATE["models_count"] = m_count
    COMFYUI_STATE["workflows_count"] = wf_count
    return {
        "is_running": bool(COMFYUI_STATE["is_running"]),
        "is_starting": bool(COMFYUI_STATE["is_starting"]),
        "is_syncing_drive": bool(COMFYUI_STATE["is_syncing_drive"]),
        "is_downloading_weight": bool(COMFYUI_STATE["is_downloading_weight"]),
        "installing_pack_id": str(COMFYUI_STATE.get("installing_pack_id") or ""),
        "progress": float(COMFYUI_STATE["progress"]),
        "status_message": str(COMFYUI_STATE["status_message"]),
        "public_url": str(COMFYUI_STATE["public_url"] or ""),
        "local_port": int(COMFYUI_STATE["local_port"]),
        "queue_running": int(COMFYUI_STATE["queue_running"]),
        "queue_pending": int(COMFYUI_STATE["queue_pending"]),
        "persistence_mode": mode,
        "persistence_label": label,
        "drive_persisted": is_drive,
        "persist_root": persist_root,
        "custom_nodes_count": cn_count,
        "models_count": m_count,
        "workflows_count": wf_count,
        "last_sync_ts": int(COMFYUI_STATE["last_sync_ts"]),
        "recent_outputs": recent_outputs,
        "preset_packs": preset_packs,
    }

def free_comfyui_vram_core() -> Tuple[bool, str]:
    if not COMFYUI_STATE.get("is_running"):
        return False, "ComfyUI 当前未运行"
    port = int(COMFYUI_STATE.get("local_port") or COMFY_BACKEND)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(req, timeout=8.0) as _:
            pass
        _reclaim_transient_gpu_memory()
        msg = "已释放 ComfyUI 驻留模型与显存缓存（画布保持在线）"
        COMFYUI_STATE["status_message"] = msg
        print(f"[ComfyUI] {msg}")
        return True, msg
    except Exception as e:
        return False, f"释放 ComfyUI 显存失败: {e}"

def sync_comfyui_to_drive_core(quiet: bool = False) -> Tuple[bool, str]:
    if COMFYUI_STATE["is_syncing_drive"]:
        return True, "正在后台同步 ComfyUI 数据到 Google Drive..."
    comfy_root, persist_root, mode, label, is_drive = rs.resolve_callable("_resolve_comfyui_persistence_paths", _resolve_comfyui_persistence_paths)()
    if not os.path.isdir(comfy_root) and not os.path.isdir(persist_root):
        return False, "ComfyUI 尚未初始化"

    COMFYUI_STATE["is_syncing_drive"] = True
    try:
        os.makedirs(os.path.join(persist_root, "models"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "custom_nodes"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "user"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "output"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "input"), exist_ok=True)

        # 1. 自动接管自定义节点在 models/ 下新建的非软链目录，迁移至持久化目录并建立软链
        models_dir = os.path.join(comfy_root, "models")
        if os.path.isdir(models_dir):
            for entry in os.listdir(models_dir):
                if entry in ("configs", "__pycache__") or entry.startswith("."):
                    continue
                local_sub = os.path.join(models_dir, entry)
                if os.path.isdir(local_sub) and not os.path.islink(local_sub):
                    persist_sub = os.path.join(persist_root, "models", entry)
                    _safe_symlink_dir(persist_sub, local_sub)

        # 2. 将本地高速 NVMe 上的 custom_nodes 增量镜像同步到持久化目录，并记录 Git 源清单
        cn_local = os.path.join(comfy_root, "custom_nodes")
        cn_persist = os.path.join(persist_root, "custom_nodes")
        manifest: Dict[str, Any] = {}
        manifest_path = os.path.join(persist_root, "custom_nodes_manifest.json")
        if os.path.exists(manifest_path):
            try:
                manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
            except Exception:
                manifest = {}

        if os.path.isdir(cn_local):
            for entry in sorted(os.listdir(cn_local)):
                if entry.startswith(".") or entry == "__pycache__":
                    continue
                src_node = os.path.join(cn_local, entry)
                if not os.path.isdir(src_node):
                    continue
                dst_node = os.path.join(cn_persist, entry)
                if shutil.which("rsync"):
                    subprocess.run(
                        ["rsync", "-a", "--delete", "--exclude=__pycache__", f"{src_node}/", f"{dst_node}/"],
                        check=False,
                    )
                else:
                    shutil.copytree(
                        src_node,
                        dst_node,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    )
                git_cfg = os.path.join(src_node, ".git", "config")
                if os.path.exists(git_cfg):
                    try:
                        cfg_txt = open(git_cfg, "r", encoding="utf-8", errors="ignore").read()
                        m_url = re.search(r"url\s*=\s*(\S+)", cfg_txt)
                        if m_url:
                            manifest[entry] = {"git_url": m_url.group(1).strip()}
                    except Exception:
                        pass

            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, ensure_ascii=False, indent=2)

        # 3. 若处于 rclone 云盘镜像模式，通过 Google Cloud 内网 8 并发同步至 Google Drive
        if mode == "rclone_drive_sync" and _has_rclone_drive_remote():
            if not quiet:
                print("[ComfyUI 云盘同步] 正在通过高速通道将自定义节点、模型与工作流同步至 Google Drive...")
            subprocess.run(
                [
                    "rclone", "copy",
                    persist_root,
                    "easycolab_gdrive:EasyColab_Drive/EasyColab_ComfyUI",
                    "--transfers", "8",
                    "--checkers", "16",
                    "--drive-chunk-size", "64M",
                    "--exclude", "**/__pycache__/**",
                    "--exclude", "**/*.part",
                    "--exclude", "**/*.tmp",
                ],
                check=False,
            )

        COMFYUI_STATE["last_sync_ts"] = time.time()
        _COMFY_STATUS_CACHE["ts"] = 0.0
        (cn_count, m_count, wf_count), _ = _scan_comfyui_counts_and_outputs()
        msg = f"已持久化至 {label}（节点 {cn_count} 个 · 权重 {m_count} 个 · 工作流 {wf_count} 个）"
        if not quiet:
            COMFYUI_STATE["status_message"] = msg
            print(f"[ComfyUI 云盘持久化] {msg}")
        return True, msg
    except Exception as e:
        err = f"ComfyUI 云盘同步异常: {e}"
        if not quiet:
            print(f"[ComfyUI] {err}")
        return False, err
    finally:
        COMFYUI_STATE["is_syncing_drive"] = False

def _comfyui_periodic_drive_sync_loop():
    while True:
        time.sleep(60.0)
        if COMFYUI_STATE.get("is_running") and not COMFYUI_STATE.get("is_starting"):
            if COMFYUI_STATE.get("queue_running", 0) == 0:
                rs.resolve_callable("sync_comfyui_to_drive_core", sync_comfyui_to_drive_core)(quiet=True)

def _ensure_comfyui_manager_tunnel_config(comfy_root: str, persist_root: str):
    """允许通过 trycloudflare 公网隧道在 ComfyUI-Manager 内直接安装 Git 节点与模型，不受默认 normal 安全锁拦截"""
    cfg_targets = [
        os.path.join(persist_root, "user", "default", "ComfyUI-Manager", "config.ini"),
        os.path.join(persist_root, "user", "__manager", "config.ini"),
        os.path.join(comfy_root, "custom_nodes", "ComfyUI-Manager", "config.ini"),
    ]
    for cfg_p in cfg_targets:
        try:
            os.makedirs(os.path.dirname(cfg_p), exist_ok=True)
            if os.path.exists(cfg_p):
                txt = open(cfg_p, "r", encoding="utf-8", errors="ignore").read()
                if "security_level" in txt:
                    new_txt = re.sub(r"security_level\s*=\s*\S+", "security_level = weak", txt)
                    if new_txt != txt:
                        with open(cfg_p, "w", encoding="utf-8") as wf:
                            wf.write(new_txt)
                    continue
            with open(cfg_p, "w", encoding="utf-8") as wf:
                wf.write("[default]\nsecurity_level = weak\npreview_method = auto\n")
        except Exception:
            pass

def _install_persisted_custom_node_requirements(comfy_root: str):
    """扫描已从 Google Drive 恢复的所有自定义节点 requirements.txt，过滤掉 torch/xformers 后批量补齐依赖"""
    import hashlib
    cn_dir = os.path.join(comfy_root, "custom_nodes")
    if not os.path.isdir(cn_dir):
        return
    collected_reqs: List[str] = []
    for entry in sorted(os.listdir(cn_dir)):
        req_file = os.path.join(cn_dir, entry, "requirements.txt")
        if not os.path.isfile(req_file):
            continue
        try:
            for raw_line in open(req_file, "r", encoding="utf-8", errors="ignore"):
                line = raw_line.split("#", 1)[0].strip()
                if not line or line.startswith("-"):
                    continue
                if _COMFY_PROTECTED_PIP_PKGS.match(line):
                    continue
                if line not in collected_reqs:
                    collected_reqs.append(line)
        except Exception:
            pass

    if not collected_reqs:
        return
    digest = hashlib.sha256("\n".join(collected_reqs).encode("utf-8")).hexdigest()
    stamp_file = "/tmp/.easycolab_comfy_node_reqs.sha256"
    if os.path.exists(stamp_file):
        try:
            if open(stamp_file, "r").read().strip() == digest:
                return
        except Exception:
            pass

    print(f"[ComfyUI] 正在为已持久化的自定义节点自动恢复 {len(collected_reqs)} 项 Python 依赖...")
    tmp_req = "/tmp/easycolab_comfy_merged_reqs.txt"
    with open(tmp_req, "w", encoding="utf-8") as wf:
        wf.write("\n".join(collected_reqs) + "\n")
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q", "-r", tmp_req])
    if rc == 0:
        with open(stamp_file, "w") as sf:
            sf.write(digest)

def _ensure_cloudflared_binary() -> str:
    cf_bin = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
    if os.path.isfile(cf_bin) and os.access(cf_bin, os.X_OK):
        return cf_bin
    dl_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    os.makedirs(os.path.dirname(cf_bin), exist_ok=True)
    urllib.request.urlretrieve(dl_url, cf_bin)
    os.chmod(cf_bin, 0o755)
    return cf_bin

def _open_cloudflared_tunnel(ingress_port: int, timeout_s: float = 22.5) -> Tuple[subprocess.Popen, str]:
    """为指定本地 ingress 端口拉起 Cloudflare Quick Tunnel 并等待返回公网 HTTPS 地址。"""
    cf_bin = _ensure_cloudflared_binary()
    tunnel_proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{int(ingress_port)}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    cf_url_box: List[str] = [""]
    cf_re = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    def _watch_cf_logs(proc: subprocess.Popen):
        try:
            if proc.stdout is None:
                return
            for raw_ln in proc.stdout:
                m = cf_re.search(raw_ln)
                if m and not cf_url_box[0]:
                    cf_url_box[0] = m.group(0)
        except Exception:
            pass

    threading.Thread(target=_watch_cf_logs, args=(tunnel_proc,), daemon=True).start()
    deadline = time.time() + max(1.0, float(timeout_s))
    while time.time() < deadline:
        if cf_url_box[0]:
            break
        if tunnel_proc.poll() is not None:
            break
        time.sleep(0.5)
    return tunnel_proc, cf_url_box[0]

def stop_comfyui_core(sync_before_stop: bool = True) -> Tuple[bool, str]:
    with COMFYUI_LOCK:
        if sync_before_stop and (COMFYUI_STATE.get("is_running") or os.path.isdir("/content/ComfyUI")):
            rs.resolve_callable("sync_comfyui_to_drive_core", sync_comfyui_to_drive_core)(quiet=True)
        for key in ("tunnel_proc", "comfy_proc"):
            proc = COMFYUI_STATE.get(key)
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                COMFYUI_STATE[key] = None
        subprocess.run(["pkill", "-f", "ComfyUI/main.py"], check=False)
        subprocess.run(["pkill", "-f", "cloudflared.*8188"], check=False)
        if os.path.exists(_COMFY_STATE_FILE):
            try:
                os.remove(_COMFY_STATE_FILE)
            except Exception:
                pass
        _reclaim_transient_gpu_memory()
        COMFYUI_STATE["is_running"] = False
        COMFYUI_STATE["is_starting"] = False
        COMFYUI_STATE["public_url"] = ""
        COMFYUI_STATE["queue_running"] = 0
        COMFYUI_STATE["queue_pending"] = 0
        COMFYUI_STATE["progress"] = 0.0
        COMFYUI_STATE["status_message"] = "ComfyUI 已停止（节点、模型与工作流已持久化保存至 Google Drive）"
        print("[ComfyUI] 服务与公网隧道已停止，GPU 显存已回收。")
        return True, COMFYUI_STATE["status_message"]

def start_comfyui_core(unload_existing: bool = False, force_restart: bool = False) -> Tuple[bool, str]:
    global _COMFY_BG_SYNC_STARTED
    touch_activity()
    _try_adopt_running_comfyui()
    if COMFYUI_STATE["is_starting"]:
        return True, "ComfyUI 正在部署启动中，请稍候..."
    if COMFYUI_STATE["is_running"] and COMFYUI_STATE.get("public_url") and not force_restart:
        return True, f"ComfyUI 已在运行: {COMFYUI_STATE['public_url']}"

    if force_restart and COMFYUI_STATE["is_running"]:
        rs.resolve_callable("stop_comfyui_core", stop_comfyui_core)(sync_before_stop=True)

    COMFYUI_STATE["is_starting"] = True
    COMFYUI_STATE["progress"] = 0.06
    COMFYUI_STATE["status_message"] = "正在初始化 ComfyUI 环境与 Google Drive 持久化目录..."

    try:
        if unload_existing:
            COMFYUI_STATE["status_message"] = "正在释放现有常驻模型显存，为 ComfyUI 腾出完整 96GB 空间..."
            rs.resolve_callable("unload_model_from_vram_core", unload_model_from_vram_core)("all")

        comfy_root, persist_root, mode, label, is_drive = rs.resolve_callable("_resolve_comfyui_persistence_paths", _resolve_comfyui_persistence_paths)()

        # 1. 若为 rclone 云盘同步模式且本实例首次拉起，先从 Google Drive 高速拉取已持久化的节点、模型与工作流
        pull_flag = "/tmp/.easycolab_comfy_drive_pulled"
        if mode == "rclone_drive_sync" and _has_rclone_drive_remote() and not os.path.exists(pull_flag):
            COMFYUI_STATE["progress"] = 0.14
            COMFYUI_STATE["status_message"] = "正在从 Google Drive 恢复历史持久化的 ComfyUI 节点、模型与工作流..."
            print("[ComfyUI] 正在从 Google Drive 同步已保存的节点、模型与工作流...")
            os.makedirs(persist_root, exist_ok=True)
            subprocess.run(
                [
                    "rclone", "copy",
                    "easycolab_gdrive:EasyColab_Drive/EasyColab_ComfyUI",
                    persist_root,
                    "--transfers", "8",
                    "--checkers", "16",
                    "--drive-chunk-size", "64M",
                ],
                check=False,
            )
            with open(pull_flag, "w") as pf:
                pf.write("1")

        # 2. 克隆或检查 ComfyUI 主程序 (置于本地高速 NVMe，保证秒级导入)
        if not os.path.exists(os.path.join(comfy_root, "main.py")):
            COMFYUI_STATE["progress"] = 0.24
            COMFYUI_STATE["status_message"] = "正在克隆官方最新 ComfyUI 核心引擎..."
            print("[ComfyUI] 正在克隆官方最新 ComfyUI 仓库...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", comfy_root],
                check=True,
            )
            COMFYUI_STATE["progress"] = 0.36
            COMFYUI_STATE["status_message"] = "正在安装 ComfyUI 核心依赖（保留当前 G4 Blackwell CUDA PyTorch）..."
            raw_reqs = open(os.path.join(comfy_root, "requirements.txt"), "r", encoding="utf-8").readlines()
            safe_reqs = [
                ln.strip() for ln in raw_reqs
                if ln.strip() and not ln.strip().startswith("#") and not _COMFY_PROTECTED_PIP_PKGS.match(ln)
            ]
            tmp_core_req = "/tmp/comfy_core_safe_reqs.txt"
            with open(tmp_core_req, "w", encoding="utf-8") as wf:
                wf.write("\n".join(safe_reqs) + "\n")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", tmp_core_req], check=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "gguf", "opencv-python-headless", "imageio-ffmpeg", "gitpython", "toml"],
                check=False,
            )

        # 3. 建立 Google Drive 软链体系：models/* 各子目录、user 工作流配置、output 成片输出全部自动软链至持久化根目录
        COMFYUI_STATE["progress"] = 0.48
        COMFYUI_STATE["status_message"] = f"正在建立 {label} 目录软链 (models / workflows / output)..."
        os.makedirs(os.path.join(persist_root, "models"), exist_ok=True)
        all_model_subdirs = set(COMFY_STANDARD_MODEL_SUBDIRS)
        for existing_sub in os.listdir(os.path.join(persist_root, "models")):
            if not existing_sub.startswith(".") and os.path.isdir(os.path.join(persist_root, "models", existing_sub)):
                all_model_subdirs.add(existing_sub)
        for sub in sorted(all_model_subdirs):
            _safe_symlink_dir(
                os.path.join(persist_root, "models", sub),
                os.path.join(comfy_root, "models", sub),
            )
        _safe_symlink_dir(os.path.join(persist_root, "user"), os.path.join(comfy_root, "user"))
        _safe_symlink_dir(os.path.join(persist_root, "output"), os.path.join(comfy_root, "output"))
        _ensure_comfyui_beginner_assets(comfy_root, persist_root)

        # 若云盘中已有任何预置包的全部权重，但缺少中文工作流文件或工作流版本尚未升级至 v2 (修复 widgets_values_named 兼容)，自动重新生成
        wf_dir_chk = os.path.join(persist_root, "user", "default", "workflows")
        schema_v2_ok = os.path.isfile(os.path.join(wf_dir_chk, ".easycolab_wf_schema_v2"))
        for pk in COMFY_PRESET_WORKFLOW_PACKS:
            if all(
                os.path.isfile(os.path.join(persist_root, "models", sub, fn))
                for sub, fn, _, _, _ in pk["models"]
            ):
                if (not schema_v2_ok) or (not all(os.path.isfile(os.path.join(wf_dir_chk, w)) for w in pk["workflows"])):
                    _build_and_install_comfy_preset_workflows(persist_root, [pk["id"]])

        # 4. 恢复已持久化的自定义节点至本地高速 NVMe，并预装 ComfyUI-Manager 与 VideoHelperSuite 视频核心节点
        COMFYUI_STATE["progress"] = 0.58
        COMFYUI_STATE["status_message"] = "正在恢复已持久化的自定义节点并配置 ComfyUI-Manager..."
        cn_local = os.path.join(comfy_root, "custom_nodes")
        cn_persist = os.path.join(persist_root, "custom_nodes")
        os.makedirs(cn_local, exist_ok=True)
        os.makedirs(cn_persist, exist_ok=True)

        for entry in sorted(os.listdir(cn_persist)):
            if entry.startswith(".") or entry in ("__pycache__", "easycolab_preset_helper"):
                continue
            src_p = os.path.join(cn_persist, entry)
            dst_p = os.path.join(cn_local, entry)
            if os.path.isdir(src_p) and not os.path.exists(dst_p):
                if shutil.which("rsync"):
                    subprocess.run(["rsync", "-a", f"{src_p}/", f"{dst_p}/"], check=False)
                else:
                    shutil.copytree(src_p, dst_p, dirs_exist_ok=True)

        # 若 manifest 中记录了节点但目录缺失，自动按 Git URL 补齐
        manifest_path = os.path.join(persist_root, "custom_nodes_manifest.json")
        if os.path.exists(manifest_path):
            try:
                manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
                for node_name, meta in manifest.items():
                    git_url = (meta or {}).get("git_url", "")
                    dst_p = os.path.join(cn_local, node_name)
                    if git_url and not os.path.exists(dst_p):
                        subprocess.run(["git", "clone", "--depth", "1", git_url, dst_p], check=False)
            except Exception:
                pass

        default_nodes = [
            ("ComfyUI-Manager", "https://github.com/ltdrdata/ComfyUI-Manager.git"),
            ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"),
        ]
        for node_dir_name, repo_url in default_nodes:
            target_node_dir = os.path.join(cn_local, node_dir_name)
            if not os.path.exists(target_node_dir):
                print(f"[ComfyUI] 正在预装核心插件 {node_dir_name}...")
                subprocess.run(["git", "clone", "--depth", "1", repo_url, target_node_dir], check=False)

        _ensure_easycolab_comfy_preset_helper(comfy_root, persist_root)
        _ensure_comfyui_manager_tunnel_config(comfy_root, persist_root)

        COMFYUI_STATE["progress"] = 0.70
        COMFYUI_STATE["status_message"] = "正在检查并安装已保存自定义节点的 Python 依赖包..."
        _install_persisted_custom_node_requirements(comfy_root)
        rs.resolve_callable("sync_comfyui_to_drive_core", sync_comfyui_to_drive_core)(quiet=True)

        # 5. 启动 ComfyUI 服务进程 (loopback) 并实时镜像输出至 EasyColab 日志面板
        COMFYUI_STATE["progress"] = 0.80
        COMFYUI_STATE["status_message"] = "正在拉起 ComfyUI 节点推理引擎 (本地后端)..."
        subprocess.run(["pkill", "-f", "ComfyUI/main.py"], check=False)
        subprocess.run(["pkill", "-f", "cloudflared.*8188"], check=False)
        time.sleep(0.6)

        port = int(COMFYUI_STATE.get("local_port") or COMFY_BACKEND)
        comfy_cmd = [
            sys.executable, "-u", os.path.join(comfy_root, "main.py"),
            "--listen", "127.0.0.1",
            "--port", str(port),
            "--preview-method", "auto",
        ]
        comfy_proc = subprocess.Popen(
            comfy_cmd,
            cwd=comfy_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        COMFYUI_STATE["comfy_proc"] = comfy_proc

        def _forward_comfy_logs(proc: subprocess.Popen):
            try:
                if proc.stdout is None:
                    return
                for raw_ln in proc.stdout:
                    ln = raw_ln.rstrip()
                    if ln:
                        print(f"[ComfyUI] {ln}")
            except Exception:
                pass

        threading.Thread(target=_forward_comfy_logs, args=(comfy_proc,), daemon=True).start()

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        ready = False
        for _ in range(90):
            if comfy_proc.poll() is not None:
                raise RuntimeError(f"ComfyUI 启动失败提前退出 (code={comfy_proc.returncode})，请查看右侧运行日志。")
            try:
                with opener.open(f"http://127.0.0.1:{port}/system_stats", timeout=1.5) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(1.0)

        if not ready:
            raise RuntimeError("等待 ComfyUI 本地后端端口 就绪超时，请查看右侧运行日志。")

        # 6. 启动 Cloudflare WebSocket 高速公网隧道
        COMFYUI_STATE["progress"] = 0.92
        COMFYUI_STATE["status_message"] = "ComfyUI 内核已就绪，正在建立 Cloudflare 高速全屏画布隧道..."
        tunnel_proc, pub_url = _open_cloudflared_tunnel(COMFY_INGRESS, timeout_s=22.5)
        COMFYUI_STATE["tunnel_proc"] = tunnel_proc

        if not pub_url:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass
            COMFYUI_STATE["tunnel_proc"] = None
            raise RuntimeError("未能获取 Cloudflare 公网隧道地址，请重试。")

        COMFYUI_STATE["public_url"] = pub_url
        COMFYUI_STATE["is_running"] = True
        COMFYUI_STATE["progress"] = 1.0
        try:
            with open(_COMFY_STATE_FILE, "w", encoding="utf-8") as sf:
                json.dump({"public_url": pub_url, "port": port}, sf)
        except Exception:
            pass
        _COMFY_STATUS_CACHE["ts"] = 0.0
        (cn_count, m_count, _), _ = _scan_comfyui_counts_and_outputs()
        ok_msg = f"ComfyUI 已就绪（{label} · 节点 {cn_count} 个 · 权重 {m_count} 个）"
        COMFYUI_STATE["status_message"] = ok_msg
        print(f"[ComfyUI] 画布公网隧道已开通: {pub_url}")

        if not _COMFY_BG_SYNC_STARTED:
            _COMFY_BG_SYNC_STARTED = True
            threading.Thread(target=_comfyui_periodic_drive_sync_loop, daemon=True).start()

        return True, ok_msg
    except Exception as e:
        err_msg = f"ComfyUI 启动出错: {e}"
        COMFYUI_STATE["status_message"] = err_msg
        COMFYUI_STATE["progress"] = 0.0
        print(f"[ComfyUI] {err_msg}")
        return False, err_msg
    finally:
        COMFYUI_STATE["is_starting"] = False
        touch_activity()

def download_comfyui_model_core(
    url_or_repo: str,
    filename: str = "",
    subfolder: str = "diffusion_models",
    hf_token: str = "",
) -> Tuple[bool, str]:
    """8 线程并发下载模型权重直达持久化的 ComfyUI/models/<subfolder>，下次重开 G4 免重复下载"""
    touch_activity()
    raw_src = (url_or_repo or "").strip()
    if not raw_src:
        return False, "请输入 HuggingFace 仓库/直链或模型下载 URL"
    if COMFYUI_STATE["is_downloading_weight"]:
        return False, "当前已有 ComfyUI 权重正在下载中，请稍候"

    clean_sub = re.sub(r"[^a-zA-Z0-9_.-]", "", (subfolder or "diffusion_models").strip()) or "diffusion_models"
    comfy_root, persist_root, mode, label, _ = rs.resolve_callable("_resolve_comfyui_persistence_paths", _resolve_comfyui_persistence_paths)()
    dest_dir = os.path.join(persist_root, "models", clean_sub)
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.isdir(comfy_root):
        _safe_symlink_dir(dest_dir, os.path.join(comfy_root, "models", clean_sub))

    COMFYUI_STATE["is_downloading_weight"] = True
    COMFYUI_STATE["progress"] = 0.05
    try:
        token = (hf_token or "").strip() or None
        hf_m = re.match(r"^https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/[^/]+/(.+?)(\?.*)?$", raw_src)
        if hf_m:
            repo_id = hf_m.group(1)
            rel_file = hf_m.group(2)
            base_name = os.path.basename(filename.strip() or rel_file)
            COMFYUI_STATE["status_message"] = f"正在 8 线程极速下载 {base_name} 至 models/{clean_sub}..."
            print(f"[ComfyUI 极速下载] {repo_id} -> {rel_file} -> models/{clean_sub}/{base_name}")
            dl_path = rs.resolve_callable("fast_parallel_hf_download", fast_parallel_hf_download)(repo_id=repo_id, filename=rel_file, dest_dir=dest_dir, hf_token=token)
            final_path = os.path.join(dest_dir, base_name)
            if os.path.abspath(dl_path) != os.path.abspath(final_path):
                shutil.move(dl_path, final_path)
        elif raw_src.startswith("http://") or raw_src.startswith("https://"):
            url_path = urllib.parse.urlparse(raw_src).path
            base_name = os.path.basename(filename.strip() or url_path) or f"model_{int(time.time())}.safetensors"
            final_path = os.path.join(dest_dir, base_name)
            COMFYUI_STATE["status_message"] = f"正在下载 {base_name} 至 models/{clean_sub}..."
            print(f"[ComfyUI 直链下载] {raw_src} -> models/{clean_sub}/{base_name}")
            headers = {"User-Agent": "EasyColab/2.0"}
            if token and urllib.parse.urlsplit(raw_src).scheme == "https" and urllib.parse.urlsplit(raw_src).hostname == "huggingface.co":
                headers["Authorization"] = f"Bearer {token}"
            req = urllib.request.Request(raw_src, headers=headers)
            tmp_p = final_path + ".part"
            with urllib.request.build_opener(_StripAuthRedirectHandler()).open(req, timeout=120) as resp, open(tmp_p, "wb") as wf:
                total_sz = int(resp.headers.get("Content-Length") or 0)
                done_sz = 0
                t0 = time.time()
                last_log = 0.0
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    wf.write(chunk)
                    done_sz += len(chunk)
                    now_t = time.time()
                    if total_sz > 0 and (now_t - last_log) >= 0.6:
                        last_log = now_t
                        pct = done_sz / float(total_sz)
                        COMFYUI_STATE["progress"] = round(min(0.98, pct), 3)
                        spd = (done_sz / (1024 ** 2)) / max(0.01, now_t - t0)
                        print(f"{base_name}: {int(pct * 100)}%| {done_sz / (1024**3):.2f}G/{total_sz / (1024**3):.2f}G [{spd:.1f}MB/s]")
            os.replace(tmp_p, final_path)
        else:
            if not filename.strip():
                return False, "填写 HuggingFace 仓库 ID (如 Comfy-Org/Wan_2.1_ComfyUI_repackaged) 时请同时填写文件路径"
            rel_file = filename.strip()
            base_name = os.path.basename(rel_file)
            COMFYUI_STATE["status_message"] = f"正在 8 线程极速下载 {base_name} 至 models/{clean_sub}..."
            dl_path = rs.resolve_callable("fast_parallel_hf_download", fast_parallel_hf_download)(repo_id=raw_src, filename=rel_file, dest_dir=dest_dir, hf_token=token)
            final_path = os.path.join(dest_dir, base_name)
            if os.path.abspath(dl_path) != os.path.abspath(final_path):
                shutil.move(dl_path, final_path)

        if mode == "rclone_drive_sync":
            rs.resolve_callable("sync_comfyui_to_drive_core", sync_comfyui_to_drive_core)(quiet=True)
        _COMFY_STATUS_CACHE["ts"] = 0.0
        sz_gb = os.path.getsize(final_path) / (1024 ** 3)
        msg = f"权重 {base_name} ({sz_gb:.2f} GB) 已下载至 models/{clean_sub} 并持久化到 {label}（在画布按 R 键即可刷新选用）"
        COMFYUI_STATE["progress"] = 1.0
        COMFYUI_STATE["status_message"] = msg
        print(f"[ComfyUI] {msg}")
        return True, msg
    except Exception as e:
        err = f"下载 ComfyUI 权重失败: {e}"
        COMFYUI_STATE["status_message"] = err
        print(f"[ComfyUI] {err}")
        return False, err
    finally:
        COMFYUI_STATE["is_downloading_weight"] = False
        touch_activity()

def _download_comfy_model_entry(
    sub: str,
    fname: str,
    repo_id: str,
    repo_rel_path: str,
    persist_root: str,
    comfy_root: str,
    token: str = "",
) -> bool:
    """将单个 ComfyUI/Toybox 权重下载至持久化目录并建立软链；已存在(>1MB)则跳过，返回是否新下载。"""
    dest_dir = os.path.join(persist_root, "models", sub)
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.isdir(comfy_root):
        _safe_symlink_dir(dest_dir, os.path.join(comfy_root, "models", sub))

    final_path = os.path.join(dest_dir, fname)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    if os.path.isfile(final_path) and os.path.getsize(final_path) > 1024 * 1024:
        return False

    dl_path = rs.resolve_callable("fast_parallel_hf_download", fast_parallel_hf_download)(
        repo_id=repo_id,
        filename=repo_rel_path,
        dest_dir=dest_dir,
        hf_token=token,
        num_connections=8,
    )
    if os.path.abspath(dl_path) != os.path.abspath(final_path):
        shutil.move(dl_path, final_path)
        # 仅清理 fast_parallel_hf_download 创建的额外顶层嵌套目录，避免误删 final_path 所在子目录
        top_nested = repo_rel_path.split("/", 1)[0]
        fname_top = fname.split("/", 1)[0] if "/" in fname else ""
        if top_nested and "/" in repo_rel_path and top_nested != fname_top:
            nested_dir = os.path.join(dest_dir, top_nested)
            if os.path.isdir(nested_dir):
                try:
                    shutil.rmtree(nested_dir, ignore_errors=True)
                except Exception:
                    pass
    return True

def install_comfyui_preset_pack_core(
    pack_id: str,
    auto_start: bool = True,
    hf_token: str = "",
) -> Tuple[bool, str]:
    """一键配齐预置中文工作流包：8 线程下载缺失模型权重 + 写入中文傻瓜注释工作流 + 生成默认参考图 + 自动拉起/刷新画布"""
    touch_activity()
    target_pack = next((p for p in COMFY_PRESET_WORKFLOW_PACKS if p["id"] == pack_id), None)
    if not target_pack:
        return False, f"未找到预置工作流包: {pack_id}"
    if COMFYUI_STATE["is_downloading_weight"] or COMFYUI_STATE.get("installing_pack_id"):
        return False, "当前已有模型或工作流包正在高速下载配齐中，请稍候..."

    comfy_root, persist_root, mode, label, _ = rs.resolve_callable("_resolve_comfyui_persistence_paths", _resolve_comfyui_persistence_paths)()
    models_list = target_pack["models"]
    total_m = len(models_list)

    COMFYUI_STATE["is_downloading_weight"] = True
    COMFYUI_STATE["installing_pack_id"] = pack_id
    COMFYUI_STATE["progress"] = 0.03
    COMFYUI_STATE["status_message"] = f"正在准备配齐「{target_pack['title']}」（共 {total_m} 个官方权重）..."

    try:
        token = (hf_token or "").strip() or ""
        # 若为 rclone 模式且尚未拉取过云盘历史权重，先快速检查恢复，避免重复下载已持久化的权重
        pull_flag = "/tmp/.easycolab_comfy_drive_pulled"
        if mode == "rclone_drive_sync" and _has_rclone_drive_remote() and not os.path.exists(pull_flag):
            COMFYUI_STATE["status_message"] = "正在先从 Google Drive 检查已持久化的历史模型权重..."
            os.makedirs(persist_root, exist_ok=True)
            subprocess.run(
                [
                    "rclone", "copy",
                    "easycolab_gdrive:EasyColab_Drive/EasyColab_ComfyUI",
                    persist_root,
                    "--transfers", "8",
                    "--checkers", "16",
                    "--drive-chunk-size", "64M",
                ],
                check=False,
            )
            with open(pull_flag, "w") as pf:
                pf.write("1")

        _ensure_comfyui_beginner_assets(comfy_root, persist_root)
        if pack_id == "minimax_h3_toybox":
            from .toybox_service import _ensure_toybox_repos_and_nodes
            _ensure_toybox_repos_and_nodes(hf_token=token)

        for idx, (sub, fname, repo_id, repo_rel_path, approx_gb) in enumerate(models_list, start=1):
            touch_activity()
            base_prog = 0.05 + 0.80 * ((idx - 1) / float(max(1, total_m)))
            COMFYUI_STATE["progress"] = round(base_prog, 3)
            COMFYUI_STATE["status_message"] = (
                f"[{idx}/{total_m}] 正在检查/下载 {fname} (约 {approx_gb:.1f} GB)..."
            )
            downloaded = _download_comfy_model_entry(
                sub=sub,
                fname=fname,
                repo_id=repo_id,
                repo_rel_path=repo_rel_path,
                persist_root=persist_root,
                comfy_root=comfy_root,
                token=token,
            )
            if not downloaded:
                print(f"[ComfyUI 预置包] ({idx}/{total_m}) 已存在跳过: models/{sub}/{fname}")
            else:
                print(f"[ComfyUI 预置包] ({idx}/{total_m}) 下载完成: models/{sub}/{fname}")

        COMFYUI_STATE["progress"] = 0.88
        COMFYUI_STATE["status_message"] = f"正在生成「{target_pack['title']}」全中文三步傻瓜注释工作流..."
        installed_wfs = _build_and_install_comfy_preset_workflows(persist_root, [pack_id])
        if os.path.isdir(comfy_root):
            _safe_symlink_dir(os.path.join(persist_root, "user"), os.path.join(comfy_root, "user"))
            _safe_symlink_dir(os.path.join(persist_root, "output"), os.path.join(comfy_root, "output"))
            _ensure_comfyui_beginner_assets(comfy_root, persist_root)

        if mode == "rclone_drive_sync":
            COMFYUI_STATE["progress"] = 0.93
            COMFYUI_STATE["status_message"] = "正在将配齐的模型与中文工作流持久化同步至 Google Drive..."
            rs.resolve_callable("sync_comfyui_to_drive_core", sync_comfyui_to_drive_core)(quiet=True)

        _COMFY_STATUS_CACHE["ts"] = 0.0
        wf_names_str = "、".join(installed_wfs) if installed_wfs else "、".join(target_pack["workflows"])

        # 若 ComfyUI 正在运行，主动轻触 /object_info 刷新模型下拉缓存；若尚未启动且开启了 auto_start，直接自动拉起画布
        if COMFYUI_STATE.get("is_running"):
            try:
                port = int(COMFYUI_STATE.get("local_port") or COMFY_BACKEND)
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f"http://127.0.0.1:{port}/object_info/UNETLoader", timeout=3.0) as _:
                    pass
            except Exception:
                pass
            COMFYUI_STATE["progress"] = 1.0
            ok_msg = f"「{target_pack['title']}」已全部配齐！打开画布左侧「工作流」点击 {wf_names_str} 即可直接传图运行"
            COMFYUI_STATE["status_message"] = ok_msg
            print(f"[ComfyUI 预置包] {ok_msg}")
            return True, ok_msg
    except Exception as e:
        err = f"配齐预置工作流包失败: {e}"
        COMFYUI_STATE["progress"] = 0.0
        COMFYUI_STATE["status_message"] = err
        print(f"[ComfyUI 预置包] {err}")
        return False, err
    finally:
        COMFYUI_STATE["is_downloading_weight"] = False
        COMFYUI_STATE["installing_pack_id"] = ""
        touch_activity()

    # 若尚未拉起 ComfyUI，在释放下载锁后自动拉起云端画布隧道
    if auto_start and not COMFYUI_STATE.get("is_running") and not COMFYUI_STATE.get("is_starting"):
        print("[ComfyUI 预置包] 权重与工作流已就绪，正在自动拉起 ComfyUI 全屏画布...")
        ok_s, msg_s = rs.resolve_callable("start_comfyui_core", start_comfyui_core)(unload_existing=False, force_restart=False)
        if ok_s:
            final_msg = f"「{target_pack['title']}」已配齐且画布已启动！在左侧「工作流」打开中文模板即可直接出片"
            COMFYUI_STATE["status_message"] = final_msg
            return True, final_msg
        return False, msg_s

    return True, f"「{target_pack['title']}」模型与中文工作流已配齐！"
