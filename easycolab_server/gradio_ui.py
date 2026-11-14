"""EasyColab 服务端模块: gradio_ui."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import gradio as gr
import os
import re
import secrets

from . import runtime_state as rs
from .comfyui_service import get_comfyui_status_dict, start_comfyui_core
from .media_pipeline import MAX_REF2VA_AUDIOS, MAX_REF2VA_IMAGES, MAX_REF2VA_VIDEOS, generate_video, load_model_to_vram_core
from .model_catalog import DEFAULT_OUTPUT_DIR, MODEL_CATALOG, _normalize_builtin_model_alias, download_model_weights_core, fetch_hf_trending_models_core, get_history_videos, register_custom_hf_model_core
from .runtime_state import API_KEY_LOCK
from .status_service import get_gpu_and_model_summary_text
from .telemetry_service import disconnect_colab_runtime
from .toybox_service import get_toybox_status_dict, prepare_toybox_assets_core, start_toybox_core, stop_toybox_core

# ==============================================================================
# 预设提示词模板与 Gradio 界面
# ==============================================================================
PRESET_EXAMPLES = [
    [
        "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.",
        "Metallic clacking of train wheels on railway tracks, rhythmic dinging of the railroad crossing warning bell, warm spring wind rustling through leaves, a soft gasp from the girl.",
        "Emotional and nostalgic solo piano melody joined by warm acoustic guitar and soaring cinematic strings.",
        "832 × 480 (16:9 横屏标准 - 推荐)",
        124,
        35,
    ],
    [
        "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur.",
        "Sharp metallic sword unsheathing and clashing clang, roaring thunder and crackling electric arcs, heavy stone debris crumbling, fierce battle cry.",
        "Epic dark orchestral battle music with fast taiko drums, intense choir, and distorted electric guitar riffs.",
        "832 × 480 (16:9 横屏标准 - 推荐)",
        124,
        40,
    ],
]

CUSTOM_CSS = """
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
    color: #ffffff;
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 16px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}
.header-banner h1 {
    margin: 0 0 6px 0;
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
}
.header-banner p {
    margin: 0;
    font-size: 0.95rem;
    color: #cbd5e1;
}
.status-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
}
"""

WEBUI_HEADER_ZH = """
<div class="header-banner">
    <h1>⚡ EasyColab · Google Colab 多模态 AI 推理与生成服务端</h1>
    <p>支持 <b>ComfyUI 节点工作台 (云盘软链持久化) · MiniMax-H3 · Qwen3.8-27B (GGUF+MTP / BF16) · Qwen-Image-2.1 · Z-Image-Turbo</b> · 自动关机看门狗</p>
</div>
"""

WEBUI_HEADER_EN = """
<div class="header-banner">
    <h1>⚡ EasyColab · Google Colab Multimodal AI Inference & Generation Server</h1>
    <p>Supports <b>ComfyUI Node Studio (Google Drive Persistence) · MiniMax-H3 · Qwen3.8-27B (GGUF+MTP / BF16) · Qwen-Image-2.1 · Z-Image-Turbo</b> · Idle Auto-Shutdown Watchdog</p>
</div>
"""

RESOLUTION_CHOICES_ZH = [
    ("832 × 480 (16:9 横屏标准 - 推荐)", "832 × 480 (16:9 横屏标准 - 推荐)"),
    ("480 × 832 (9:16 竖屏短视频)", "480 × 832 (9:16 竖屏短视频)"),
    ("1280 × 720 (16:9 720P 高清)", "1280 × 720 (16:9 720P 高清)"),
    ("720 × 1280 (9:16 720P 竖屏高清)", "720 × 1280 (9:16 720P 竖屏高清)"),
    ("640 × 640 (1:1 方形)", "640 × 640 (1:1 方形)"),
]

RESOLUTION_CHOICES_EN = [
    ("832 × 480 (16:9 Landscape · Recommended)", "832 × 480 (16:9 横屏标准 - 推荐)"),
    ("480 × 832 (9:16 Portrait Short)", "480 × 832 (9:16 竖屏短视频)"),
    ("1280 × 720 (16:9 720P HD)", "1280 × 720 (16:9 720P 高清)"),
    ("720 × 1280 (9:16 720P Portrait HD)", "720 × 1280 (9:16 720P 竖屏高清)"),
    ("640 × 640 (1:1 Square)", "640 × 640 (1:1 方形)"),
]

def create_ui() -> gr.Blocks:
    with gr.Blocks(title="EasyColab · Colab 多模型 AI 服务端") as demo:
        ui_lang_state = gr.State(value="zh")

        with gr.Row():
            with gr.Column(scale=9):
                header_html = gr.HTML(WEBUI_HEADER_ZH)
            with gr.Column(scale=1, min_width=130):
                lang_toggle_btn = gr.Button("🌐 English", size="sm", variant="secondary")

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group():
                    sec1_md = gr.Markdown("### 📦 1. 模型权重管理中心 (先下载检查，再加载显存)")
                    model_selector_dropdown = gr.Dropdown(
                        choices=[
                            (f"{mspec['name']} · {mspec['category']}", mid)
                            for mid, mspec in MODEL_CATALOG.items()
                        ],
                        value=rs.SELECTED_MODEL_ID,
                        label="🤖 当前活动模型 (Active Model)",
                    )
                    gpu_status_text = gr.Textbox(
                        label="硬件、算力消耗与模型检查报告",
                        value=get_gpu_and_model_summary_text(),
                        lines=9,
                        interactive=False,
                        elem_classes=["status-box"],
                    )
                    refresh_gpu_btn = gr.Button("🔍 检查 / 刷新模型与算力状态", size="sm")

                    precision_Choice = gr.Radio(
                        choices=[
                            ("BF16 原生精度版 (官方满血权重 · G4 96GB 极速推理)", "bf16"),
                        ],
                        value="bf16",
                        label="模型权重精度",
                    )

                    offload_choice = gr.Radio(
                        choices=[
                            ("⚡ 零拷贝极速映射 + GPU 显存常驻 (推荐 · 5秒秒加载)", "disk"),
                            ("CPU 内存全量预载 (耗时较长 · 约3~5分钟)", "cpu"),
                        ],
                        value="disk",
                        label="显存加载与卸载策略 (VRAM Strategy)",
                    )

                    with gr.Accordion("🔑 Hugging Face Token (可选 / Optional)", open=False):
                        hf_token_input = gr.Textbox(
                            label="HF Access Token",
                            placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx",
                            type="password",
                        )

                    with gr.Accordion("➕ 添加自定义 HuggingFace 模型仓库 (Custom HF Repo Import)", open=False):
                        with gr.Row():
                            trending_hf_dropdown = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="🔥 HuggingFace 实时热门趋势模型 (点击右侧刷新拉取最新热榜)",
                                allow_custom_value=True,
                                scale=4,
                            )
                            refresh_trending_hf_btn = gr.Button("🔄 拉取实时热榜", size="sm", scale=1)
                        custom_repo_input = gr.Textbox(
                            label="HuggingFace Repo ID 或完整链接",
                            placeholder="从上方实时热榜选择，或输入任意 Owner/Repo",
                        )
                        custom_pattern_input = gr.Textbox(
                            label="文件名 / 通配符过滤 (Include Pattern，可选)",
                            placeholder="例如: *Q4_K_M*.gguf 或 *.safetensors（留空自动优选）",
                        )
                        custom_name_input = gr.Textbox(
                            label="自定义显示名称 (可选)",
                            placeholder="留空则默认使用仓库名",
                        )
                        add_custom_model_btn = gr.Button("➕ 解析并导入自定义 HF 模型", variant="secondary", size="sm")

                    with gr.Row():
                        download_model_btn = gr.Button(
                            "📥 1. 极速下载模型权重",
                            variant="secondary",
                        )
                        load_model_btn = gr.Button(
                            "🚀 2. 加载模型至显存",
                            variant="primary",
                        )

                with gr.Group():
                    sec2_md = gr.Markdown("### ⚙️ 2. 视频与采样参数")
                    resolution_dropdown = gr.Dropdown(
                        choices=RESOLUTION_CHOICES_ZH,
                        value="832 × 480 (16:9 横屏标准 - 推荐)",
                        label="输出分辨率 (Resolution)",
                    )
                    with gr.Row():
                        num_frames_slider = gr.Slider(
                            minimum=90,
                            maximum=362,
                            step=17,
                            value=124,
                            label="生成总帧数 (90~362帧 ≈ 4~15秒, 17n+5 对齐)",
                        )
                        fps_slider = gr.Slider(
                            minimum=16,
                            maximum=30,
                            step=1,
                            value=24,
                            label="帧率 (FPS)",
                        )
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=60,
                        step=5,
                        value=35,
                        label="采样步数 (Inference Steps，推荐 30~50)",
                    )
                    with gr.Row():
                        seed_input = gr.Number(
                            value=42,
                            precision=0,
                            label="随机种子 (Seed)",
                        )
                        randomize_seed_chk = gr.Checkbox(
                            value=True,
                            label="🎲 每次随机种子",
                        )

                    save_dir_input = gr.Textbox(
                        value=rs.DEFAULT_OUTPUT_DIR,
                        label="📂 视频自动保存路径",
                    )

                with gr.Accordion("🧸 H3 Toybox 独立创作台 & ComfyUI 工作台 (共用引擎 + 密码隧道)", open=True):
                    gr.Markdown(
                        "共用 **同一个 ComfyUI 引擎、模型权重与 Drive 持久化目录**，一次下好权重可在两套界面无缝切换：\n"
                        "- **🧸 H3 Toybox**：素材库文件夹、`@图片1` 提示词改写、Ref2VA/首尾帧/视频延伸、P-H3 生图，通过**独立带登录密码的 Cloudflare 隧道**访问（用户名 `easycolab` + 你的部署密码）。\n"
                        "- **🎨 ComfyUI 原生画布**：随时可同机打开原生节点画布调试工作流。"
                    )
                    toybox_status_md = gr.Markdown(value="")
                    with gr.Row():
                        toybox_inc_rewriter_chk = gr.Checkbox(value=True, label="下载 9B 本地改写模型 (~7GB)")
                        toybox_inc_image_chk = gr.Checkbox(value=True, label="下载 P-H3 生图套件 (~10GB)")
                        toybox_unload_chk = gr.Checkbox(value=True, label="启动前释放本地模型显存")
                    with gr.Row():
                        toybox_prepare_btn = gr.Button("📥 1. 一键配齐 Toybox 节点与权重", variant="secondary", size="sm")
                        toybox_start_btn = gr.Button("🧸 2. 启动 H3 Toybox (带密码独立隧道)", variant="primary", size="sm")
                    with gr.Row():
                        comfy_start_web_btn = gr.Button("🎨 启动 ComfyUI 原生画布隧道", variant="secondary", size="sm")
                        toybox_stop_btn = gr.Button("⏹️ 停止 H3 Toybox", size="sm")
                        toybox_refresh_btn = gr.Button("🔄 刷新工作台状态", size="sm")

                with gr.Accordion("🔌 OpenAI 兼容 API 与 Token 认证 / External API Gateway", open=False):
                    gr.Markdown(
                        "支持接入 **Open WebUI / Cherry Studio / ChatBox / NextChat / Claude Code / OpenAI SDK**：\n"
                        "- Base URL: `<你的服务端地址>/v1`\n"
                        "- 登录密码 (`Basic Auth`) 仅保护本 Web UI；`/v1/*` 独立使用下方 **API Token**（`Authorization: Bearer <Token>` 或 `x-api-key`），两者互不冲突。\n"
                        "- 支持端点：`GET /v1/models`、`POST /v1/chat/completions`（含 LLM 与图/视频路由）、`POST /v1/images/generations`、`/v1/images/edits`、`POST/GET/DELETE /v1/videos`"
                    )
                    api_token_input = gr.Textbox(
                        value=lambda: rs.API_AUTH_KEY,
                        label="🔑 OpenAI / Anthropic API Token（支持逗号分隔多 Token）",
                        placeholder="sk-colab-...",
                    )
                    with gr.Row():
                        save_api_token_btn = gr.Button("💾 保存 / 应用 Token", variant="secondary", size="sm")
                        regen_api_token_btn = gr.Button("🔄 重新生成随机 Token", size="sm")

                with gr.Accordion("💡 Google Colab 算力省额度工具 / Compute Quota Saver", open=True):
                    disconnect_btn = gr.Button("🛑 停止任务并释放 Colab 运行时 (省计算单元)", variant="stop")

            with gr.Column(scale=6):
                task_mode_state = gr.State(value="t2va")

                with gr.Tabs() as mode_tabs:
                    with gr.TabItem("✨ 文生音视频 / T2VA", id="t2va") as tab_t2va:
                        t2va_desc_md = gr.Markdown("输入画面描述与声音细节，同步生成高清视频与 **32kHz 立体声音轨**。")

                    with gr.TabItem("🖼️ 首尾帧生音视频 / FL2VA", id="fl2va") as tab_fl2va:
                        fl2va_desc_md = gr.Markdown("在下方图像区上传 **首帧** 和/或 **尾帧** 图片，按关键帧锚点生成过渡视频与立体声。")

                    with gr.TabItem("🎭 主体参考音视频 / Ref2VA (最多9图+3视频+3音频)", id="ref2va") as tab_ref2va:
                        ref2va_desc_md = gr.Markdown(
                            "支持官方满血多模态参考上限：最多上传 **9 张参考图**（提示词用 `<Picture 1>`~`<Picture 9>` 引用角色/场景）+ **3 段参考视频**（用 `<Video 1>`~`<Video 3>` 引用动作/运镜）+ **3 段参考音频**（用 `<Audio 1>`~`<Audio 3>` 引用人声/配乐），总参考资产上限 **12 个**。\n"
                            "💡 **提示词联动**：在下方提示词输入框中输入 **`@`** 符号（如 `@`、`@图`、`@视频`、`@音频`、`@1`），即可触发下拉选择并自动将 `@...` 替换为对应的 `<Picture N>` / `<Video N>` / `<Audio N>` 顺序标签！"
                        )
                        with gr.Row():
                            ref_images_files_input = gr.File(
                                file_count="multiple",
                                file_types=["image"],
                                label="🖼️ 多张主体/场景参考图 (最多 9 张 · 对应 <Picture 1> ~ <Picture 9>)",
                            )
                            ref_videos_files_input = gr.File(
                                file_count="multiple",
                                file_types=["video"],
                                label="🎬 动作/运镜参考视频 (最多 3 段 · 对应 <Video 1> ~ <Video 3>)",
                            )
                            ref_audios_files_input = gr.File(
                                file_count="multiple",
                                file_types=["audio"],
                                label="🎵 音色/节拍参考音频 (最多 3 段 · 对应 <Audio 1> ~ <Audio 3>)",
                            )
                        ref2va_mapping_md = gr.Markdown(
                            "🔗 **当前已上传参考资产与标签对应关系**：尚未上传多模态参考资产（支持在提示词中输入 `@` 触发选择）"
                        )

                with gr.Row():
                    first_frame_input = gr.Image(
                        label="📌 首帧图片 / 参考图 1 (<Picture 1>)",
                        type="pil",
                        height=220,
                    )
                    last_frame_input = gr.Image(
                        label="🏁 尾帧图片 / 参考图 2 (<Picture 2>，可选)",
                        type="pil",
                        height=220,
                    )

                with gr.Group():
                    with gr.Row():
                        ref2va_at_selector = gr.Dropdown(
                            choices=[
                                ("🖼️ @图片1 → <Picture 1>", "<Picture 1>"),
                                ("🖼️ @图片2 → <Picture 2>", "<Picture 2>"),
                                ("🎬 @视频1 → <Video 1>", "<Video 1>"),
                                ("🎵 @音频1 → <Audio 1>", "<Audio 1>"),
                            ],
                            value=None,
                            label="⚡ @ 触发参考资产选择器（在下方提示词中输入 @ 自动过滤，或点击直接插入对应标签）",
                            scale=4,
                        )
                    visual_prompt_input = gr.Textbox(
                        label="🎥 画面主体与镜头描述 (输入 @ 触发插入 <Picture N> / <Video N> / <Audio N>)",
                        placeholder="描述画面主体、动作、运镜方式、光影风格...（输入 @ 可快速选择已上传的图片、视频或音频标签）",
                        lines=4,
                    )
                    with gr.Row():
                        soundscape_prompt_input = gr.Textbox(
                            label="🔊 环境音效与人声描述 (Soundscape & Foley - 可选)",
                            placeholder="例如：雨滴敲击声、脚步踩水声、温柔的女声独白...",
                            lines=2,
                        )
                        music_prompt_input = gr.Textbox(
                            label="🎵 背景配乐风格 (Background Music - 可选)",
                            placeholder="例如：低沉的赛博朋克合成器电子乐、恢弘管弦乐...",
                            lines=2,
                        )

                generate_btn = gr.Button(
                    "🎬 开始生成音视频 (Generate Video + Stereo Audio)",
                    variant="primary",
                    size="lg",
                )

                with gr.Group():
                    output_video = gr.Video(
                        label="🎞️ 生成结果预览 (请开启播放器声音体验原生立体声)",
                        height=420,
                        autoplay=True,
                    )
                    run_status_box = gr.Textbox(
                        label="运行日志与状态",
                        value="请先在左侧点击「1. 极速下载模型权重」并「2. 加载模型至显存」...",
                        interactive=False,
                        lines=2,
                        elem_classes=["status-box"],
                    )

                with gr.Accordion("📚 提示词一键体验模板 / Preset Prompt Templates", open=True):
                    gr.Examples(
                        examples=PRESET_EXAMPLES,
                        inputs=[
                            visual_prompt_input,
                            soundscape_prompt_input,
                            music_prompt_input,
                            resolution_dropdown,
                            num_frames_slider,
                            steps_slider,
                        ],
                    )

                with gr.Accordion("🗂️ 本次会话历史生成记录 / Session History", open=False):
                    history_gallery = gr.Gallery(
                        label="已生成的视频文件",
                        value=get_history_videos(rs.DEFAULT_OUTPUT_DIR),
                        columns=3,
                        height=240,
                    )
                    refresh_history_btn = gr.Button("🔄 刷新历史记录", size="sm")

        def _toggle_webui_lang(cur_lang: str, cur_prec: str, cur_offload: str, cur_res: str, cur_status: str):
            new_lang = "en" if cur_lang == "zh" else "zh"
            is_en = new_lang == "en"
            default_zh_status = "请先在左侧点击「1. 极速下载模型权重」并「2. 加载模型至显存」..."
            default_en_status = "Click '1. Download Weights' then '2. Load to VRAM' on the left to get started..."
            if is_en and cur_status == default_zh_status:
                next_status = default_en_status
            elif not is_en and cur_status == default_en_status:
                next_status = default_zh_status
            else:
                next_status = cur_status

            return (
                new_lang,
                WEBUI_HEADER_EN if is_en else WEBUI_HEADER_ZH,
                "🌐 简体中文" if is_en else "🌐 English",
                "### 📦 1. Model Weights Center (Download First, Then Load to VRAM)" if is_en else "### 📦 1. 模型权重管理中心 (先下载检查，再加载显存)",
                gr.update(label="Hardware, CU Usage & Model Inspection Report" if is_en else "硬件、算力消耗与模型检查报告"),
                "🔍 Check / Refresh GPU & Model Status" if is_en else "🔍 检查 / 刷新模型与算力状态",
                gr.update(
                    choices=[
                        ("Native BF16 Full Precision (Recommended for G4 96GB)" if is_en else "BF16 原生精度版 (官方满血权重 · G4 96GB 极速推理)", "bf16"),
                    ],
                    value=cur_prec or "bf16",
                    label="Model Precision" if is_en else "模型权重精度",
                ),
                gr.update(
                    choices=[
                        ("⚡ Zero-Copy Fast Mapping + GPU VRAM Resident (Recommended · ~5s)" if is_en else "⚡ 零拷贝极速映射 + GPU 显存常驻 (推荐 · 5秒秒加载)", "disk"),
                        ("Full CPU RAM Preload (Slower · ~3-5 min)" if is_en else "CPU 内存全量预载 (耗时较长 · 约3~5分钟)", "cpu"),
                    ],
                    value=cur_offload or "disk",
                    label="VRAM Loading Strategy" if is_en else "显存加载与卸载策略 (VRAM Strategy)",
                ),
                "📥 1. Download Model Weights" if is_en else "📥 1. 极速下载模型权重",
                "🚀 2. Load Model to VRAM" if is_en else "🚀 2. 加载模型至显存",
                "### ⚙️ 2. Video & Sampling Parameters" if is_en else "### ⚙️ 2. 视频与采样参数",
                gr.update(
                    choices=RESOLUTION_CHOICES_EN if is_en else RESOLUTION_CHOICES_ZH,
                    value=cur_res or "832 × 480 (16:9 横屏标准 - 推荐)",
                    label="Output Resolution" if is_en else "输出分辨率 (Resolution)",
                ),
                gr.update(label="Total Frames (90~362 frames ≈ 4~15s, 17n+5 aligned)" if is_en else "生成总帧数 (90~362帧 ≈ 4~15秒, 17n+5 对齐)"),
                gr.update(label="Frame Rate (FPS)" if is_en else "帧率 (FPS)"),
                gr.update(label="Inference Steps (30~50 recommended)" if is_en else "采样步数 (Inference Steps，推荐 30~50)"),
                gr.update(label="Seed" if is_en else "随机种子 (Seed)"),
                gr.update(label="🎲 Randomize Seed Every Run" if is_en else "🎲 每次随机种子"),
                gr.update(label="📂 Output Save Directory" if is_en else "📂 视频自动保存路径"),
                "🛑 Stop Jobs & Release Colab Runtime (Save CU)" if is_en else "🛑 停止任务并释放 Colab 运行时 (省计算单元)",
                "Describe visual scenes and soundscapes to generate HD video with **32kHz stereo audio**." if is_en else "输入画面描述与声音细节，同步生成高清视频与 **32kHz 立体声音轨**。",
                gr.update(label="📌 First Frame Image (index 0)" if is_en else "📌 首帧图片 (First Frame - index 0)"),
                gr.update(label="🏁 Last Frame Image (index -1, Optional)" if is_en else "🏁 尾帧图片 (Last Frame - index -1，可选)"),
                gr.update(
                    label="🎥 Visual & Camera Prompt" if is_en else "🎥 画面主体与镜头描述 (Visual & Camera Prompt)",
                    placeholder="Describe subject, motion, camera movement, lighting, and visual style..." if is_en else "描述画面主体、动作、运镜方式、光影风格...",
                ),
                gr.update(
                    label="🔊 Soundscape & Dialogue (Optional)" if is_en else "🔊 环境音效与人声描述 (Soundscape & Foley - 可选)",
                    placeholder="e.g., Raindrops hitting pavement, footsteps in puddles, soft spoken dialogue..." if is_en else "例如：雨滴敲击声、脚步踩水声、温柔的女声独白...",
                ),
                gr.update(
                    label="🎵 Background Music Style (Optional)" if is_en else "🎵 背景配乐风格 (Background Music - 可选)",
                    placeholder="e.g., Moody cyberpunk synthwave, soaring orchestral strings..." if is_en else "例如：低沉的赛博朋克合成器电子乐、恢弘管弦乐...",
                ),
                "🎬 Generate Video + Stereo Audio" if is_en else "🎬 开始生成音视频 (Generate Video + Stereo Audio)",
                gr.update(label="🎞️ Output Preview (Unmute player for native stereo audio)" if is_en else "🎞️ 生成结果预览 (请开启播放器声音体验原生立体声)"),
                gr.update(label="Run Status & Logs" if is_en else "运行日志与状态", value=next_status),
                gr.update(label="Generated Videos" if is_en else "已生成的视频文件"),
                "🔄 Refresh History" if is_en else "🔄 刷新历史记录",
            )

        lang_toggle_btn.click(
            fn=_toggle_webui_lang,
            inputs=[ui_lang_state, precision_Choice, offload_choice, resolution_dropdown, run_status_box],
            outputs=[
                ui_lang_state,
                header_html,
                lang_toggle_btn,
                sec1_md,
                gpu_status_text,
                refresh_gpu_btn,
                precision_Choice,
                offload_choice,
                download_model_btn,
                load_model_btn,
                sec2_md,
                resolution_dropdown,
                num_frames_slider,
                fps_slider,
                steps_slider,
                seed_input,
                randomize_seed_chk,
                save_dir_input,
                disconnect_btn,
                t2va_desc_md,
                first_frame_input,
                last_frame_input,
                visual_prompt_input,
                soundscape_prompt_input,
                music_prompt_input,
                generate_btn,
                output_video,
                run_status_box,
                history_gallery,
                refresh_history_btn,
            ],
        )

        tab_t2va.select(fn=lambda: "t2va", inputs=None, outputs=task_mode_state)
        tab_fl2va.select(fn=lambda: "fl2va", inputs=None, outputs=task_mode_state)
        tab_ref2va.select(fn=lambda: "ref2va", inputs=None, outputs=task_mode_state)

        def _on_webui_select_model(mid: str):
            # (shared state on runtime_state)
            norm_id = _normalize_builtin_model_alias(mid)
            if norm_id and norm_id in MODEL_CATALOG:
                rs.SELECTED_MODEL_ID = norm_id
            return get_gpu_and_model_summary_text()

        model_selector_dropdown.change(
            fn=_on_webui_select_model,
            inputs=[model_selector_dropdown],
            outputs=[gpu_status_text],
        )

        refresh_gpu_btn.click(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        def _refresh_gradio_trending_choices(tok: str):
            items = fetch_hf_trending_models_core("all", 14, tok, force_refresh=True)
            choices = [
                (
                    f"🔥 {m['repo_id']} ({(m.get('param_label') + ' · ') if m.get('param_label') else ''}{m.get('deploy_label_zh', '')} · {'🔒需Token · ' if m.get('is_gated') else ''}{m.get('pipeline_tag', '')} · ❤️{m.get('likes', 0)})",
                    f"{m['repo_id']}|{m.get('suggested_pattern', '')}",
                )
                for m in items
            ]
            return gr.update(choices=choices, value=choices[0][1] if choices else None)

        def _on_select_trending_repo(val: str):
            if not val:
                return gr.update(), gr.update()
            parts = str(val).split("|", 1)
            repo = parts[0].strip()
            pat = parts[1].strip() if len(parts) > 1 else ""
            return repo, pat

        refresh_trending_hf_btn.click(
            fn=_refresh_gradio_trending_choices,
            inputs=[hf_token_input],
            outputs=[trending_hf_dropdown],
        )
        trending_hf_dropdown.change(
            fn=_on_select_trending_repo,
            inputs=[trending_hf_dropdown],
            outputs=[custom_repo_input, custom_pattern_input],
        )

        add_custom_model_btn.click(
            fn=lambda repo, pat, cname, tok: register_custom_hf_model_core(repo, cname, pat, "auto", tok)[1],
            inputs=[custom_repo_input, custom_pattern_input, custom_name_input, hf_token_input],
            outputs=run_status_box,
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        download_model_btn.click(
            fn=lambda tok: rs.resolve_callable("download_model_weights_core", download_model_weights_core)(rs.SELECTED_MODEL_ID, tok)[1],
            inputs=[hf_token_input],
            outputs=run_status_box,
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        load_model_btn.click(
            fn=lambda prec, off, tok: rs.resolve_callable("load_model_to_vram_core", load_model_to_vram_core)(rs.SELECTED_MODEL_ID, prec, off, tok)[1],
            inputs=[precision_Choice, offload_choice, hf_token_input],
            outputs=run_status_box,
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        def _build_ref2va_at_candidates(first_im, last_im, img_files, vid_files, aud_files, prompt_txt: str = ""):
            choices: List[Tuple[str, str]] = []
            md_parts: List[str] = []

            img_names: List[str] = []
            if first_im is not None:
                img_names.append("首帧图片")
            if last_im is not None:
                img_names.append("尾帧图片")
            for f in (img_files or []):
                fname = os.path.basename(getattr(f, "name", str(f)))
                img_names.append(fname)
            img_names = img_names[:MAX_REF2VA_IMAGES]

            vid_names: List[str] = [
                os.path.basename(getattr(f, "name", str(f))) for f in (vid_files or [])
            ][:MAX_REF2VA_VIDEOS]
            aud_names: List[str] = [
                os.path.basename(getattr(f, "name", str(f))) for f in (aud_files or [])
            ][:MAX_REF2VA_AUDIOS]

            for i, name in enumerate(img_names, start=1):
                tag = f"<Picture {i}>"
                choices.append((f"🖼️ @图片{i} → {tag} ({name})", tag))
                md_parts.append(f"`{tag}` ↔ 🖼️ {name}")
            for i, name in enumerate(vid_names, start=1):
                tag = f"<Video {i}>"
                choices.append((f"🎬 @视频{i} → {tag} ({name})", tag))
                md_parts.append(f"`{tag}` ↔ 🎬 {name}")
            for i, name in enumerate(aud_names, start=1):
                tag = f"<Audio {i}>"
                choices.append((f"🎵 @音频{i} → {tag} ({name})", tag))
                md_parts.append(f"`{tag}` ↔ 🎵 {name}")

            if not choices:
                choices = [
                    ("🖼️ @图片1 → <Picture 1> (待上传参考图 1)", "<Picture 1>"),
                    ("🖼️ @图片2 → <Picture 2> (待上传参考图 2)", "<Picture 2>"),
                    ("🎬 @视频1 → <Video 1> (待上传参考视频 1)", "<Video 1>"),
                    ("🎵 @音频1 → <Audio 1> (待上传参考音频 1)", "<Audio 1>"),
                ]
                md_str = "🔗 **当前已上传参考资产与标签对应关系**：尚未上传多模态参考资产（支持在提示词中输入 `@` 触发选择）"
            else:
                md_str = "🔗 **当前已上传参考资产与标签对应关系（严格按 1 起始顺序号对应）**：" + " ｜ ".join(md_parts)

            # 若提示词末尾或内部包含 @ 过滤词（如 @图 / @vid / @aud / @1），优先过滤匹配项
            m = re.search(r"@([^\s<>@]*)$", (prompt_txt or ""))
            if m:
                kw = m.group(1).strip().lower()
                if kw:
                    filtered = [
                        (lbl, val) for lbl, val in choices
                        if kw in lbl.lower() or kw in val.lower()
                    ]
                    if filtered:
                        choices = filtered
            return gr.update(choices=choices, value=None), md_str

        def _on_select_at_reference_tag(selected_tag: Optional[str], current_prompt: str):
            if not selected_tag:
                return gr.update(), gr.update(value=None)
            txt = current_prompt or ""
            # 若用户在提示词中输入了 @ 或 @关键词，直接将该 @片段替换为选中的 <Picture N> / <Video N> / <Audio N>
            if re.search(r"@[^\s<>@]*$", txt):
                new_txt = re.sub(r"@[^\s<>@]*$", f"{selected_tag} ", txt)
            elif "@" in txt:
                idx = txt.rfind("@")
                end_idx = idx + 1
                while end_idx < len(txt) and not txt[end_idx].isspace() and txt[end_idx] not in "<>@":
                    end_idx += 1
                new_txt = txt[:idx] + selected_tag + " " + txt[end_idx:].lstrip()
            else:
                sep = "" if (not txt or txt.endswith(" ")) else " "
                new_txt = f"{txt}{sep}{selected_tag} "
            return new_txt, gr.update(value=None)

        for _ref_comp in (first_frame_input, last_frame_input, ref_images_files_input, ref_videos_files_input, ref_audios_files_input):
            _ref_comp.change(
                fn=_build_ref2va_at_candidates,
                inputs=[first_frame_input, last_frame_input, ref_images_files_input, ref_videos_files_input, ref_audios_files_input, visual_prompt_input],
                outputs=[ref2va_at_selector, ref2va_mapping_md],
            )

        ref2va_at_selector.change(
            fn=_on_select_at_reference_tag,
            inputs=[ref2va_at_selector, visual_prompt_input],
            outputs=[visual_prompt_input, ref2va_at_selector],
        )

        generate_btn.click(
            fn=generate_video,
            inputs=[
                task_mode_state,
                visual_prompt_input,
                soundscape_prompt_input,
                music_prompt_input,
                first_frame_input,
                last_frame_input,
                resolution_dropdown,
                num_frames_slider,
                steps_slider,
                fps_slider,
                seed_input,
                randomize_seed_chk,
                save_dir_input,
                ref_images_files_input,
                ref_videos_files_input,
                ref_audios_files_input,
            ],
            outputs=[output_video, run_status_box, seed_input, history_gallery],
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        refresh_history_btn.click(
            fn=get_history_videos,
            inputs=save_dir_input,
            outputs=history_gallery,
        )

        def _format_toybox_comfy_status_md() -> str:
            tb = get_toybox_status_dict()
            cm = get_comfyui_status_dict()
            auth_tag = "🔒 已开启 Basic Auth 密码保护" if tb.get("password_protected") else "🌐 本地/无密码预览"
            tb_link = (
                f"[{tb['public_url']}]({tb['public_url']})"
                if tb.get("public_url")
                else ("运行中（本地后端已就绪）" if tb.get("is_running") else "未启动")
            )
            cm_link = (
                f"[{cm['public_url']}]({cm['public_url']})"
                if cm.get("public_url")
                else ("运行中" if cm.get("is_running") else "未启动")
            )
            return (
                f"**H3 Toybox 独立隧道 ({auth_tag})**：{tb_link}\n\n"
                f"**ComfyUI 原生画布**：{cm_link}\n\n"
                f"**就绪度**：专用节点 `{tb['nodes_ready']}/{tb['nodes_total']}` · "
                f"核心权重 `{tb['core_models_ready']}/{tb['core_models_total']}` · "
                f"9B改写 `{'✅' if tb['rewriter_ready'] else '⬜'}` · "
                f"P-H3生图 `{'✅' if tb['image_ready'] else '⬜'}` · 持久化 `{tb['persistence_label']}`\n\n"
                f"**最新状态**：{tb['status_message'] or cm['status_message']}"
            )

        def _webui_prepare_toybox(inc_rw: bool, inc_img: bool, hf_tok: str):
            _, msg = rs.resolve_callable("prepare_toybox_assets_core", prepare_toybox_assets_core)(
                include_rewriter=bool(inc_rw),
                include_image=bool(inc_img),
                hf_token=hf_tok or "",
                auto_start=False,
            )
            return _format_toybox_comfy_status_md(), msg

        def _webui_start_toybox(unload_ex: bool, hf_tok: str):
            _, msg = rs.resolve_callable("start_toybox_core", start_toybox_core)(
                unload_existing=bool(unload_ex),
                force_restart=False,
                hf_token=hf_tok or "",
            )
            return _format_toybox_comfy_status_md(), msg

        def _webui_start_comfyui(unload_ex: bool):
            _, msg = rs.resolve_callable("start_comfyui_core", start_comfyui_core)(unload_existing=bool(unload_ex), force_restart=False)
            return _format_toybox_comfy_status_md(), msg

        def _webui_stop_toybox():
            _, msg = rs.resolve_callable("stop_toybox_core", stop_toybox_core)(stop_comfyui_too=False)
            return _format_toybox_comfy_status_md(), msg

        toybox_prepare_btn.click(
            fn=_webui_prepare_toybox,
            inputs=[toybox_inc_rewriter_chk, toybox_inc_image_chk, hf_token_input],
            outputs=[toybox_status_md, run_status_box],
        )
        toybox_start_btn.click(
            fn=_webui_start_toybox,
            inputs=[toybox_unload_chk, hf_token_input],
            outputs=[toybox_status_md, run_status_box],
        )
        comfy_start_web_btn.click(
            fn=_webui_start_comfyui,
            inputs=[toybox_unload_chk],
            outputs=[toybox_status_md, run_status_box],
        )
        toybox_stop_btn.click(
            fn=_webui_stop_toybox,
            inputs=None,
            outputs=[toybox_status_md, run_status_box],
        )
        toybox_refresh_btn.click(
            fn=_format_toybox_comfy_status_md,
            inputs=None,
            outputs=toybox_status_md,
        )
        demo.load(fn=_format_toybox_comfy_status_md, inputs=None, outputs=toybox_status_md)

        def _webui_save_api_token(raw_tokens: str):
            # (shared state on runtime_state)
            toks = [t.strip() for t in re.split(r"[\s,;]+", raw_tokens or "") if t.strip()]
            if not toks:
                return gr.update(), "⚠️ API Token 不能为空！"
            with API_KEY_LOCK:
                rs.API_AUTH_KEY = toks[0]
                rs.EXTRA_API_TOKENS = toks[1:]
            return ", ".join(toks), f"✅ 已更新 API Token（当前共 {len(toks)} 个有效密钥，与 Web 登录认证互不冲突）"

        def _webui_regen_api_token():
            # (shared state on runtime_state)
            new_tok = f"sk-colab-{secrets.token_urlsafe(24)}"
            with API_KEY_LOCK:
                rs.API_AUTH_KEY = new_tok
                rs.EXTRA_API_TOKENS = []
            return new_tok, "✅ 已生成新的随机 API Token，请复制到外部客户端的 API Key 栏。"

        save_api_token_btn.click(
            fn=_webui_save_api_token,
            inputs=[api_token_input],
            outputs=[api_token_input, run_status_box],
        )
        regen_api_token_btn.click(
            fn=_webui_regen_api_token,
            inputs=None,
            outputs=[api_token_input, run_status_box],
        )

        disconnect_btn.click(
            fn=disconnect_colab_runtime,
            inputs=None,
            outputs=run_status_box,
        )

    return demo
