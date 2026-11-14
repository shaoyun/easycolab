"""EasyColab 服务端模块: toybox_service."""

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
from .comfyui_service import COMFY_STANDARD_MODEL_SUBDIRS, _COMFY_STATUS_CACHE, _build_and_install_comfy_preset_workflows, _download_comfy_model_entry, _ensure_comfyui_beginner_assets, _install_persisted_custom_node_requirements, _open_cloudflared_tunnel, _resolve_comfyui_persistence_paths, _safe_symlink_dir, start_comfyui_core, stop_comfyui_core, sync_comfyui_to_drive_core
from .media_pipeline import unload_model_from_vram_core
from .model_catalog import TOYBOX_CORE_MODELS, TOYBOX_CUSTOM_NODES, TOYBOX_ENHANCER_REPO_URL, TOYBOX_IMAGE_MODELS, TOYBOX_REPO_URL, TOYBOX_REWRITER_MODELS, fast_parallel_hf_download
from .runtime_state import COMFYUI_STATE, COMFY_BACKEND, LOADED_RUNTIMES, TOYBOX_BACKEND, TOYBOX_INGRESS, TOYBOX_LOCK, TOYBOX_STATE, _DEPLOY_PASSWORD, _IS_LOCAL_PREVIEW, _has_rclone_drive_remote, authenticated_ingress, torch, touch_activity

# ==============================================================================
# H3 Toybox 独立 Web 创作台（复用底层 ComfyUI 引擎、模型权重与 Google Drive 持久化）
# ==============================================================================
_TOYBOX_ADOPT_CHECKED = False
_TOYBOX_STATE_FILE = "/tmp/.easycolab_toybox_state.json"
def _resolve_toybox_runtime_paths() -> Dict[str, str]:
    comfy_root, persist_root, mode, label, is_drive = rs.resolve_callable("_resolve_comfyui_persistence_paths", _resolve_comfyui_persistence_paths)()
    base_root = "/content" if os.path.exists("/content") else os.getcwd()
    repo_dir = os.path.join(base_root, "minimax-h3-colab")
    app_dir = os.path.join(repo_dir, "toybox")
    enhancer_dir = os.path.join(base_root, "t8-enhancer")
    data_dir = os.path.join(persist_root, "H3Toybox")
    secrets_path = os.path.join(base_root, "h3toybox_secrets.json")
    return {
        "comfy_root": comfy_root,
        "persist_root": persist_root,
        "persistence_mode": mode,
        "persistence_label": label,
        "is_drive": is_drive,
        "base_root": base_root,
        "repo_dir": repo_dir,
        "app_dir": app_dir,
        "enhancer_dir": enhancer_dir,
        "data_dir": data_dir,
        "secrets_path": secrets_path,
    }

_ZHCONV_MODULE = None

_ZHCONV_TRIED_PIP = False

# 台湾习惯用语 / HTML 语言标记 -> 大陆简体地道 UI 表达（按长词优先匹配）
_TOYBOX_TERM_PRE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ('lang="zh-Hant"', 'lang="zh-CN"'),
    ("lang='zh-Hant'", "lang='zh-CN'"),
    (
        "這台還沒有安裝生圖模型：回到 Colab 第 ② 格勾選「INSTALL_IMAGE」再執行一次，然後重跑 ③",
        "当前环境尚未安装生图模型：请在 EasyColab 面板勾选「下载 P-H3 生图套件 (~10GB)」配齐权重并重启 Toybox",
    ),
    ("相容端點網址", "兼容接口地址"),
    ("相容端點", "兼容接口"),
    ("端點網址", "接口地址"),
    ("推測解碼", "投机解码"),
    ("執行階段", "运行时"),
    ("執行失敗", "运行失败"),
    ("進度監看", "进度监控"),
    ("雲端硬碟", "Google Drive"),
    ("點下面的方塊", "点击下方卡片"),
    ("點方塊", "点击卡片"),
    ("點開查看", "展开查看"),
    ("點開資料夾", "打开文件夹"),
    ("資料夾", "文件夹"),
    ("頭尾格", "首尾帧"),
    ("這一格", "这一帧"),
    ("第一格", "第一帧"),
    ("最後一格", "最后一帧"),
    ("首格", "首帧"),
    ("尾格", "尾帧"),
    ("一格", "一帧"),
    ("參考圖", "参考图"),
    ("繁體中文", "简体中文"),
    ("繁體", "简体"),
    ("影片", "视频"),
    ("音訊", "音频"),
    ("圖片", "图片"),
    ("檔案", "文件"),
    ("端點", "端点"),
    ("網址", "网址"),
    ("金鑰", "密钥"),
    ("解析度", "分辨率"),
    ("預設", "默认"),
    ("連線中", "连接中"),
    ("未連線", "未连接"),
    ("已連線", "已连接"),
    ("連不上", "连不上"),
    ("連線", "连接"),
    ("儲存", "保存"),
    ("已存的", "已保存的"),
    ("已存", "已保存"),
    ("專案", "项目"),
    ("伺服器", "服务器"),
    ("本機", "本地"),
    ("空陣列", "空数组"),
    ("陣列", "数组"),
    ("物件", "对象"),
    ("回覆", "回复"),
    ("回傳", "返回"),
    ("逾時", "超时"),
    ("點開", "展开"),
    ("不支援", "不支持"),
    ("支援", "支持"),
    ("載入", "加载"),
    ("送出中", "提交中"),
    ("送回", "打回"),
    ("排隊", "排队"),
    ("套用", "应用"),
    ("這台", "当前机器"),
    ("貼上", "粘贴"),
    ("隨機", "随机"),
    ("關閉", "关闭"),
    ("開啟", "开启"),
    ("進度", "进度"),
    ("總秒數", "总秒数"),
    ("秒數", "秒数"),
    ("長邊", "长边"),
    ("清單", "列表"),
    ("一隻", "一只"),
    ("抱著牠", "抱着它"),
    ("這裡", "这里"),
    ("那裡", "那里"),
    ("裡面的", "里面的"),
    ("裡面", "里面"),
    ("手上的書", "手里的书"),
    ("素材庫裡", "素材库里"),
    ("畫面裡", "画面里"),
)

_TOYBOX_TERM_POST_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("已保存存", "已保存"),
    ("头尾格", "首尾帧"),
    ("首格", "首帧"),
    ("尾格", "尾帧"),
)

# 离线单字兜底映射表（在未安装 zhconv 时仍可将 Toybox 全部繁体字符准确转为简体）
_TOYBOX_TRAD_TO_SIMP_PAIRS = (
    "丟丢 並并 係系 個个 們们 備备 傳传 儲储 兩两 內内 "
    "寫写 刪删 別别 則则 劃划 劇剧 動动 勢势 區区 參参 "
    "雙双 嗎吗 啟启 問问 單单 圖图 團团 圓圆 國国 場场 "
    "塊块 執执 墊垫 夾夹 學学 實实 寧宁 對对 導导 尋寻 "
    "將将 專专 屆届 層层 廣广 張张 強强 當当 錄录 徹彻 "
    "後后 從从 總总 態态 應应 戰战 戲戏 擁拥 擇择 擊击 "
    "敗败 敘叙 數数 斷断 於于 無无 時时 書书 會会 東东 "
    "構构 槍枪 樓楼 標标 樣样 機机 檔档 檢检 權权 決决 "
    "沒没 治治 測测 準准 演演 潛潜 濕湿 灑洒 為为 照照 "
    "畫画 當当 發发 監监 盡尽 盤盘 確确 碼码 稱称 窗窗 "
    "筆笔 等等 箱箱 節节 簾帘 籤签 約约 純纯 級级 細细 "
    "終终 組组 結结 絕绝 給给 經经 綱纲 網网 緊紧 線线 "
    "編编 縫缝 縮缩 繼继 續续 缺缺 義义 習习 聲声 聯联 "
    "聽听 舉举 舊旧 處处 虛虚 號号 衛卫 補补 裝装 "
    "製制 複复 覆覆 見见 規规 覽览 觀观 視视 解解 訊讯 "
    "記记 設设 詞词 話话 說说 誰谁 調调 請请 證证 讀读 "
    "變变 讓让 貨货 貼贴 資资 較较 載载 輕轻 輪轮 輯辑 "
    "輸输 轉转 這这 進进 連连 過过 適适 選选 還还 邊边 "
    "醫医 重重 金金 鈴铃 鋪铺 鏡镜 鐮镰 鑰钥 長长 門门 "
    "閉闭 開开 間间 關关 陣阵 除除 陰阴 陽阳 隊队 階阶 "
    "際际 隨随 隻只 集集 雙双 雲云 電电 需需 面面 響响 "
    "頁页 頂顶 順顺 須须 預预 頭头 題题 頻频 風风 餐餐 "
    "駐驻 驗验 體体 鬥斗 麼么 點点 凱凯 傘伞 來来 偵侦 "
    "傢家 優优 條条 構构 燈灯 貓猫 貞贞 豪豪 "
    "側侧 創创 員员 帶带 庫库 幾几 搖摇 撐撑 溫温 現现 環环 裡里 與与 著着 術术"
).split()

_TOYBOX_TRAD_TO_SIMP_MAP = str.maketrans({
    pair[0]: pair[1]
    for pair in _TOYBOX_TRAD_TO_SIMP_PAIRS
    if len(pair) == 2 and pair[0] != pair[1]
})

def _ensure_zhconv(allow_pip: bool = True):
    """懒加载 zhconv（在线时按需自动安装，离线环境返回 None 并退回内置精准词表）。"""
    global _ZHCONV_MODULE, _ZHCONV_TRIED_PIP
    if _ZHCONV_MODULE is not None:
        return _ZHCONV_MODULE
    try:
        import zhconv  # type: ignore
        _ZHCONV_MODULE = zhconv
        return _ZHCONV_MODULE
    except Exception:
        pass
    if allow_pip and not _ZHCONV_TRIED_PIP:
        _ZHCONV_TRIED_PIP = True
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "zhconv"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import zhconv  # type: ignore
            _ZHCONV_MODULE = zhconv
            return _ZHCONV_MODULE
        except Exception:
            pass
    return None

def _convert_toybox_text_to_zh_cn(text: str, allow_pip: bool = True) -> str:
    """将 H3 Toybox 文本（HTML/JS/CSS/Python）无损转为大陆简体中文，同时保护 @标签 兼容正则/字典。"""
    if not text:
        return text

    # 保护 MENTION_RE / MENTION_KIND 原生繁简英混合匹配定义，确保旧版 @圖片1 与新版 @图片1 均可用
    preserved_blocks: List[str] = []

    def _stash_match(m: re.Match) -> str:
        idx = len(preserved_blocks)
        preserved_blocks.append(m.group(0))
        return f"__EASYCOLAB_KEEP_{idx}__"

    out = re.sub(
        r"(?:const\s+MENTION_RE\s*=\s*[^\n]+;|const\s+MENTION_KIND\s*=\s*\{[^\n]+\};|MENTION_RE\s*=\s*re\.compile\([^\n]+\)|MENTION_KIND\s*=\s*\{[^}]+\}|#\s*@圖片1[^\n]*)",
        _stash_match,
        text,
    )

    for src, dst in _TOYBOX_TERM_PRE_PAIRS:
        if src in out:
            out = out.replace(src, dst)

    zhconv_mod = _ensure_zhconv(allow_pip=allow_pip)
    if zhconv_mod is not None:
        try:
            out = zhconv_mod.convert(out, "zh-cn")
        except Exception:
            out = out.translate(_TOYBOX_TRAD_TO_SIMP_MAP)
    else:
        out = out.translate(_TOYBOX_TRAD_TO_SIMP_MAP)

    for src, dst in _TOYBOX_TERM_POST_PAIRS:
        if src in out:
            out = out.replace(src, dst)

    for idx, orig in enumerate(preserved_blocks):
        out = out.replace(f"__EASYCOLAB_KEEP_{idx}__", orig)
    return out

def _localize_toybox_to_zh_cn(app_dir: str, allow_pip: bool = True) -> int:
    """幂等将 H3 Toybox 前端界面与后端提示文案内化为大陆简体中文，并在写回前校验 Python 语法。"""
    if not app_dir or not os.path.isdir(app_dir):
        return 0
    targets = [
        os.path.join(app_dir, "static", "index.html"),
        os.path.join(app_dir, "static", "app.js"),
        os.path.join(app_dir, "static", "style.css"),
        os.path.join(app_dir, "server.py"),
    ]
    changed = 0
    for fpath in targets:
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception:
            continue
        converted = _convert_toybox_text_to_zh_cn(raw, allow_pip=allow_pip)
        if converted == raw:
            continue
        if fpath.endswith(".py"):
            try:
                import ast as _ast
                _ast.parse(converted, filename=os.path.basename(fpath))
            except SyntaxError as se:
                print(f"[H3 Toybox] 警告: 简体本地化后 {os.path.basename(fpath)} 语法校验未通过，跳过写回: {se}")
                continue
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(converted)
        changed += 1
    if changed:
        print(f"[H3 Toybox] 已将 {changed} 个界面/后端文件本地化为简体中文 (zh-CN)。")
    return changed

def _clone_or_pull_git_repo(url: str, dest_dir: str, reset_local: bool = False) -> bool:
    """若目标目录不存在则浅克隆仓库（返回 True 表示新安装），存在则静默 fast-forward 更新。"""
    if not os.path.isdir(dest_dir):
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, dest_dir], check=True)
        return True
    if reset_local:
        subprocess.run(["git", "-C", dest_dir, "checkout", "--", "."], check=False)
    subprocess.run(["git", "-C", dest_dir, "pull", "--ff-only"], check=False)
    return False

def _ensure_toybox_repos_and_nodes(hf_token: str = "") -> Tuple[bool, str, bool]:
    """确保 H3 Toybox 源码、3 组专用 ComfyUI 节点、提示词指南与 PyAV 依赖就绪，返回 (ok, msg, newly_installed_nodes)。"""
    paths = _resolve_toybox_runtime_paths()
    comfy_root = paths["comfy_root"]
    persist_root = paths["persist_root"]
    app_dir = paths["app_dir"]
    newly_installed_nodes = False

    if not os.path.isdir(os.path.join(comfy_root, "custom_nodes")):
        os.makedirs(os.path.join(comfy_root, "custom_nodes"), exist_ok=True)

    # 1. 拉取 H3 Toybox 本体与外置导演增强技能目录，并将 Toybox 界面/提示转为大陆简体中文
    _clone_or_pull_git_repo(TOYBOX_REPO_URL, paths["repo_dir"], reset_local=True)
    _localize_toybox_to_zh_cn(app_dir)
    _clone_or_pull_git_repo(TOYBOX_ENHANCER_REPO_URL, paths["enhancer_dir"])

    # 2. 安装/检查 Toybox 所需的 3 个 ComfyUI 自定义节点（仅在缺失时浅克隆，避免覆盖已有依赖）
    for node_name, node_url in TOYBOX_CUSTOM_NODES:
        dst_node = os.path.join(comfy_root, "custom_nodes", node_name)
        persist_node = os.path.join(persist_root, "custom_nodes", node_name)
        if not os.path.isdir(dst_node) and os.path.isdir(persist_node):
            if shutil.which("rsync"):
                subprocess.run(["rsync", "-a", f"{persist_node}/", f"{dst_node}/"], check=False)
            else:
                shutil.copytree(persist_node, dst_node, dirs_exist_ok=True)
        if not os.path.isdir(dst_node):
            subprocess.run(["git", "clone", "--depth", "1", node_url, dst_node], check=True)
            newly_installed_nodes = True

    # 3. 补齐 semantic_bridge 与模型子目录软链（供 BUNNY_H3_ActionLogic_Bridge 等权重共享）
    for sub in COMFY_STANDARD_MODEL_SUBDIRS:
        _safe_symlink_dir(os.path.join(persist_root, "models", sub), os.path.join(comfy_root, "models", sub))

    # 4. 确保 PyAV (av) 与 ffmpeg 依赖存在（server.py 提取视频帧与拼接尾帧需要）
    try:
        import av as _av_chk  # noqa: F401
    except Exception:
        print("[H3 Toybox] 正在补装 PyAV (av) 音视频处理依赖...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av"], check=True)
    if not shutil.which("ffmpeg") and shutil.which("apt-get"):
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=False)
    cuda_ver = str(getattr(torch.version, "cuda", "") or "")
    if torch.cuda.is_available() and cuda_ver and not cuda_ver.startswith("13"):
        print(f"[H3 Toybox] 提示: 当前 PyTorch CUDA={cuda_ver}，如加载 int8 ConvRot 权重报错，建议升级至 cu130 PyTorch。")

    # 5. 安装/补齐节点 requirements.txt（保留 PyTorch/CUDA ABI）
    _install_persisted_custom_node_requirements(comfy_root)

    # 6. 下载 H3 官方中英文改写指南到 toybox/guides/
    guides_dir = os.path.join(app_dir, "guides")
    os.makedirs(guides_dir, exist_ok=True)
    token = (hf_token or "").strip() or os.environ.get("HF_TOKEN", "").strip()
    for guide_name in ("VIDEO_PROMPT_WRITING_GUIDE_ref_en.md", "VIDEO_PROMPT_WRITING_GUIDE_base_en.md"):
        guide_dst = os.path.join(guides_dir, guide_name)
        if os.path.isfile(guide_dst) and os.path.getsize(guide_dst) > 128:
            continue
        try:
            dl_g = rs.resolve_callable("fast_parallel_hf_download", fast_parallel_hf_download)(
                repo_id="MiniMaxAI/MiniMax-H3",
                filename=f"docs/{guide_name}",
                dest_dir=guides_dir,
                hf_token=token,
                num_connections=2,
            )
            if os.path.abspath(dl_g) != os.path.abspath(guide_dst) and os.path.isfile(dl_g):
                shutil.move(dl_g, guide_dst)
            docs_tmp = os.path.join(guides_dir, "docs")
            if os.path.isdir(docs_tmp):
                shutil.rmtree(docs_tmp, ignore_errors=True)
        except Exception as ge:
            print(f"[H3 Toybox] 下载官方指南 {guide_name} 提示: {ge}")

    _COMFY_STATUS_CACHE["ts"] = 0.0
    return True, "H3 Toybox 源码、节点与依赖已就绪", newly_installed_nodes

def _configure_toybox_runtime_files(
    paths: Dict[str, str],
    comfy_port: int,
    save_api_keys: bool = False,
) -> Tuple[Dict[str, Any], str, str]:
    """将 H3 Toybox 的 config.json 与 ComfyUI/output/H3Toybox 软链精准对接至 EasyColab 底层工作台。"""
    comfy_root = paths["comfy_root"]
    persist_root = paths["persist_root"]
    data_dir = paths["data_dir"]
    videos_dir = os.path.join(data_dir, "videos")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(os.path.join(comfy_root, "input"), exist_ok=True)

    # 对接 ComfyUI/output/H3Toybox -> <persist_root>/H3Toybox/videos（迁移已有成品避免丢档）
    out_dir = os.path.join(comfy_root, "output", "H3Toybox")
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    legacy_dirs: List[str] = []
    if os.path.islink(out_dir):
        prev = os.path.realpath(out_dir)
        if os.path.abspath(prev) != os.path.abspath(videos_dir):
            legacy_dirs.append(prev)
            os.unlink(out_dir)
    elif os.path.isdir(out_dir):
        legacy_dirs.append(out_dir)
    for old_d in legacy_dirs:
        if os.path.isdir(old_d) and os.path.abspath(old_d) != os.path.abspath(videos_dir):
            for fn in os.listdir(old_d):
                src_f = os.path.join(old_d, fn)
                dst_f = os.path.join(videos_dir, fn)
                if os.path.isfile(src_f) and not os.path.exists(dst_f):
                    try:
                        shutil.move(src_f, dst_f)
                    except Exception:
                        pass
            if old_d == out_dir:
                shutil.rmtree(out_dir, ignore_errors=True)
    if not os.path.exists(out_dir):
        os.symlink(videos_dir, out_dir)

    cfg_path = os.path.join(data_dir, "config.json")
    try:
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as rf:
                cfg = json.load(rf)
        else:
            cfg = {}
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}

    cfg.update({
        "comfy_url": f"http://127.0.0.1:{int(comfy_port)}",
        "comfy_input_dir": os.path.join(comfy_root, "input"),
        "skills_dir": os.path.join(paths["enhancer_dir"], "directional_skills"),
        "free_after_rewrite": True,
    })
    cfg.pop("guides", None)

    agent_cfg = dict(cfg.get("agent") or {})
    if not agent_cfg.get("provider"):
        agent_cfg["provider"] = "gemini"
    cfg["agent"] = agent_cfg

    ph3_installed = all(
        os.path.isfile(os.path.join(persist_root, "models", sub, fname))
        for sub, fname, _, _, _ in TOYBOX_IMAGE_MODELS
    )
    img_cfg = dict(cfg.get("image") or {})
    img_cfg["engine"] = "ph3"
    img_cfg["installed"] = ph3_installed
    cfg["image"] = img_cfg

    with open(cfg_path, "w", encoding="utf-8") as cf:
        json.dump(cfg, cf, ensure_ascii=False, indent=1)

    secrets_path = os.path.join(data_dir, "secrets.json") if save_api_keys else paths["secrets_path"]
    return cfg, data_dir, secrets_path

def get_toybox_status_dict() -> Dict[str, Any]:
    paths = _resolve_toybox_runtime_paths()
    comfy_root = paths["comfy_root"]
    persist_root = paths["persist_root"]
    app_dir = paths["app_dir"]
    repo_installed = os.path.isfile(os.path.join(app_dir, "server.py"))
    nodes_ready = sum(
        1 for name, _ in TOYBOX_CUSTOM_NODES
        if os.path.isdir(os.path.join(comfy_root, "custom_nodes", name))
        or os.path.isdir(os.path.join(persist_root, "custom_nodes", name))
    )
    core_ready = sum(
        1 for sub, fname, _, _, _ in TOYBOX_CORE_MODELS
        if os.path.isfile(os.path.join(persist_root, "models", sub, fname))
        and os.path.getsize(os.path.join(persist_root, "models", sub, fname)) > 1024 * 1024
    )
    rewriter_ready = all(
        os.path.isfile(os.path.join(persist_root, "models", sub, fname))
        and os.path.getsize(os.path.join(persist_root, "models", sub, fname)) > 1024 * 1024
        for sub, fname, _, _, _ in TOYBOX_REWRITER_MODELS
    )
    image_ready = all(
        os.path.isfile(os.path.join(persist_root, "models", sub, fname))
        and os.path.getsize(os.path.join(persist_root, "models", sub, fname)) > 1024 * 1024
        for sub, fname, _, _, _ in TOYBOX_IMAGE_MODELS
    )
    password_protected = bool(
        authenticated_ingress is not None
        and (_DEPLOY_PASSWORD or (sys.stdin.isatty() and not _IS_LOCAL_PREVIEW))
    )
    return {
        "is_running": bool(TOYBOX_STATE.get("is_running")),
        "is_starting": bool(TOYBOX_STATE.get("is_starting")),
        "is_preparing_assets": bool(TOYBOX_STATE.get("is_preparing_assets")),
        "progress": float(TOYBOX_STATE.get("progress") or 0.0),
        "status_message": str(TOYBOX_STATE.get("status_message") or ""),
        "public_url": str(TOYBOX_STATE.get("public_url") or ""),
        "local_port": int(TOYBOX_STATE.get("local_port") or TOYBOX_BACKEND),
        "ingress_port": int(TOYBOX_STATE.get("ingress_port") or TOYBOX_INGRESS),
        "password_protected": password_protected,
        "repo_installed": repo_installed,
        "nodes_ready": nodes_ready,
        "nodes_total": len(TOYBOX_CUSTOM_NODES),
        "core_models_ready": core_ready,
        "core_models_total": len(TOYBOX_CORE_MODELS),
        "rewriter_ready": rewriter_ready,
        "image_ready": image_ready,
        "data_dir": paths["data_dir"],
        "persistence_label": paths["persistence_label"],
        "comfyui_running": bool(COMFYUI_STATE.get("is_running")),
        "comfyui_public_url": str(COMFYUI_STATE.get("public_url") or ""),
    }

def _try_adopt_running_toybox():
    """若 app.py 重启前实例上已有运行中的 H3 Toybox 与独立隧道，自动无缝接管。"""
    global _TOYBOX_ADOPT_CHECKED
    if _TOYBOX_ADOPT_CHECKED or TOYBOX_STATE.get("is_running") or TOYBOX_STATE.get("is_starting"):
        return
    _TOYBOX_ADOPT_CHECKED = True

    port = int(TOYBOX_STATE.get("local_port") or TOYBOX_BACKEND)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/api/config", timeout=1.0) as resp:
            if resp.status != 200:
                return
    except Exception:
        return

    pub_url = ""
    if os.path.isfile(_TOYBOX_STATE_FILE):
        try:
            st_obj = json.load(open(_TOYBOX_STATE_FILE, "r", encoding="utf-8"))
            pub_url = str(st_obj.get("public_url") or "").strip()
        except Exception:
            pub_url = ""

    if not pub_url:
        try:
            tunnel_proc, pub_url = _open_cloudflared_tunnel(TOYBOX_INGRESS, timeout_s=12.0)
            if pub_url:
                TOYBOX_STATE["tunnel_proc"] = tunnel_proc
            else:
                tunnel_proc.terminate()
        except Exception:
            pass

    TOYBOX_STATE["is_running"] = True
    TOYBOX_STATE["public_url"] = pub_url
    TOYBOX_STATE["progress"] = 1.0
    TOYBOX_STATE["status_message"] = "已自动接管在线 H3 Toybox 创作台（带密码独立隧道）"
    if pub_url:
        try:
            with open(_TOYBOX_STATE_FILE, "w", encoding="utf-8") as sf:
                json.dump({"public_url": pub_url, "port": port}, sf)
        except Exception:
            pass
    print(f"[H3 Toybox] 已接管正在运行的服务: {pub_url or f'127.0.0.1:{port}'}")

def _poll_local_toybox_runtime_quiet():
    proc = TOYBOX_STATE.get("toybox_proc")
    if proc is not None and proc.poll() is not None:
        TOYBOX_STATE["is_running"] = False
        if not TOYBOX_STATE.get("is_starting"):
            TOYBOX_STATE["status_message"] = f"H3 Toybox 进程已退出 (code={proc.returncode})，可点击重新启动"
        TOYBOX_STATE["toybox_proc"] = None
        return
    if not TOYBOX_STATE.get("is_running"):
        _try_adopt_running_toybox()

def prepare_toybox_assets_core(
    include_rewriter: bool = True,
    include_image: bool = True,
    hf_token: str = "",
    auto_start: bool = False,
) -> Tuple[bool, str]:
    """一键配齐 H3 Toybox 源码、ComfyUI 自定义节点与模型权重（自动跳过与现有 ComfyUI 包共用的底座文件）。"""
    touch_activity()
    if TOYBOX_STATE.get("is_preparing_assets") or COMFYUI_STATE.get("is_downloading_weight"):
        return False, "当前已有模型或插件正在高速下载配齐中，请稍候..."

    paths = _resolve_toybox_runtime_paths()
    comfy_root = paths["comfy_root"]
    persist_root = paths["persist_root"]
    mode = paths["persistence_mode"]
    token = (hf_token or "").strip() or os.environ.get("HF_TOKEN", "").strip()

    items: List[Tuple[str, str, str, str, float]] = list(TOYBOX_CORE_MODELS)
    if include_rewriter:
        items.extend(TOYBOX_REWRITER_MODELS)
    if include_image:
        items.extend(TOYBOX_IMAGE_MODELS)
    total_m = len(items)

    TOYBOX_STATE["is_preparing_assets"] = True
    TOYBOX_STATE["progress"] = 0.03
    TOYBOX_STATE["status_message"] = "正在拉取 H3 Toybox 源码与 ComfyUI 专用节点（SelfLift / Semantic Bridge / Motion Context）..."

    try:
        pull_flag = "/tmp/.easycolab_comfy_drive_pulled"
        if mode == "rclone_drive_sync" and _has_rclone_drive_remote() and not os.path.exists(pull_flag):
            TOYBOX_STATE["status_message"] = "正在先从 Google Drive 检查并恢复已持久化的 ComfyUI / Toybox 数据..."
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
        _, _, newly_installed_nodes = _ensure_toybox_repos_and_nodes(hf_token=token)

        for idx, (sub, fname, repo_id, repo_rel_path, approx_gb) in enumerate(items, start=1):
            touch_activity()
            TOYBOX_STATE["progress"] = round(0.08 + 0.80 * ((idx - 1) / float(max(1, total_m))), 3)
            TOYBOX_STATE["status_message"] = (
                f"[{idx}/{total_m}] 正在校验/8线程下载 {os.path.basename(fname)} (约 {approx_gb:.1f} GB)..."
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
            action_tag = "下载完成" if downloaded else "共享底座已存在跳过"
            print(f"[H3 Toybox] ({idx}/{total_m}) {action_tag}: models/{sub}/{fname}")

        _build_and_install_comfy_preset_workflows(persist_root, ["minimax_h3_toybox"])
        if mode == "rclone_drive_sync":
            TOYBOX_STATE["progress"] = 0.94
            TOYBOX_STATE["status_message"] = "正在同步 H3 Toybox 节点与权重至 Google Drive..."
            rs.resolve_callable("sync_comfyui_to_drive_core", sync_comfyui_to_drive_core)(quiet=True)

        _COMFY_STATUS_CACHE["ts"] = 0.0
        TOYBOX_STATE["progress"] = 1.0
        ok_msg = f"H3 Toybox 节点与 {total_m} 项权重已全部配齐（{paths['persistence_label']}）！"
        TOYBOX_STATE["status_message"] = ok_msg

        if newly_installed_nodes and COMFYUI_STATE.get("is_running") and not auto_start:
            print("[H3 Toybox] 检测到新安装 ComfyUI 节点，正在重启 ComfyUI 以加载新节点...")
            rs.resolve_callable("start_comfyui_core", start_comfyui_core)(unload_existing=False, force_restart=True)
    except Exception as e:
        err = f"配齐 H3 Toybox 资源失败: {e}"
        TOYBOX_STATE["progress"] = 0.0
        TOYBOX_STATE["status_message"] = err
        print(f"[H3 Toybox] {err}")
        return False, err
    finally:
        TOYBOX_STATE["is_preparing_assets"] = False
        touch_activity()

    if auto_start:
        return rs.resolve_callable("start_toybox_core", start_toybox_core)(unload_existing=True, force_restart=False, hf_token=token)
    return True, ok_msg

def start_toybox_core(
    unload_existing: bool = True,
    force_restart: bool = False,
    hf_token: str = "",
    save_api_keys: bool = False,
) -> Tuple[bool, str]:
    """启动 H3 Toybox 后端并建立带 Basic Auth 密码保护的独立 Cloudflare 隧道（复用同一 ComfyUI 引擎）。"""
    touch_activity()
    _try_adopt_running_toybox()
    if TOYBOX_STATE.get("is_starting"):
        return True, "H3 Toybox 正在部署启动中，请稍候..."
    if TOYBOX_STATE.get("is_running") and TOYBOX_STATE.get("public_url") and not force_restart:
        return True, f"H3 Toybox 已在运行: {TOYBOX_STATE['public_url']}"

    if force_restart and TOYBOX_STATE.get("is_running"):
        rs.resolve_callable("stop_toybox_core", stop_toybox_core)(stop_comfyui_too=False)

    TOYBOX_STATE["is_starting"] = True
    TOYBOX_STATE["progress"] = 0.08
    TOYBOX_STATE["status_message"] = "正在检查 GPU/CUDA 环境与 H3 Toybox 源码及 ComfyUI 节点..."

    try:
        if torch.cuda.is_available():
            try:
                cap = torch.cuda.get_device_capability(0)
                if cap < (8, 0):
                    raise RuntimeError(
                        f"当前 GPU 架构算力 {cap} 不支持 H3 int8 ConvRot 权重，请在 Colab 中选择 L4 / A100 / RTX PRO 6000。"
                    )
            except RuntimeError:
                raise
            except Exception:
                pass

        token = (hf_token or "").strip() or os.environ.get("HF_TOKEN", "").strip()
        _, _, newly_installed_nodes = _ensure_toybox_repos_and_nodes(hf_token=token)
        paths = _resolve_toybox_runtime_paths()

        # 确保共用底层 ComfyUI 引擎已启动；若刚刚补充安装了新节点则强制重启 ComfyUI 使新节点生效
        TOYBOX_STATE["progress"] = 0.35
        if not COMFYUI_STATE.get("is_running"):
            TOYBOX_STATE["status_message"] = "正在拉起共享底层 ComfyUI 引擎（共驻工作台 + 持久化软链）..."
            ok_c, msg_c = rs.resolve_callable("start_comfyui_core", start_comfyui_core)(unload_existing=unload_existing, force_restart=False)
            if not ok_c:
                raise RuntimeError(f"底层 ComfyUI 引擎启动失败: {msg_c}")
        elif newly_installed_nodes:
            TOYBOX_STATE["status_message"] = "检测到新装 H3 Toybox 节点，正在重启共享 ComfyUI 引擎以注册节点..."
            ok_c, msg_c = rs.resolve_callable("start_comfyui_core", start_comfyui_core)(unload_existing=unload_existing, force_restart=True)
            if not ok_c:
                raise RuntimeError(f"底层 ComfyUI 引擎重启失败: {msg_c}")
        elif unload_existing and LOADED_RUNTIMES:
            rs.resolve_callable("unload_model_from_vram_core", unload_model_from_vram_core)("all")

        comfy_port = int(COMFYUI_STATE.get("local_port") or COMFY_BACKEND)
        toybox_port = int(TOYBOX_STATE.get("local_port") or TOYBOX_BACKEND)

        TOYBOX_STATE["progress"] = 0.62
        TOYBOX_STATE["status_message"] = "正在写入 H3 Toybox 配置并对接 ComfyUI 输入/输出目录..."
        _, data_dir, secrets_path = _configure_toybox_runtime_files(
            paths, comfy_port=comfy_port, save_api_keys=save_api_keys
        )

        # 清理残留 Toybox 进程与对应 8190 隧道
        subprocess.run(["pkill", "-f", "toybox/server.py"], check=False)
        subprocess.run(["pkill", "-f", f"cloudflared.*{TOYBOX_INGRESS}"], check=False)
        time.sleep(0.4)

        child_env = os.environ.copy()
        child_env["H3TOYBOX_PORT"] = str(toybox_port)
        child_env["H3TOYBOX_DATA"] = data_dir
        child_env["H3TOYBOX_SECRETS"] = secrets_path

        TOYBOX_STATE["progress"] = 0.76
        TOYBOX_STATE["status_message"] = f"正在启动 H3 Toybox Web 后端 (127.0.0.1:{toybox_port})..."
        toybox_proc = subprocess.Popen(
            [sys.executable, "-u", "server.py"],
            cwd=paths["app_dir"],
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        TOYBOX_STATE["toybox_proc"] = toybox_proc

        def _forward_toybox_logs(proc: subprocess.Popen):
            try:
                if proc.stdout is None:
                    return
                for raw_ln in proc.stdout:
                    ln = raw_ln.rstrip()
                    if ln:
                        print(f"[H3 Toybox] {ln}")
            except Exception:
                pass

        threading.Thread(target=_forward_toybox_logs, args=(toybox_proc,), daemon=True).start()

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        ready = False
        for _ in range(45):
            if toybox_proc.poll() is not None:
                raise RuntimeError(f"H3 Toybox 启动提前退出 (code={toybox_proc.returncode})")
            try:
                with opener.open(f"http://127.0.0.1:{toybox_port}/api/config", timeout=1.5) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(1.0)

        if not ready:
            raise RuntimeError("等待 H3 Toybox 本地后端端口就绪超时。")

        TOYBOX_STATE["progress"] = 0.90
        TOYBOX_STATE["status_message"] = (
            f"正在开通带登录密码保护的独立 Cloudflare 隧道 (Ingress :{TOYBOX_INGRESS} -> :{toybox_port})..."
        )
        tunnel_proc, pub_url = _open_cloudflared_tunnel(TOYBOX_INGRESS, timeout_s=22.5)
        TOYBOX_STATE["tunnel_proc"] = tunnel_proc

        if not pub_url:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass
            TOYBOX_STATE["tunnel_proc"] = None
            raise RuntimeError("未能获取 H3 Toybox Cloudflare 公网隧道地址，请重试。")

        TOYBOX_STATE["public_url"] = pub_url
        TOYBOX_STATE["is_running"] = True
        TOYBOX_STATE["progress"] = 1.0
        try:
            with open(_TOYBOX_STATE_FILE, "w", encoding="utf-8") as sf:
                json.dump({"public_url": pub_url, "port": toybox_port}, sf)
        except Exception:
            pass
        ok_msg = f"H3 Toybox 已就绪（独立密码隧道 · 复用 ComfyUI 引擎 · 存档：{paths['persistence_label']}）"
        TOYBOX_STATE["status_message"] = ok_msg
        print(f"[H3 Toybox] 独立带密码公网隧道已开通: {pub_url}")
        return True, ok_msg
    except Exception as e:
        err_msg = f"H3 Toybox 启动出错: {e}"
        TOYBOX_STATE["status_message"] = err_msg
        TOYBOX_STATE["progress"] = 0.0
        print(f"[H3 Toybox] {err_msg}")
        return False, err_msg
    finally:
        TOYBOX_STATE["is_starting"] = False
        touch_activity()

def stop_toybox_core(stop_comfyui_too: bool = False) -> Tuple[bool, str]:
    with TOYBOX_LOCK:
        for key in ("tunnel_proc", "toybox_proc"):
            proc = TOYBOX_STATE.get(key)
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                TOYBOX_STATE[key] = None
        subprocess.run(["pkill", "-f", "toybox/server.py"], check=False)
        subprocess.run(["pkill", "-f", f"cloudflared.*{TOYBOX_INGRESS}"], check=False)
        if os.path.exists(_TOYBOX_STATE_FILE):
            try:
                os.remove(_TOYBOX_STATE_FILE)
            except Exception:
                pass
        if os.path.isdir("/content/ComfyUI") or COMFYUI_STATE.get("is_running"):
            rs.resolve_callable("sync_comfyui_to_drive_core", sync_comfyui_to_drive_core)(quiet=True)
        TOYBOX_STATE["is_running"] = False
        TOYBOX_STATE["is_starting"] = False
        TOYBOX_STATE["public_url"] = ""
        TOYBOX_STATE["progress"] = 0.0
        TOYBOX_STATE["status_message"] = "H3 Toybox 已停止（素材库与历史作品已保存，底层 ComfyUI 状态保留）"
    if stop_comfyui_too:
        rs.resolve_callable("stop_comfyui_core", stop_comfyui_core)(sync_before_stop=True)
    return True, TOYBOX_STATE["status_message"]
