"""EasyColab 服务端模块: llm_runtime."""

from PIL import Image
from typing import Any, Dict, List, Optional, Sequence, Tuple
import atexit
import base64
import gc
import io
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid

from . import runtime_state as rs
from .runtime_state import torch
from .model_catalog import MODEL_CATALOG
from .qwen_kernels import GenerationAbortedError, _clamp_context_length, _make_abort_stopping_criteria, _make_timed_token_streamer
from .runtime_state import ABORT_GENERATION_EVENT, GPU_LOCK, JOBS, LOADED_RUNTIMES, MODEL_DOWNLOAD_STATE, _JOB_CANCEL_EVENTS, _load_pil_from_url_or_b64, _make_noproxy_opener, decode_base64_image, torch, touch_activity
from .schemas import GenerateVideoRequest

def _is_llm_model_id(model_id: Optional[str]) -> bool:
    if not model_id:
        return False
    return MODEL_CATALOG.get(model_id, {}).get("modality") == "llm_vlm"

def _stop_gguf_server_bundle(bundle: Optional[Dict[str, Any]]):
    if not bundle or bundle.get("backend") != "gguf":
        return
    proc = bundle.pop("server_proc", None)
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        except Exception:
            pass
        time.sleep(0.25)

def _cleanup_all_gguf_servers():
    for rt in list(LOADED_RUNTIMES.values()):
        _stop_gguf_server_bundle(rt.get("llm_bundle"))
    _stop_gguf_server_bundle(rs.LLM_BUNDLE)

atexit.register(_cleanup_all_gguf_servers)

if os.path.exists("/content"):
    subprocess.run(["pkill", "-f", "llama-server"], check=False)

def _build_llama_ld_env(install_dir: str) -> Dict[str, str]:
    """收集 llama.cpp 动态库目录 + PyTorch 内置 NVIDIA CUDA Runtime/cuBLAS 目录"""
    env = os.environ.copy()
    ld_dirs: List[str] = []
    if os.path.isdir(install_dir):
        for root, _, files in os.walk(install_dir):
            if any(f == "llama-server" or ".so" in f for f in files):
                if root not in ld_dirs:
                    ld_dirs.append(root)

    try:
        import site
        site_roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    except Exception:
        site_roots = []
    site_roots.append(os.path.dirname(os.path.dirname(torch.__file__)))

    for sp in site_roots:
        nv_root = os.path.join(sp, "nvidia")
        if os.path.isdir(nv_root):
            for sub in os.listdir(nv_root):
                lib_d = os.path.join(nv_root, sub, "lib")
                if os.path.isdir(lib_d) and lib_d not in ld_dirs:
                    ld_dirs.append(lib_d)

    for sys_lib in ("/usr/lib64-nvidia", "/usr/local/nvidia/lib64", "/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"):
        if os.path.isdir(sys_lib) and sys_lib not in ld_dirs:
            ld_dirs.append(sys_lib)

    prev_ld = env.get("LD_LIBRARY_PATH", "")
    if prev_ld:
        ld_dirs.append(prev_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_dirs)
    return env

def _download_file_to_path(url: str, dst_path: str, desc: str = ""):
    req = urllib.request.Request(url, headers={"User-Agent": "EasyColab-Engine/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dst_path, "wb") as wf:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            wf.write(chunk)
            done += len(chunk)
            if desc and total > 0:
                pct = int(done * 100 / total)
                MODEL_DOWNLOAD_STATE["message"] = f"{desc} ({done/(1024**2):.0f}/{total/(1024**2):.0f} MB · {pct}%)"

def _ensure_llama_server_binary() -> Tuple[str, Dict[str, str]]:
    """
    自动安装/校验官方 ggml-org/llama.cpp Linux x64 CUDA 12.8 预编译 llama-server + cudart 二进制。
    固定使用 ubuntu-cuda-12.x（NVIDIA 驱动向下兼容 CUDA 12.x/13.0/13.5+ 全系驱动，避免 CUDA 13.5 PTX 在 Colab 13.0 驱动上报 unsupported toolchain）。
    """
    install_dir = (
        "/content/llama_cpp_cuda"
        if os.path.exists("/content")
        else os.path.join(os.getcwd(), "llama_cpp_cuda")
    )
    os.makedirs(install_dir, exist_ok=True)

    def _find_server_bin() -> Optional[str]:
        for root, _, files in os.walk(install_dir):
            if "llama-server" in files:
                cand = os.path.join(root, "llama-server")
                try:
                    os.chmod(cand, 0o755)
                except Exception:
                    pass
                return cand
        return None

    def _has_cudart12() -> bool:
        for _, _, files in os.walk(install_dir):
            if any(f.startswith("libcudart.so.12") or f.startswith("libcublas.so.12") for f in files):
                return True
        return False

    def _verify_server_bin(bin_p: str, run_env: Dict[str, str]) -> Tuple[bool, str]:
        try:
            proc = subprocess.run(
                [bin_p, "--version"],
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=15,
            )
            if proc.returncode != 0:
                return False, (proc.stdout or "").strip()
            ver_txt = (proc.stdout or "").strip()
            if torch.cuda.is_available():
                dev_proc = subprocess.run(
                    [bin_p, "--list-devices"],
                    env=run_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=15,
                )
                dev_txt = (dev_proc.stdout or "").strip()
                if dev_proc.returncode != 0 or "CUDA0" not in dev_txt:
                    return False, f"CUDA 设备未挂载 ({dev_txt[:200]})"
            return True, ver_txt
        except Exception as ex:
            return False, str(ex)

    existing_bin = _find_server_bin()
    if existing_bin and _has_cudart12():
        env = _build_llama_ld_env(install_dir)
        ok, _ = _verify_server_bin(existing_bin, env)
        if ok:
            return existing_bin, env

    shutil.rmtree(install_dir, ignore_errors=True)
    os.makedirs(install_dir, exist_ok=True)

    cuda_prefix = "ubuntu-cuda-12."
    fallback_tag = "b12180"
    main_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/llama-{fallback_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"
    cudart_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/cudart-llama-{fallback_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"

    try:
        api_req = urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "EasyColab-Engine/1.0"},
        )
        with urllib.request.urlopen(api_req, timeout=10) as resp:
            releases = json.loads(resp.read().decode("utf-8"))
        for rel in releases:
            tag = str(rel.get("tag_name") or "")
            if not tag.startswith("b"):
                continue
            found_main, found_rt = None, None
            for asset in rel.get("assets") or []:
                aname = str(asset.get("name") or "")
                aurl = str(asset.get("browser_download_url") or "")
                if not aurl or not aname.endswith("-x64.tar.gz"):
                    continue
                if aname.startswith("llama-") and cuda_prefix in aname:
                    found_main = aurl
                elif aname.startswith("cudart-llama-") and cuda_prefix in aname:
                    found_rt = aurl
            if found_main:
                main_url = found_main
                if found_rt:
                    cudart_url = found_rt
                break
    except Exception as api_err:
        print(f"[GGUF 引擎] GitHub Releases 查询跳过，使用稳定镜像版本 {fallback_tag} ({api_err})")

    print(f"[GGUF 引擎] 正在下载官方预编译 llama.cpp CUDA 12 推理后端: {os.path.basename(main_url)} ...")
    tar_path = os.path.join(install_dir, "llama_cuda_bin.tar.gz")
    _download_file_to_path(main_url, tar_path, desc="正在下载 llama.cpp CUDA 引擎")
    subprocess.check_call(["tar", "-xzf", tar_path, "-C", install_dir])
    try:
        os.remove(tar_path)
    except Exception:
        pass

    if cudart_url:
        print(f"[GGUF 引擎] 正在下载配套 CUDA 12 Runtime 动态库 ({os.path.basename(cudart_url)})...")
        rt_tar = os.path.join(install_dir, "cudart_bin.tar.gz")
        _download_file_to_path(cudart_url, rt_tar, desc="正在下载 CUDA Runtime 动态库")
        subprocess.check_call(["tar", "-xzf", rt_tar, "-C", install_dir])
        try:
            os.remove(rt_tar)
        except Exception:
            pass

    bin_path = _find_server_bin()
    if not bin_path:
        raise RuntimeError("解压 llama.cpp CUDA 归档后未找到 llama-server 可执行文件")

    env = _build_llama_ld_env(install_dir)
    ok, ver_out = _verify_server_bin(bin_path, env)
    if not ok:
        raise RuntimeError(f"llama-server CUDA 启动自检失败: {ver_out[:300]}")

    first_line = ver_out.splitlines()[0] if ver_out else "ready"
    print(f"[GGUF 引擎] llama-server CUDA 引擎校验通过 ({first_line})")
    return bin_path, env

def _pick_free_local_port() -> int:
    """在非临时端口段 (18080~18999) 选取空闲本地监听端口，规避 Colab 容器 ephemeral 端口段 (32768+) 抢占冲突。"""
    for cand in range(18080, 18999):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", cand))
                return cand
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

def _start_qwen_gguf_server(
    gguf_path: str,
    mmproj_path: Optional[str],
    ctx_len: int,
    cur_free_gb: float,
    mtp_path: Optional[str] = None,
    model_id: str = "qwen38_27b_gguf",
    model_name: str = "Qwen3.8-27B",
    total_weights_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """
    启动驻留 GPU 的 llama-server 实例加载 GGUF 主模型 + 可选 mmproj 视觉投影器 + 可选 MTP 投机解码头。
    集成 Unsloth Native MTP 投机解码 (--spec-type draft-mtp)、GPU 端采样 (--backend-sampling)、
    超大 MoE 专家权重自动异构卸载 (-ot ".ffn_.*_exps.=CPU") 与 q8_0/q4_0 高带宽 KV Cache。
    """
    bin_path, env = _ensure_llama_server_binary()

    try:
        weights_gb = float(total_weights_gb) if (total_weights_gb and total_weights_gb > 0) else (os.path.getsize(gguf_path) / (1024 ** 3))
    except Exception:
        weights_gb = 18.0
    if mmproj_path and os.path.exists(mmproj_path):
        try:
            weights_gb += os.path.getsize(mmproj_path) / (1024 ** 3)
        except Exception:
            pass

    q8_kv_gb = (ctx_len / 262144.0) * 8.8
    q4_kv_gb = (ctx_len / 262144.0) * 4.6

    if cur_free_gb >= weights_gb + 1.5 + q8_kv_gb:
        primary_kv = "q8_0"
        primary_kv_gb = q8_kv_gb
    else:
        primary_kv = "q4_0"
        primary_kv_gb = q4_kv_gb

    has_mtp_file = bool(mtp_path and os.path.exists(mtp_path))

    # 计算精确 -ngl 层数并显式 --fit off，支持单卡装下全量 GPU 驻留，或超大 MoE / 大模型自动异构卸载
    if cur_free_gb >= (weights_gb + 1.2 + primary_kv_gb):
        primary_ngl = 999
        primary_moe_ot = False
    else:
        avail_for_weights = max(4.0, cur_free_gb - primary_kv_gb - 2.4)
        layer_unit_gb = max(0.20, weights_gb / 64.0)
        primary_ngl = max(8, min(64, int(avail_for_weights / layer_unit_gb)))
        primary_moe_ot = weights_gb > (cur_free_gb - 1.5)

    primary_mtp = has_mtp_file and primary_ngl == 999 and (cur_free_gb >= weights_gb + 2.8 + primary_kv_gb)
    # attempts 元素: (kv_type, ngl_val, use_mtp, moe_cpu_ot)
    attempts: List[Tuple[str, int, bool, bool]] = []
    if primary_moe_ot:
        # 优先尝试将 Attention 常驻 GPU (-ngl 999) + MoE 路由专家驻留大内存 (-ot .ffn_.*_exps.=CPU)
        attempts.append((primary_kv, 999, False, True))
    attempts.append((primary_kv, primary_ngl, primary_mtp, False))

    if has_mtp_file and not primary_mtp and not primary_moe_ot and (cur_free_gb >= weights_gb + 2.8 + q4_kv_gb):
        attempts.insert(0, ("q4_0", 999, True, False))

    if primary_ngl == 999:
        if primary_mtp:
            attempts.append((primary_kv, 999, False, False))
        fallback_ngl = max(12, min(64, int(max(4.0, cur_free_gb - q4_kv_gb - 2.8) / max(0.20, weights_gb / 64.0))))
        attempts.append(("q4_0", fallback_ngl, False, False))

    opener = _make_noproxy_opener()
    last_err = ""

    for attempt_idx, (kv_type, ngl_val, use_mtp, moe_ot) in enumerate(attempts):
        port = _pick_free_local_port()
        cmd = [
            bin_path,
            "-m", gguf_path,
            "--fit", "off",
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(ctx_len),
            "--kv-unified",
            "--parallel", "1",
            "--flash-attn", "on",
            "--cache-type-k", kv_type,
            "--cache-type-v", kv_type,
            "-b", "2048",
            "-ub", "1024",
            "--backend-sampling",
            "--samplers", "top_k;top_p;min_p;temperature",
            "--temp", "1.0",
            "--top-p", "0.95",
            "--top-k", "20",
            "--min-p", "0.0",
            "--repeat-penalty", "1.0",
            "--presence-penalty", "0.0",
            "--jinja",
            "--reasoning-format", "deepseek",
            "-ngl", str(ngl_val),
        ]
        if moe_ot:
            cmd.extend(["-ot", ".ffn_.*_exps.=CPU"])
        if use_mtp and mtp_path:
            cmd.extend([
                "--model-draft", mtp_path,
                "-ngld", "999",
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", "4",
                "--spec-draft-p-min", "0.45",
                "--cache-type-k-draft", kv_type,
                "--cache-type-v-draft", kv_type,
            ])
        if mmproj_path and os.path.exists(mmproj_path):
            cmd.extend(["--mmproj", mmproj_path, "--image-min-tokens", "1024"])

        if moe_ot:
            fit_desc = "MoE 异构驻留 (Attention GPU + Experts CPU)"
        elif ngl_val >= 99:
            fit_desc = "全量 GPU 驻留"
        else:
            fit_desc = f"混合驻留 ({ngl_val}/65 层 GPU)"
        mtp_desc = " + MTP 多词元投机解码" if use_mtp else ""
        print(
            f"[GGUF 引擎] 正在启动 {model_name}{mtp_desc} 服务 "
            f"(上下文 {ctx_len // 1024}K={ctx_len}, KV={kv_type}, GPU采样, {fit_desc}, port={port})..."
        )
        MODEL_DOWNLOAD_STATE["progress"] = 0.55
        MODEL_DOWNLOAD_STATE["message"] = (
            f"正在将 {model_name}{mtp_desc} 装载至 GPU ({ctx_len // 1024}K 上下文 · KV {kv_type} · {fit_desc})..."
        )

        recent_logs: List[str] = []
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _drain_stdout(p: subprocess.Popen, log_buf: List[str]):
            try:
                assert p.stdout is not None
                for raw_line in p.stdout:
                    ln = raw_line.rstrip("\r\n")
                    if not ln:
                        continue
                    log_buf.append(ln)
                    if len(log_buf) > 100:
                        del log_buf[:25]
            except Exception:
                pass

        threading.Thread(target=_drain_stdout, args=(proc, recent_logs), daemon=True).start()

        health_url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + 240.0
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with opener.open(health_url, timeout=2.0) as h_resp:
                    if h_resp.status == 200:
                        ready = True
                        break
            except urllib.error.HTTPError as he:
                # 503 表示 llama-server 正在加载权重到显存
                if he.code == 503:
                    MODEL_DOWNLOAD_STATE["progress"] = min(0.95, round(MODEL_DOWNLOAD_STATE["progress"] + 0.01, 2))
            except Exception:
                pass
            time.sleep(0.5)

        if ready:
            ctx_k_str = f"{ctx_len // 1024}K"
            if model_id == "qwen38_27b_gguf":
                quant_tag = "UD-Q4_K_M+MTP" if use_mtp else "UD-Q4_K_M"
                mode_label = f"Qwen3.8-27B · {quant_tag} · {ctx_k_str} (KV:{kv_type})"
            else:
                extra_tag = " · MTP" if use_mtp else (" · MoE异构" if moe_ot else "")
                mode_label = f"{model_name} · GGUF{extra_tag} · {ctx_k_str} (KV:{kv_type})"
            print(f"[GGUF 引擎] {model_name} 已就绪 ({mode_label})")
            return {
                "backend": "gguf",
                "model_id": model_id,
                "model_name": model_name,
                "port": port,
                "base_url": f"http://127.0.0.1:{port}",
                "server_proc": proc,
                "context_length": ctx_len,
                "kv_type": kv_type,
                "mtp_enabled": use_mtp,
                "mode_label": mode_label,
                "gguf_path": gguf_path,
                "mmproj_path": mmproj_path,
                "mtp_path": mtp_path if use_mtp else None,
            }

        # 未就绪：清理子进程并记录日志
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        tail_txt = "\n".join(recent_logs[-12:])
        last_err = tail_txt or "llama-server 启动超时"
        if attempt_idx + 1 < len(attempts):
            print(f"[GGUF 引擎] 当前配置 (KV={kv_type}, ngl={ngl_val}, MTP={use_mtp}, MoE_OT={moe_ot}) 显存触顶，自动降级重试...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.5)

    raise RuntimeError(f"{model_name} GGUF 服务启动失败: {last_err[-400:]}")

class _CausalLMProcessorAdapter:
    """将纯文本 AutoTokenizer 适配为与多模态 AutoProcessor 统一接口，兼容自定义 CausalLM 仓库"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        flat_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            c = m.get("content", "")
            if isinstance(c, list):
                txt = "\n".join(
                    str(b.get("text", ""))
                    for b in c
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                flat_msgs.append({"role": role, "content": txt})
            else:
                flat_msgs.append({"role": role, "content": str(c)})
        try:
            return self.tokenizer.apply_chat_template(flat_msgs, **kwargs)
        except Exception:
            parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in flat_msgs]
            parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

    def __call__(self, text=None, images=None, padding=True, return_tensors="pt", **kwargs):
        return self.tokenizer(text, padding=padding, return_tensors=return_tensors)

def _split_thinking_and_answer(raw_text: str) -> Tuple[str, str]:
    """解析 Qwen3.8 <think>...</think> 推理链与最终回复"""
    text = raw_text or ""
    if "</think>" in text:
        think_part, ans_part = text.split("</think>", 1)
        think_part = think_part.replace("<think>", "").strip()
        return think_part, ans_part.strip()
    if text.lstrip().startswith("<think>"):
        return text.replace("<think>", "", 1).strip(), ""
    return "", text.strip()

def _is_bundle_alive(bundle: Optional[Dict[str, Any]]) -> bool:
    if not bundle:
        return False
    if bundle.get("backend") == "gguf":
        proc = bundle.get("server_proc")
        return proc is not None and proc.poll() is None
    return bundle.get("model") is not None

def _get_qwen_llm_bundle(preferred_model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    candidates: List[str] = []
    if preferred_model_id:
        low = preferred_model_id.lower()
        if preferred_model_id in MODEL_CATALOG and _is_llm_model_id(preferred_model_id):
            candidates.append(preferred_model_id)
        for mid, mspec in MODEL_CATALOG.items():
            if mspec.get("modality") != "llm_vlm" or mid in candidates:
                continue
            if low == str(mspec.get("repo_id") or "").lower() or low == str(mspec.get("name") or "").lower():
                candidates.append(mid)
        if preferred_model_id in ("qwen38_27b", "qwen38_27b_gguf") and preferred_model_id not in candidates:
            candidates.append(preferred_model_id)
        elif ("gguf" in low or "q4_k" in low or "unsloth" in low) and "qwen38_27b_gguf" not in candidates:
            candidates.append("qwen38_27b_gguf")
        elif ("qwen3.8" in low or "qwen38" in low):
            for qid in ("qwen38_27b_gguf", "qwen38_27b"):
                if qid not in candidates:
                    candidates.append(qid)

    if _is_llm_model_id(rs.SELECTED_MODEL_ID) and rs.SELECTED_MODEL_ID not in candidates:
        candidates.append(rs.SELECTED_MODEL_ID)

    for fallback_id in list(LOADED_RUNTIMES.keys()) + ["qwen38_27b_gguf", "qwen38_27b"]:
        if _is_llm_model_id(fallback_id) and fallback_id not in candidates:
            candidates.append(fallback_id)

    for cid in candidates:
        rt = LOADED_RUNTIMES.get(cid)
        if rt:
            b = rt.get("llm_bundle")
            if _is_bundle_alive(b):
                return b
            # GGUF 子进程意外退出时清理陈旧表项
            if b and b.get("backend") == "gguf":
                LOADED_RUNTIMES.pop(cid, None)

    if _is_bundle_alive(rs.LLM_BUNDLE):
        return rs.LLM_BUNDLE
    return None

def _count_bundle_tokens(bundle: Optional[Dict[str, Any]], text: str) -> int:
    if not text:
        return 1
    if bundle is not None:
        if bundle.get("backend") == "gguf":
            try:
                opener = _make_noproxy_opener()
                tok_req = urllib.request.Request(
                    f"{bundle['base_url']}/tokenize",
                    data=json.dumps({"content": text}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with opener.open(tok_req, timeout=4.0) as resp:
                    obj = json.loads(resp.read().decode("utf-8"))
                    toks = obj.get("tokens")
                    if isinstance(toks, list):
                        return max(1, len(toks))
            except Exception:
                pass
        elif bundle.get("processor") is not None:
            try:
                return max(1, len(bundle["processor"].tokenizer.encode(text)))
            except Exception:
                pass
    return max(1, len(text) // 3)

def _run_qwen_llm_job(job_id: str, req: GenerateVideoRequest, target_llm_id: Optional[str] = None):
    """运行 Qwen3.8-27B (BF16 原生 或 Unsloth UD-Q4_K_M GGUF) 深度推理对话与多模态视觉理解"""
    # (shared state on runtime_state)
    start_time = time.time()
    if req.context_length is not None:
        rs.QWEN_CONTEXT_LENGTH = _clamp_context_length(req.context_length)

    bundle = _get_qwen_llm_bundle(preferred_model_id=target_llm_id or req.model_id)
    if bundle is None:
        req_name = MODEL_CATALOG.get(target_llm_id or req.model_id or rs.SELECTED_MODEL_ID, {}).get("name", "大语言模型")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["message"] = f"{req_name} 尚未加载至 GPU 显存！请先在左侧点击「加载到显存」。"
        return

    bundle_mid = str(bundle.get("model_id") or target_llm_id or rs.SELECTED_MODEL_ID)
    model_label = str(
        bundle.get("model_name")
        or MODEL_CATALOG.get(bundle_mid, {}).get("name")
        or ("Qwen3.8-27B-GGUF" if bundle.get("backend") == "gguf" else "Qwen3.8-27B")
    )
    first_img = decode_base64_image(req.first_frame_base64)
    last_img = decode_base64_image(req.last_frame_base64)
    attached_images = [im for im in (first_img, last_img) if im is not None]

    messages: List[Dict[str, Any]] = []
    if req.chat_messages:
        for m in req.chat_messages[:-1]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content:
                messages.append({"role": role, "content": [{"type": "text", "text": content}]})

    user_content: List[Dict[str, Any]] = []
    for im in attached_images:
        user_content.append({"type": "image", "image": im})
    user_content.append({"type": "text", "text": req.visual_prompt})
    messages.append({"role": "user", "content": user_content})

    JOBS[job_id]["media_type"] = "text"
    JOBS[job_id]["progress"] = 0.12
    JOBS[job_id]["message"] = f"{model_label} 正在编码多模态上下文并启动推理..."

    job_cancel_event = _JOB_CANCEL_EVENTS.setdefault(job_id, threading.Event())
    if job_cancel_event.is_set():
        raise GenerationAbortedError("已手动停止当前推理任务")
    ABORT_GENERATION_EVENT.clear()
    bundle_ctx = int(bundle.get("context_length") or rs.QWEN_CONTEXT_LENGTH)
    max_toks = max(64, min(65536, int(req.max_new_tokens or 16384), max(256, bundle_ctx - 256)))
    think_on = bool(req.enable_thinking)
    # 遵循 Unsloth Qwen3.8 官方部署采样表：Thinking (temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence=0.0)；Non-Thinking (temp=0.7, top_p=0.80, top_k=20, min_p=0.0, presence=1.5)
    default_temp = 1.0 if think_on else 0.7
    default_top_p = 0.95 if think_on else 0.80
    default_presence = 0.0 if think_on else 1.5
    temp = max(0.0, min(2.0, float(req.temperature if req.temperature is not None else default_temp)))
    eff_top_p = max(0.05, min(1.0, float(req.top_p if req.top_p is not None else default_top_p)))
    eff_presence = max(0.0, min(2.0, float(req.presence_penalty if req.presence_penalty is not None else default_presence)))
    eff_top_k = max(0, int(req.top_k if req.top_k is not None else 20))
    eff_min_p = max(0.0, min(1.0, float(req.min_p if req.min_p is not None else 0.0)))
    eff_repeat = max(0.5, min(2.0, float(req.repeat_penalty if req.repeat_penalty is not None else 1.0)))
    eff_effort = str(req.reasoning_effort or "medium").strip().lower()

    streamer, _, start_in_thinking, gen_thread, err_box, cancel_event = _prepare_qwen_streamer_generation(
        hf_messages=messages,
        pil_images=attached_images,
        hf_tools=None,
        enable_thinking=think_on,
        max_new_tokens=max_toks,
        temperature=temp,
        top_p=eff_top_p,
        preferred_model_id=bundle.get("model_id"),
        top_k=eff_top_k,
        min_p=eff_min_p,
        presence_penalty=eff_presence,
        repeat_penalty=eff_repeat,
        reasoning_effort=eff_effort,
    )

    parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
    think_parts: List[str] = []
    ans_parts: List[str] = []
    token_chunks = 0
    last_ui_sync = 0.0

    try:
        for piece in streamer:
            if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                cancel_event.set()
                break
            token_chunks += 1
            for kind, val in parser.feed(piece):
                if kind == "thinking":
                    think_parts.append(val)
                elif kind == "text":
                    ans_parts.append(val)
            now_ts = time.time()
            # 按 ~10Hz 节流拼接流式字符串快照，避免 100+ tok/s 高速解码下逐 token O(N^2) 字符串拷贝争抢 GIL
            if token_chunks <= 2 or (now_ts - last_ui_sync) >= 0.10:
                last_ui_sync = now_ts
                think_txt = "".join(think_parts).strip()
                ans_txt = "".join(ans_parts).strip()
                JOBS[job_id]["thinking_output"] = think_txt
                JOBS[job_id]["text_output"] = ans_txt
                JOBS[job_id]["progress"] = min(0.95, round(0.15 + 0.80 * min(1.0, token_chunks / float(max(128.0, max_toks * 0.5))), 3))
                elapsed_now = round(now_ts - start_time, 1)
                JOBS[job_id]["elapsed"] = elapsed_now
                tps = streamer.tok_per_sec
                speed_tag = f" · {tps} tok/s" if tps > 0 else ""
                JOBS[job_id]["message"] = (
                    f"{model_label} 正在深度思考 ({elapsed_now}s{speed_tag})..."
                    if (think_txt and not ans_txt)
                    else f"{model_label} 正在流式输出回复 ({elapsed_now}s{speed_tag})..."
                )
                touch_activity()
    except Exception:
        cancel_event.set()
    finally:
        cancel_event.set()

    tail_events, _ = parser.finish()
    for kind, val in tail_events:
        if kind == "thinking":
            think_parts.append(val)
        elif kind == "text":
            ans_parts.append(val)

    gen_thread.join(timeout=10.0)
    if err_box[0] is not None:
        raise err_box[0]

    think_final = "".join(think_parts).strip()
    ans_final = "".join(ans_parts).strip()
    if req.enable_thinking and not think_final and "</think>" in ans_final:
        think_final, ans_final = _split_thinking_and_answer("<think>\n" + ans_final)

    if job_cancel_event.is_set() or (ABORT_GENERATION_EVENT.is_set() and not (think_final or ans_final)):
        raise GenerationAbortedError("已手动停止当前推理任务")

    elapsed = round(time.time() - start_time, 1)
    final_tps = streamer.tok_per_sec
    gen_toks = streamer.generated_tokens
    tps_summary = f" · 共 {gen_toks} tokens · {final_tps} tok/s" if gen_toks > 0 and final_tps > 0 else ""
    JOBS[job_id]["thinking_output"] = think_final
    JOBS[job_id]["text_output"] = ans_final
    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["progress"] = 1.0
    JOBS[job_id]["elapsed"] = elapsed
    JOBS[job_id]["message"] = f"{model_label} 推理完成！耗时 {elapsed}s{tps_summary}"
    print(f"[{model_label}] 推理完成 | {gen_toks} tokens | {final_tps} tok/s | 耗时 {elapsed}s")

def _extract_openai_chat_media_inputs(messages: List[Dict[str, Any]]) -> Tuple[str, List[str], List[str], List[str]]:
    """从 /v1/chat/completions 的 messages 中提取最后用户指令及附图/视频/音频输入（用于驱动媒体模型）。"""
    text_chunks: List[str] = []
    image_srcs: List[str] = []
    video_srcs: List[str] = []
    audio_srcs: List[str] = []

    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") not in ("user", "system"):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if msg.get("role") == "user" and content.strip():
                text_chunks = [content.strip()]
        elif isinstance(content, list):
            user_texts: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = str(part.get("type") or "").lower()
                if ptype == "text" and part.get("text"):
                    user_texts.append(str(part["text"]).strip())
                elif ptype in ("image_url", "image"):
                    img_obj = part.get("image_url", part.get("image", ""))
                    url_str = img_obj.get("url", "") if isinstance(img_obj, dict) else str(img_obj)
                    if url_str.strip():
                        image_srcs.append(url_str.strip())
                elif ptype in ("video_url", "video"):
                    vid_obj = part.get("video_url", part.get("video", ""))
                    url_str = vid_obj.get("url", "") if isinstance(vid_obj, dict) else str(vid_obj)
                    if url_str.strip():
                        video_srcs.append(url_str.strip())
                elif ptype in ("input_audio", "audio_url", "audio"):
                    aud_obj = part.get("input_audio", part.get("audio_url", part.get("audio", "")))
                    if isinstance(aud_obj, dict):
                        val = str(aud_obj.get("data") or aud_obj.get("url") or "").strip()
                    else:
                        val = str(aud_obj or "").strip()
                    if val:
                        audio_srcs.append(val)
            if msg.get("role") == "user" and user_texts:
                text_chunks = user_texts
    return "\n".join(text_chunks).strip(), image_srcs, video_srcs, audio_srcs

def _extract_qwen_tool_calls(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """解析 Qwen3 / Qwen3.8 输出中的 <tool_call>...</tool_call> (兼容 JSON 与 XML 参数格式)"""
    if "<tool_call>" not in raw_text:
        return raw_text, []

    tool_calls: List[Dict[str, Any]] = []
    tc_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    for match in tc_pattern.finditer(raw_text):
        inner = match.group(1).strip()
        call_id = f"toolu_{uuid.uuid4().hex[:16]}"
        parsed_name = None
        parsed_args: Dict[str, Any] = {}

        # 格式 1: JSON {"name": "...", "arguments": {...}}
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict) and "name" in obj:
                parsed_name = str(obj["name"])
                raw_args = obj.get("arguments", obj.get("parameters", {}))
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except Exception:
                        parsed_args = {"raw": raw_args}
                elif isinstance(raw_args, dict):
                    parsed_args = raw_args
        except Exception:
            pass

        # 格式 2: Qwen3.5/3.8 XML <function=fn_name><parameter=k>v</parameter></function>
        if not parsed_name:
            fn_m = re.search(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", inner, re.DOTALL)
            if fn_m:
                parsed_name = fn_m.group(1).strip()
                body = fn_m.group(2)
                for pm in re.finditer(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
                    p_key = pm.group(1).strip()
                    p_val_raw = pm.group(2).strip()
                    try:
                        parsed_args[p_key] = json.loads(p_val_raw)
                    except Exception:
                        parsed_args[p_key] = p_val_raw

        if parsed_name:
            tool_calls.append({"id": call_id, "name": parsed_name, "arguments": parsed_args})

    cleaned_text = tc_pattern.sub("", raw_text).strip()
    return cleaned_text, tool_calls

class _IncrementalQwenStreamParser:
    """流式增量拆分 <think>...</think>、正文与 <tool_call>...</tool_call>，杜绝半截标签外泄"""

    def __init__(self, start_in_thinking: bool):
        self.in_thinking = start_in_thinking
        self.in_tool_call = False
        self.buf = ""
        self.tool_raw = ""

    def feed(self, piece: str) -> List[Tuple[str, str]]:
        events: List[Tuple[str, str]] = []
        self.buf += piece

        while self.buf:
            if self.in_tool_call:
                self.tool_raw += self.buf
                self.buf = ""
                break

            if self.in_thinking:
                # 忽略开头的显式 <think>
                if self.buf.lstrip().startswith("<think>"):
                    idx = self.buf.find("<think>")
                    self.buf = self.buf[idx + len("<think>"):]
                    continue
                end_idx = self.buf.find("</think>")
                if end_idx != -1:
                    before = self.buf[:end_idx]
                    if before:
                        events.append(("thinking", before))
                    self.buf = self.buf[end_idx + len("</think>"):].lstrip("\n")
                    self.in_thinking = False
                    continue
                # 保留末尾可能未闭合的 '<' 前缀
                lt = self.buf.rfind("<")
                if lt != -1 and len(self.buf) - lt < 12 and "</think>".startswith(self.buf[lt:]):
                    safe = self.buf[:lt]
                    if safe:
                        events.append(("thinking", safe))
                    self.buf = self.buf[lt:]
                    break
                events.append(("thinking", self.buf))
                self.buf = ""
            else:
                if self.buf.lstrip().startswith("<think>"):
                    idx = self.buf.find("<think>")
                    before = self.buf[:idx]
                    if before:
                        events.append(("text", before))
                    self.buf = self.buf[idx + len("<think>"):]
                    self.in_thinking = True
                    continue

                tc_idx = self.buf.find("<tool_call>")
                if tc_idx != -1:
                    before = self.buf[:tc_idx]
                    if before:
                        events.append(("text", before))
                    self.tool_raw += self.buf[tc_idx:]
                    self.buf = ""
                    self.in_tool_call = True
                    break

                lt = self.buf.rfind("<")
                if lt != -1 and len(self.buf) - lt < 14 and (
                    "<tool_call>".startswith(self.buf[lt:]) or "<think>".startswith(self.buf[lt:])
                ):
                    safe = self.buf[:lt]
                    if safe:
                        events.append(("text", safe))
                    self.buf = self.buf[lt:]
                    break

                events.append(("text", self.buf))
                self.buf = ""

        return events

    def finish(self) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
        events: List[Tuple[str, str]] = []
        if self.buf:
            if self.in_tool_call:
                self.tool_raw += self.buf
            elif self.in_thinking:
                events.append(("thinking", self.buf))
            else:
                events.append(("text", self.buf))
            self.buf = ""

        _, tool_calls = _extract_qwen_tool_calls(self.tool_raw) if self.tool_raw else ("", [])
        return events, tool_calls

def _normalize_to_hf_messages_and_images(
    raw_messages: List[Dict[str, Any]],
    system_prompt: Any = None,
    protocol: str = "openai",
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """将 OpenAI 或 Anthropic 消息数组统一转换为 Qwen3.8 Processor 标准多模态结构"""
    hf_messages: List[Dict[str, Any]] = []
    pil_images: List[Image.Image] = []

    if system_prompt:
        if isinstance(system_prompt, str) and system_prompt.strip():
            hf_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})
        elif isinstance(system_prompt, list):
            sys_txt = "\n".join(
                b.get("text", "") for b in system_prompt if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if sys_txt:
                hf_messages.append({"role": "system", "content": [{"type": "text", "text": sys_txt}]})

    for msg in raw_messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "tool":
            txt = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            hf_messages.append({"role": "tool", "content": txt})
            continue

        blocks: List[Dict[str, Any]] = []
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype == "text":
                    blocks.append({"type": "text", "text": str(part.get("text", ""))})
                elif ptype == "image_url":
                    img_obj = part.get("image_url", {})
                    url_str = img_obj.get("url", "") if isinstance(img_obj, dict) else str(img_obj)
                    im = _load_pil_from_url_or_b64(url_str)
                    if im is not None:
                        pil_images.append(im)
                        blocks.append({"type": "image", "image": im})
                elif ptype == "image":
                    src = part.get("source", {})
                    im = None
                    if isinstance(src, dict):
                        if src.get("type") == "base64":
                            im = decode_base64_image(src.get("data", ""))
                        elif src.get("type") == "url":
                            im = _load_pil_from_url_or_b64(src.get("url", ""))
                    if im is not None:
                        pil_images.append(im)
                        blocks.append({"type": "image", "image": im})
                elif ptype == "tool_use":
                    tc_str = json.dumps({"name": part.get("name", ""), "arguments": part.get("input", {})}, ensure_ascii=False)
                    blocks.append({"type": "text", "text": f"<tool_call>\n{tc_str}\n</tool_call>"})
                elif ptype == "tool_result":
                    tr_content = part.get("content", "")
                    if isinstance(tr_content, list):
                        tr_txt = "\n".join(x.get("text", "") for x in tr_content if isinstance(x, dict))
                    else:
                        tr_txt = str(tr_content)
                    hf_messages.append({"role": "tool", "content": tr_txt})

        # OpenAI assistant historical tool_calls
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                raw_a = fn.get("arguments", "{}")
                try:
                    a_obj = json.loads(raw_a) if isinstance(raw_a, str) else raw_a
                except Exception:
                    a_obj = {}
                tc_json = json.dumps({"name": fn.get("name", ""), "arguments": a_obj}, ensure_ascii=False)
                blocks.append({"type": "text", "text": f"<tool_call>\n{tc_json}\n</tool_call>"})

        if blocks:
            hf_messages.append({"role": role, "content": blocks})

    return hf_messages, pil_images

def _normalize_hf_tools(raw_tools: Optional[List[Dict[str, Any]]], protocol: str) -> Optional[List[Dict[str, Any]]]:
    if not raw_tools:
        return None
    hf_tools: List[Dict[str, Any]] = []
    for t in raw_tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            hf_tools.append(t)
        elif "name" in t:
            # Anthropic tool schema -> HF OpenAI function tool schema
            hf_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
    return hf_tools if hf_tools else None

class _GGUFStreamIterator:
    """与 _TimedStreamer 接口一致的 GGUF 流式迭代器，统一供原生客户端与 OpenAI/Anthropic 网关消费"""

    def __init__(self, est_prompt_tokens: int = 1):
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self.prompt_tokens: int = max(1, int(est_prompt_tokens))
        self.generated_tokens: int = 0
        self.first_token_ts: Optional[float] = None
        self._server_tps: float = 0.0

    def put_text(self, text: str, token_inc: int = 1):
        if not text:
            return
        now = time.time()
        if self.first_token_ts is None:
            self.first_token_ts = now
        if token_inc > 0:
            self.generated_tokens += token_inc
        self._q.put(text)

    def end(self):
        self._q.put(None)

    @property
    def tok_per_sec(self) -> float:
        if self._server_tps > 0:
            return round(self._server_tps, 1)
        if self.first_token_ts is None or self.generated_tokens <= 1:
            return 0.0
        dt = max(0.05, time.time() - self.first_token_ts)
        return round((self.generated_tokens - 1) / dt, 1)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item

def _pil_to_jpeg_data_url(im: Image.Image) -> str:
    rgb = im.convert("RGB")
    max_side = max(rgb.size)
    if max_side > 2048:
        scale = 2048.0 / float(max_side)
        rgb = rgb.resize(
            (max(64, int(rgb.size[0] * scale)), max(64, int(rgb.size[1] * scale))),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def _hf_messages_to_gguf_openai_messages(hf_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    oai_msgs: List[Dict[str, Any]] = []
    est_chars = 0
    est_imgs = 0

    for msg in hf_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "tool":
            s_txt = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            est_chars += len(s_txt)
            oai_msgs.append({"role": "tool", "content": s_txt, "tool_call_id": msg.get("tool_call_id", "call_0")})
            continue

        if isinstance(content, str):
            est_chars += len(content)
            oai_msgs.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            has_img = any(isinstance(b, dict) and b.get("type") == "image" and b.get("image") is not None for b in content)
            if not has_img:
                txt = "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
                est_chars += len(txt)
                oai_msgs.append({"role": role, "content": txt})
            else:
                parts: List[Dict[str, Any]] = []
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "image" and b.get("image") is not None:
                        est_imgs += 1
                        parts.append({"type": "image_url", "image_url": {"url": _pil_to_jpeg_data_url(b["image"])}})
                    elif b.get("type") == "text":
                        t_str = str(b.get("text", ""))
                        est_chars += len(t_str)
                        parts.append({"type": "text", "text": t_str})
                oai_msgs.append({"role": role, "content": parts})

    est_tokens = max(1, est_chars // 3 + est_imgs * 512)
    return oai_msgs, est_tokens

def _prepare_gguf_streamer_generation(
    bundle: Dict[str, Any],
    hf_messages: List[Dict[str, Any]],
    hf_tools: Optional[List[Dict[str, Any]]],
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: Optional[float] = None,
    repeat_penalty: float = 1.0,
    reasoning_effort: Optional[str] = None,
):
    """通过驻留 GPU 的 llama-server 流式执行 Qwen3.8-27B-UD-Q4_K_M 推理 (遵循 Unsloth 官方 GGUF 部署参数规范，支持最高 256K 上下文与视觉输入)"""
    oai_msgs, est_prompt_tokens = _hf_messages_to_gguf_openai_messages(hf_messages)
    bundle_ctx = int(bundle.get("context_length") or rs.QWEN_CONTEXT_LENGTH)
    max_toks = max(16, min(65536, int(max_new_tokens or 16384), max(256, bundle_ctx - min(est_prompt_tokens, bundle_ctx - 256))))

    streamer = _GGUFStreamIterator(est_prompt_tokens=est_prompt_tokens)
    cancel_event = threading.Event()
    err_box: List[Optional[Exception]] = [None]

    eff_presence = (
        float(presence_penalty)
        if presence_penalty is not None
        else (0.0 if enable_thinking else 1.5)
    )
    tpl_kwargs: Dict[str, Any] = {"enable_thinking": bool(enable_thinking)}
    if enable_thinking and reasoning_effort and reasoning_effort not in ("default", "none"):
        tpl_kwargs["reasoning_effort"] = str(reasoning_effort)

    eff_temp = max(0.0, min(2.0, float(temperature)))
    eff_top_p_val = max(0.05, min(1.0, float(top_p)))
    eff_top_k_val = max(0, int(top_k))
    eff_min_p_val = max(0.0, min(1.0, float(min_p)))
    eff_pres_val = max(0.0, min(2.0, eff_presence))
    eff_rep_val = max(0.5, min(2.0, float(repeat_penalty)))

    # 针对 Qwen3.8 超大词表 (248,320) + MTP 多词元投机解码定制采样管线：
    # - 无惩罚项时 (Thinking 默认) 仅保留 GPU 原生采样器，100% 在 GPU 端完成采样，零 PCIe Logits 拷贝；
    # - 启用惩罚项时 (Non-Thinking presence_penalty=1.5) 先在 GPU 执行 top_k 将 248K 词表筛至 20 项，再计算 penalties。
    has_penalties = (eff_pres_val > 1e-5) or (abs(eff_rep_val - 1.0) > 1e-4)
    req_samplers: List[str] = []
    if eff_top_k_val > 0:
        req_samplers.append("top_k")
    if has_penalties:
        req_samplers.append("penalties")
    if eff_top_p_val < 0.9999:
        req_samplers.append("top_p")
    if eff_min_p_val > 1e-6:
        req_samplers.append("min_p")
    req_samplers.append("temperature")

    payload: Dict[str, Any] = {
        "model": "Qwen3.8-27B-UD-Q4_K_M",
        "messages": oai_msgs,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_toks,
        "temperature": eff_temp,
        "top_p": eff_top_p_val,
        "top_k": eff_top_k_val,
        "min_p": eff_min_p_val,
        "presence_penalty": eff_pres_val,
        "repeat_penalty": eff_rep_val,
        "samplers": req_samplers,
        "chat_template_kwargs": tpl_kwargs,
    }
    if hf_tools:
        payload["tools"] = hf_tools

    def _locked_gguf_generate():
        if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            streamer.end()
            return

        with GPU_LOCK:
            if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                streamer.end()
                return

            srv_proc = bundle.get("server_proc")
            if srv_proc is not None and srv_proc.poll() is not None:
                err_box[0] = RuntimeError(
                    f"llama-server 子进程已意外退出 (code={srv_proc.returncode})，请在左侧重新点击「加载到显存」。"
                )
                streamer.end()
                return

            opener = _make_noproxy_opener()
            url = f"{bundle['base_url']}/v1/chat/completions"
            req_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            http_req = urllib.request.Request(
                url,
                data=req_bytes,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                method="POST",
            )

            in_reasoning = False
            tc_acc: Dict[int, Dict[str, str]] = {}
            resp = None
            raw_sock = None
            stop_watch = threading.Event()

            try:
                # 严禁将 raw_sock.settimeout 设为短秒数：Python 3.13 SocketIO 一旦触发单次超时会将 _timeout_occurred 永久置 True，
                # 导致后续 resp.readline() 立即抛出 OSError("cannot read from timed out object")。
                # 此处保持长超时 900s 以覆盖 256K 超长上下文预填充，并由独立守护线程监听中断信号关闭套接字。
                resp = opener.open(http_req, timeout=900.0)
                raw_sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)

                def _abort_watcher():
                    while not stop_watch.wait(0.25):
                        if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            if raw_sock is not None:
                                try:
                                    raw_sock.shutdown(socket.SHUT_RDWR)
                                except Exception:
                                    pass
                            if resp is not None:
                                try:
                                    resp.close()
                                except Exception:
                                    pass
                            break

                threading.Thread(target=_abort_watcher, daemon=True).start()

                while True:
                    if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                        break
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue

                    if isinstance(chunk.get("error"), dict):
                        raise RuntimeError(str(chunk["error"].get("message") or chunk["error"]))

                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        if usage.get("prompt_tokens"):
                            streamer.prompt_tokens = int(usage["prompt_tokens"])
                        if usage.get("completion_tokens"):
                            streamer.generated_tokens = max(streamer.generated_tokens, int(usage["completion_tokens"]))

                    timings = chunk.get("timings")
                    if isinstance(timings, dict) and timings.get("predicted_per_second"):
                        streamer._server_tps = float(timings["predicted_per_second"])

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    r_txt = delta.get("reasoning_content")
                    if r_txt:
                        if not in_reasoning:
                            streamer.put_text("<think>\n" + str(r_txt), token_inc=1)
                            in_reasoning = True
                        else:
                            streamer.put_text(str(r_txt), token_inc=1)

                    c_txt = delta.get("content")
                    if c_txt:
                        if in_reasoning:
                            streamer.put_text("</think>\n" + str(c_txt), token_inc=1)
                            in_reasoning = False
                        else:
                            streamer.put_text(str(c_txt), token_inc=1)

                    d_tcs = delta.get("tool_calls")
                    if isinstance(d_tcs, list):
                        for tc_item in d_tcs:
                            if not isinstance(tc_item, dict):
                                continue
                            idx = int(tc_item.get("index", 0))
                            slot = tc_acc.setdefault(idx, {"name": "", "arguments": ""})
                            fn_obj = tc_item.get("function") or {}
                            if fn_obj.get("name"):
                                slot["name"] += str(fn_obj["name"])
                            if fn_obj.get("arguments"):
                                slot["arguments"] += str(fn_obj["arguments"])

                if in_reasoning:
                    streamer.put_text("</think>\n", token_inc=0)
                    in_reasoning = False

                if tc_acc:
                    for idx in sorted(tc_acc.keys()):
                        slot = tc_acc[idx]
                        if not slot["name"]:
                            continue
                        try:
                            arg_val = json.loads(slot["arguments"]) if slot["arguments"] else {}
                        except Exception:
                            arg_val = {"raw": slot["arguments"]}
                        tc_str = json.dumps({"name": slot["name"], "arguments": arg_val}, ensure_ascii=False)
                        streamer.put_text(f"<tool_call>\n{tc_str}\n</tool_call>", token_inc=0)

            except urllib.error.HTTPError as he:
                err_body = ""
                try:
                    err_body = he.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                if he.code == 400 and ("context" in err_body.lower() or "exceed" in err_body.lower()):
                    err_box[0] = RuntimeError(
                        f"输入长度超出当前 GGUF 上下文窗口 ({bundle_ctx // 1024}K = {bundle_ctx} tokens)！"
                        f"请在左侧调大「上下文长度」（最高 256K）后点击重新加载模型。({err_body[:180]})"
                    )
                else:
                    err_box[0] = RuntimeError(f"GGUF 推理服务返回 HTTP {he.code}: {err_body[:240]}")
            except Exception as ex:
                if not (cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set()):
                    err_box[0] = ex
            finally:
                stop_watch.set()
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                streamer.end()
                touch_activity()

    t = threading.Thread(target=_locked_gguf_generate, daemon=True)
    t.start()
    return streamer, streamer.prompt_tokens, False, t, err_box, cancel_event

def _prepare_qwen_streamer_generation(
    hf_messages: List[Dict[str, Any]],
    pil_images: List[Image.Image],
    hf_tools: Optional[List[Dict[str, Any]]],
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    preferred_model_id: Optional[str] = None,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: Optional[float] = None,
    repeat_penalty: float = 1.0,
    reasoning_effort: Optional[str] = None,
):
    """构建 Qwen3.8 (GGUF 或 BF16 Transformers) 后台流式推理任务并返回 (streamer, prompt_tokens, start_in_thinking, gen_thread, err_box, cancel_event)"""
    bundle = _get_qwen_llm_bundle(preferred_model_id=preferred_model_id)
    if bundle is not None and bundle.get("backend") == "gguf":
        return _prepare_gguf_streamer_generation(
            bundle=bundle,
            hf_messages=hf_messages,
            hf_tools=hf_tools,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            reasoning_effort=reasoning_effort,
        )

    model = bundle["model"]
    processor = bundle["processor"]
    bundle_ctx = int(bundle.get("context_length") or rs.QWEN_CONTEXT_LENGTH)

    tpl_kwargs: Dict[str, Any] = dict(
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if enable_thinking and reasoning_effort and reasoning_effort not in ("default", "none"):
        tpl_kwargs["reasoning_effort"] = str(reasoning_effort)
    if hf_tools:
        tpl_kwargs["tools"] = hf_tools

    try:
        prompt_text = processor.apply_chat_template(hf_messages, **tpl_kwargs)
    except TypeError:
        tpl_kwargs.pop("reasoning_effort", None)
        try:
            prompt_text = processor.apply_chat_template(hf_messages, **tpl_kwargs)
        except TypeError:
            tpl_kwargs.pop("enable_thinking", None)
            prompt_text = processor.apply_chat_template(hf_messages, **tpl_kwargs)

    start_in_thinking = enable_thinking and (
        prompt_text.rstrip().endswith("<think>") or "<think>" in prompt_text[-24:]
    ) and not prompt_text.rstrip().endswith("</think>")

    prompt_tokens = max(1, len(processor.tokenizer.encode(prompt_text)))
    streamer = _make_timed_token_streamer(processor.tokenizer, timeout=180.0)

    max_toks = max(16, min(65536, int(max_new_tokens or 16384), max(128, bundle_ctx - prompt_tokens)))
    do_sample = temperature > 0.01
    cancel_event = threading.Event()
    err_box: List[Optional[Exception]] = [None]

    def _locked_generate():
        if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            try:
                streamer.end()
            except Exception:
                pass
            return

        with GPU_LOCK:
            if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                try:
                    streamer.end()
                except Exception:
                    pass
                return

            inputs = None
            try:
                if pil_images:
                    inputs = processor(text=[prompt_text], images=pil_images, padding=True, return_tensors="pt").to(model.device)
                else:
                    inputs = processor(text=[prompt_text], padding=True, return_tensors="pt").to(model.device)

                seq_in_len = int(inputs["input_ids"].shape[-1])
                if seq_in_len >= bundle_ctx:
                    raise RuntimeError(
                        f"输入长度 ({seq_in_len} tokens) 超出当前设定的上下文上限 ({bundle_ctx // 1024}K = {bundle_ctx} tokens)，"
                        f"请在左侧调大「上下文长度」（最高支持 256K）。"
                    )
                eff_max_toks = max(16, min(max_toks, bundle_ctx - seq_in_len))

                gen_kwargs: Dict[str, Any] = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=eff_max_toks,
                    do_sample=do_sample,
                    stopping_criteria=_make_abort_stopping_criteria(cancel_event),
                )
                if do_sample:
                    gen_kwargs["temperature"] = max(0.05, min(2.0, float(temperature)))
                    gen_kwargs["top_p"] = max(0.05, min(1.0, float(top_p)))
                    if int(top_k) > 0:
                        gen_kwargs["top_k"] = int(top_k)

                with torch.inference_mode():
                    model.generate(**gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                err_box[0] = RuntimeError(
                    "GPU 显存不足 (OOM)：BF16 原生精度长上下文峰值显存较高，建议切换使用独立的「Qwen3.8-27B-GGUF (UD-Q4_K_M)」量化模型，或适当调低上下文长度。"
                )
                try:
                    streamer.end()
                except Exception:
                    pass
            except Exception as ex:
                err_box[0] = ex
                try:
                    streamer.end()
                except Exception:
                    pass
            finally:
                if inputs is not None:
                    del inputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                touch_activity()

    t = threading.Thread(target=_locked_generate, daemon=True)
    t.start()
    return streamer, prompt_tokens, start_in_thinking, t, err_box, cancel_event
