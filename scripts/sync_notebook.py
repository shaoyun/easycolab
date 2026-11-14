#!/usr/bin/env python3
"""将 runtime_auth.py、easycolab_server/* 与 app.py 自动同步至 EasyColab_Server.ipynb。"""

import base64
import json
import os
import zlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(REPO_ROOT, "EasyColab_Server.ipynb")


def _encode_server_bundle() -> str:
    srv_dir = os.path.join(REPO_ROOT, "easycolab_server")
    files = {}
    for fname in sorted(os.listdir(srv_dir)):
        fpath = os.path.join(srv_dir, fname)
        if fname.endswith(".py") and os.path.isfile(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                files[fname] = f.read()
    raw = json.dumps(files, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")


def sync_notebook() -> None:
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    with open(os.path.join(REPO_ROOT, "runtime_auth.py"), "r", encoding="utf-8") as f:
        auth_src = f.read()
    with open(os.path.join(REPO_ROOT, "app.py"), "r", encoding="utf-8") as f:
        app_src = f.read()

    bundle_b64 = _encode_server_bundle()

    # Cell 1: 检查 GPU 环境（替代历史冗余的巨型单文件副本，避免“全部运行”时阻塞）
    nb["cells"][1]["source"] = [
        "#@title 1️⃣ 第一步：检查 Colab GPU 架构与运行环境\n",
        "import subprocess\n",
        'subprocess.run(["nvidia-smi"], check=False)\n',
    ]
    nb["cells"][1].setdefault("metadata", {})["cellView"] = "form"

    # Cell 4: 写入 runtime_auth.py
    nb["cells"][4]["source"] = ["%%writefile runtime_auth.py\n"] + auth_src.splitlines(keepends=True)

    # Cell 5: 部署 easycolab_server/*.py 模块包与 app.py 主入口
    write_pkg_cell = (
        "#@title 4️⃣ 第四步：写入 EasyColab 模块化服务端包 (easycolab_server/*) 与主程序 (app.py)\n"
        "import base64, json, os, shutil, zlib\n\n"
        f'_BUNDLE_B64 = "{bundle_b64}"\n'
        "_pkg_files = json.loads(zlib.decompress(base64.b64decode(_BUNDLE_B64)).decode(\"utf-8\"))\n"
        'shutil.rmtree("easycolab_server", ignore_errors=True)\n'
        'os.makedirs("easycolab_server", exist_ok=True)\n'
        "for _name, _code in _pkg_files.items():\n"
        '    with open(os.path.join("easycolab_server", os.path.basename(_name)), "w", encoding="utf-8") as _f:\n'
        "        _f.write(_code)\n"
        "with open(\"app.py\", \"w\", encoding=\"utf-8\") as _af:\n"
        f"    _af.write({json.dumps(app_src, ensure_ascii=False)})\n"
        'print(f"✅ 已写入 easycolab_server ({len(_pkg_files)} 个模块) 与 app.py")\n'
    )
    nb["cells"][5]["source"] = write_pkg_cell.splitlines(keepends=True)
    nb["cells"][5].setdefault("metadata", {})["cellView"] = "form"

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print(f"Synced EasyColab_Server.ipynb (bundle compressed len={len(bundle_b64)} chars)")


if __name__ == "__main__":
    sync_notebook()
