#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

MODE="${1:-web}"
VENV_DIR="${PROJECT_ROOT}/.venv_dev"

ensure_py_env() {
    local req_import="$1"
    shift
    local pkgs=("$@")

    if python3 -c "import ${req_import}" >/dev/null 2>&1; then
        PY_BIN="python3"
        return
    fi

    if [ ! -x "${VENV_DIR}/bin/python" ]; then
        echo "📦 正在创建本地轻量调试虚拟环境 (${VENV_DIR})..."
        python3 -m venv "${VENV_DIR}"
    fi
    PY_BIN="${VENV_DIR}/bin/python"
    if ! "${PY_BIN}" -c "import ${req_import}" >/dev/null 2>&1; then
        echo "📥 正在安装调试所需依赖: ${pkgs[*]} ..."
        "${PY_BIN}" -m pip install -q --upgrade pip
        "${PY_BIN}" -m pip install "${pkgs[@]}"
    fi
}

case "${MODE}" in
    web|--web)
        ensure_py_env "gradio, fastapi, PIL" gradio fastapi uvicorn Pillow
        PORT="${EASYCOLAB_PORT:-7860}"
        echo "🌐 正在启动本地网页工作台: http://127.0.0.1:${PORT} (按 Ctrl+C 停止)..."
        EASYCOLAB_LOCAL_PREVIEW=1 EASYCOLAB_PORT="${PORT}" exec "${PY_BIN}" "${PROJECT_ROOT}/app.py" --local
        ;;

    mac|--mac)
        APP_DIR="${PROJECT_ROOT}/EasyColab.app"
        CONTENTS_DIR="${APP_DIR}/Contents"
        MACOS_DIR="${CONTENTS_DIR}/MacOS"
        RESOURCES_DIR="${CONTENTS_DIR}/Resources"
        mkdir -p "${MACOS_DIR}" "${RESOURCES_DIR}"

        # 若尚未生成过 Info.plist 或图标，先调一次完整构建初始化包结构
        if [ ! -f "${CONTENTS_DIR}/Info.plist" ]; then
            bash "${PROJECT_ROOT}/macos_app/build_macos_app.sh"
        else
            for res_file in "VERSION" "app.py" "runtime_auth.py" "colab_bridge.py" "EasyColab_Server.ipynb" "requirements.txt"; do
                if [ -f "${PROJECT_ROOT}/${res_file}" ]; then
                    cp -f "${PROJECT_ROOT}/${res_file}" "${RESOURCES_DIR}/${res_file}"
                fi
            done
            ARCH="$(uname -m)"
            echo "⚡ 正在极速编译 macOS 客户端 (${ARCH}, Debug -Onone，跳过图标生成与 zip 打包)..."
            swiftc -Onone \
                -target "${ARCH}-apple-macos13.0" \
                -parse-as-library \
                -framework SwiftUI \
                -framework AppKit \
                -framework AVKit \
                -framework UniformTypeIdentifiers \
                "${PROJECT_ROOT}/macos_app/EasyColabStudio.swift" \
                "${PROJECT_ROOT}/macos_app/Sources"/*.swift \
                -o "${MACOS_DIR}/EasyColab"
            codesign --force --sign - "${APP_DIR}" >/dev/null 2>&1 || true
        fi

        echo "🚀 正在启动 EasyColab.app..."
        pkill -x EasyColab >/dev/null 2>&1 || true
        open "${APP_DIR}"
        ;;

    qt|--qt|py|--py)
        ensure_py_env "PySide6, requests" PySide6 requests
        echo "🖥️  正在免编译启动 Python 桌面客户端..."
        exec "${PY_BIN}" -m windows_app
        ;;

    *)
        echo "用法: ./dev.sh [web | mac | qt]"
        echo "  ./dev.sh web  (默认) 在本地浏览器启动 Gradio 网页端预览 (http://127.0.0.1:7860)"
        echo "  ./dev.sh mac         3 秒极速单架构编译并启动 macOS 原生客户端 (跳过重画图标与打包 zip)"
        echo "  ./dev.sh qt          零编译直接运行跨平台 Python 桌面客户端"
        exit 1
        ;;
esac
