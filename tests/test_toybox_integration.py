"""Tests for H3 Toybox + shared ComfyUI workbench + password-protected ingress tunnel integration."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ["EASYCOLAB_LOCAL_PREVIEW"] = "1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import app as easyapp  # noqa: E402
import runtime_auth  # noqa: E402


class ToyboxIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fastapi_app = FastAPI()
        easyapp.attach_macos_api_routes(cls.fastapi_app)
        cls.client = TestClient(cls.fastapi_app)

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.comfy_root = os.path.join(self.tmp_dir.name, "ComfyUI")
        self.persist_root = os.path.join(self.tmp_dir.name, "Persist_ComfyUI")
        os.makedirs(self.comfy_root, exist_ok=True)
        os.makedirs(self.persist_root, exist_ok=True)
        self.orig_comfy_state = dict(easyapp.COMFYUI_STATE)
        self.orig_toybox_state = dict(easyapp.TOYBOX_STATE)

    def tearDown(self):
        easyapp.COMFYUI_STATE.clear()
        easyapp.COMFYUI_STATE.update(self.orig_comfy_state)
        easyapp.TOYBOX_STATE.clear()
        easyapp.TOYBOX_STATE.update(self.orig_toybox_state)
        self.tmp_dir.cleanup()

    def test_nginx_has_separate_password_protected_toybox_server(self):
        cfg = runtime_auth.nginx_config("/tmp/authdir")
        self.assertIn(f"listen 127.0.0.1:{runtime_auth.TOYBOX_INGRESS};", cfg)
        self.assertIn(f"proxy_pass http://127.0.0.1:{runtime_auth.TOYBOX_BACKEND};", cfg)
        # Ensure Basic Auth + CSRF are enabled on Toybox tunnel just like ComfyUI
        self.assertGreaterEqual(cfg.count('auth_basic "EasyColab";'), 3)

    def test_semantic_bridge_subdir_and_preset_pack_registered(self):
        self.assertIn("semantic_bridge", easyapp.COMFY_STANDARD_MODEL_SUBDIRS)
        pack_ids = {p["id"] for p in easyapp.COMFY_PRESET_WORKFLOW_PACKS}
        self.assertIn("minimax_h3_toybox", pack_ids)

    def test_configure_toybox_wires_shared_comfy_and_preserves_user_settings(self):
        data_dir = os.path.join(self.persist_root, "H3Toybox")
        os.makedirs(data_dir, exist_ok=True)
        cfg_file = os.path.join(data_dir, "config.json")
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump({"rewriter": {"provider": "gemini"}, "steps": {"fast": [8, 4]}}, f)

        # Create a legacy video file inside ComfyUI/output/H3Toybox to verify safe migration
        old_out = os.path.join(self.comfy_root, "output", "H3Toybox")
        os.makedirs(old_out, exist_ok=True)
        with open(os.path.join(old_out, "clip_01.mp4"), "wb") as f:
            f.write(b"fake-mp4")

        # Mark P-H3-Image weights present
        for sub, fname, _, _, _ in easyapp.TOYBOX_IMAGE_MODELS:
            p = os.path.join(self.persist_root, "models", sub, fname)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(b"0" * (2 * 1024 * 1024))

        paths = {
            "comfy_root": self.comfy_root,
            "persist_root": self.persist_root,
            "persistence_mode": "local_only",
            "persistence_label": "测试存储",
            "is_drive": False,
            "base_root": self.tmp_dir.name,
            "repo_dir": os.path.join(self.tmp_dir.name, "minimax-h3-colab"),
            "app_dir": os.path.join(self.tmp_dir.name, "minimax-h3-colab", "toybox"),
            "enhancer_dir": os.path.join(self.tmp_dir.name, "t8-enhancer"),
            "data_dir": data_dir,
            "secrets_path": os.path.join(self.tmp_dir.name, "h3toybox_secrets.json"),
        }

        cfg, _, secrets_path = easyapp._configure_toybox_runtime_files(
            paths, comfy_port=28188, save_api_keys=False
        )
        self.assertEqual(cfg["comfy_url"], "http://127.0.0.1:28188")
        self.assertEqual(cfg["comfy_input_dir"], os.path.join(self.comfy_root, "input"))
        self.assertTrue(cfg["free_after_rewrite"])
        self.assertEqual(cfg["rewriter"]["provider"], "gemini")
        self.assertEqual(cfg["image"]["engine"], "ph3")
        self.assertTrue(cfg["image"]["installed"])
        self.assertEqual(secrets_path, paths["secrets_path"])
        self.assertTrue(os.path.islink(old_out))
        self.assertTrue(os.path.isfile(os.path.join(data_dir, "videos", "clip_01.mp4")))

    def test_nested_semantic_bridge_download_skips_and_moves_cleanly(self):
        fake_src = os.path.join(self.tmp_dir.name, "dl_tmp", "semantic_bridge", "t8_compat", "bridge.safetensors")
        os.makedirs(os.path.dirname(fake_src), exist_ok=True)
        with open(fake_src, "wb") as f:
            f.write(b"x" * (2 * 1024 * 1024))

        with mock.patch.object(easyapp, "fast_parallel_hf_download", return_value=fake_src) as m_dl:
            did_dl = easyapp._download_comfy_model_entry(
                sub="semantic_bridge",
                fname="t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors",
                repo_id="dseditor/minimax-h3-studio-assets",
                repo_rel_path="semantic_bridge/t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors",
                persist_root=self.persist_root,
                comfy_root=self.comfy_root,
                token="",
            )
            self.assertTrue(did_dl)
            final_p = os.path.join(
                self.persist_root, "models", "semantic_bridge",
                "t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors",
            )
            self.assertTrue(os.path.isfile(final_p))

            # Second call should skip already-installed weight without calling downloader
            did_dl2 = easyapp._download_comfy_model_entry(
                sub="semantic_bridge",
                fname="t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors",
                repo_id="dseditor/minimax-h3-studio-assets",
                repo_rel_path="semantic_bridge/t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors",
                persist_root=self.persist_root,
                comfy_root=self.comfy_root,
                token="",
            )
            self.assertFalse(did_dl2)
            self.assertEqual(m_dl.call_count, 1)

    def test_toybox_api_status_and_control_routes(self):
        with mock.patch.object(
            easyapp,
            "_resolve_comfyui_persistence_paths",
            return_value=(self.comfy_root, self.persist_root, "local_only", "测试存储", False),
        ):
            r_status = self.client.get("/minimax_api/toybox/status")
            self.assertEqual(r_status.status_code, 200)
            body = r_status.json()
            self.assertTrue(body["ok"])
            self.assertIn("toybox", body)
            self.assertIn("comfyui", body)
            self.assertEqual(body["toybox"]["ingress_port"], runtime_auth.TOYBOX_INGRESS)

            with mock.patch.object(easyapp, "start_toybox_core", return_value=(True, "ok")) as m_start:
                r_start = self.client.post("/minimax_api/toybox/start", json={"unload_existing": True})
                self.assertEqual(r_start.status_code, 200)
                self.assertIn("toybox", r_start.json())

            with mock.patch.object(easyapp, "stop_toybox_core", return_value=(True, "stopped")) as m_stop:
                r_stop = self.client.post("/minimax_api/toybox/stop", json={"stop_comfyui_too": False})
                self.assertEqual(r_stop.status_code, 200)
                m_stop.assert_called_once_with(stop_comfyui_too=False)

    def test_localize_toybox_to_zh_cn_is_idempotent_and_preserves_mention_compat(self):
        app_dir = os.path.join(self.tmp_dir.name, "toybox")
        static_dir = os.path.join(app_dir, "static")
        os.makedirs(static_dir, exist_ok=True)

        index_html = '<html lang="zh-Hant"><head><title>H3 Toybox</title></head><body><h2>素材庫</h2><p>參考圖解析度預設儲存</p></body></html>'
        app_js = (
            'const KIND_LABEL = { image: "圖片", video: "影片", audio: "音訊" };\n'
            'const MENTION_RE = /@(圖片|图片|Image|影片|视频|Video|音訊|音频|Audio)\\s?(\\d+)/gi;\n'
            'const MENTION_KIND = { "圖片": "image", "图片": "image", "影片": "video", "视频": "video" };\n'
            'const TASK_NAME = { I2VA: "首格", FL2VA: "頭尾格", L2VA: "尾格" };\n'
            'const MSG = "連線中，把資料夾裡的檔案存進雲端硬碟金鑰";\n'
        )
        style_css = "/* 🖼 圖片 / 🎬 影片 / 📁 專案 */\n"
        server_py = (
            'import re\n'
            '# @圖片1 / @影片1 / @音訊1\n'
            'MENTION_RE = re.compile(r"@(圖片|图片|Image|影片|视频|Video|音訊|音频|Audio)\\s?(\\d+)", re.IGNORECASE)\n'
            'MENTION_KIND = {"圖片": "image", "图片": "image"}\n'
            'ERR = "頭尾格模式至少要放一張首格或尾格，否則無法推測解碼。"\n'
        )

        with open(os.path.join(static_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(index_html)
        with open(os.path.join(static_dir, "app.js"), "w", encoding="utf-8") as f:
            f.write(app_js)
        with open(os.path.join(static_dir, "style.css"), "w", encoding="utf-8") as f:
            f.write(style_css)
        with open(os.path.join(app_dir, "server.py"), "w", encoding="utf-8") as f:
            f.write(server_py)

        changed_first = easyapp._localize_toybox_to_zh_cn(app_dir, allow_pip=False)
        self.assertEqual(changed_first, 4)

        changed_second = easyapp._localize_toybox_to_zh_cn(app_dir, allow_pip=False)
        self.assertEqual(changed_second, 0)

        with open(os.path.join(static_dir, "index.html"), encoding="utf-8") as f:
            out_html = f.read()
        with open(os.path.join(static_dir, "app.js"), encoding="utf-8") as f:
            out_js = f.read()
        with open(os.path.join(static_dir, "style.css"), encoding="utf-8") as f:
            out_css = f.read()
        with open(os.path.join(app_dir, "server.py"), encoding="utf-8") as f:
            out_srv = f.read()

        self.assertIn('lang="zh-CN"', out_html)
        self.assertIn("素材库", out_html)
        self.assertIn("参考图分辨率默认保存", out_html)
        self.assertIn('KIND_LABEL = { image: "图片", video: "视频", audio: "音频" }', out_js)
        self.assertIn('I2VA: "首帧", FL2VA: "首尾帧", L2VA: "尾帧"', out_js)
        self.assertIn("连接中，把文件夹里的文件存进Google Drive密钥", out_js)
        # @標籤 compat regex/dicts must keep both Traditional and Simplified aliases
        self.assertIn('@(圖片|图片|Image|影片|视频|Video|音訊|音频|Audio)', out_js)
        self.assertIn('@(圖片|图片|Image|影片|视频|Video|音訊|音频|Audio)', out_srv)
        self.assertIn("项目", out_css)
        self.assertIn("首尾帧模式至少要放一张首帧或尾帧，否则无法投机解码。", out_srv)


if __name__ == "__main__":
    unittest.main()
