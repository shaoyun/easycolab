"""Tests for OpenAI-compatible APIs (/v1/models, /v1/chat/completions, /v1/images/*, /v1/videos*) and token/login auth isolation."""

import base64
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

os.environ["EASYCOLAB_LOCAL_PREVIEW"] = "1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import app as easyapp  # noqa: E402
import runtime_auth  # noqa: E402


def _make_tiny_png_bytes(color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return buf.getvalue()


class OpenAIGatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fastapi_app = FastAPI()
        easyapp.attach_macos_api_routes(cls.fastapi_app)
        cls.client = TestClient(cls.fastapi_app)

    def setUp(self):
        self.orig_key = easyapp.API_AUTH_KEY
        self.orig_extra = list(easyapp.EXTRA_API_TOKENS)
        self.orig_runtimes = dict(easyapp.LOADED_RUNTIMES)
        self.orig_loaded_id = easyapp.LOADED_MODEL_ID
        self.orig_out_dir = easyapp.DEFAULT_OUTPUT_DIR
        self.tmp_dir = tempfile.TemporaryDirectory()
        easyapp.DEFAULT_OUTPUT_DIR = self.tmp_dir.name
        with easyapp.API_KEY_LOCK:
            easyapp.API_AUTH_KEY = "sk-test-primary-token"
            easyapp.EXTRA_API_TOKENS = ["sk-test-secondary-token"]
        easyapp.LOADED_RUNTIMES.clear()
        easyapp.LOADED_MODEL_ID = None
        easyapp.JOBS.clear()

    def tearDown(self):
        with easyapp.API_KEY_LOCK:
            easyapp.API_AUTH_KEY = self.orig_key
            easyapp.EXTRA_API_TOKENS = self.orig_extra
        easyapp.LOADED_RUNTIMES.clear()
        easyapp.LOADED_RUNTIMES.update(self.orig_runtimes)
        easyapp.LOADED_MODEL_ID = self.orig_loaded_id
        easyapp.DEFAULT_OUTPUT_DIR = self.orig_out_dir
        easyapp.JOBS.clear()
        self.tmp_dir.cleanup()

    def test_token_auth_isolation_and_cors_preflight(self):
        # 1. Missing or Basic login header alone must not authenticate OpenAI routes
        r_no = self.client.get("/v1/models")
        self.assertEqual(r_no.status_code, 401)
        self.assertNotIn("basic", r_no.headers.get("www-authenticate", "").lower())

        basic_hdr = "Basic " + base64.b64encode(b"easycolab:admin-login-pass").decode()
        r_basic = self.client.get("/v1/models", headers={"Authorization": basic_hdr})
        self.assertEqual(r_basic.status_code, 401)

        # 2. x-api-key succeeds even when Basic login header is also sent (no conflict)
        r_both = self.client.get(
            "/v1/models",
            headers={"Authorization": basic_hdr, "x-api-key": "sk-test-primary-token"},
        )
        self.assertEqual(r_both.status_code, 200)
        self.assertEqual(r_both.headers.get("access-control-allow-origin"), "*")

        # 3. Secondary token & Bearer auth work
        r_sec = self.client.get("/v1/models", headers={"Authorization": "Bearer sk-test-secondary-token"})
        self.assertEqual(r_sec.status_code, 200)

        # 4. CORS preflight works without token, while non-inference admin route stays untouched
        r_opt = self.client.options("/v1/videos")
        self.assertEqual(r_opt.status_code, 204)
        self.assertEqual(r_opt.headers.get("access-control-allow-origin"), "*")
        self.assertIn("POST", r_opt.headers.get("access-control-allow-methods", ""))

        r_admin = self.client.get("/minimax_api/status")
        self.assertEqual(r_admin.status_code, 200)
        self.assertIsNone(r_admin.headers.get("access-control-allow-origin"))

    def test_models_list_and_retrieve_includes_ref2va(self):
        headers = {"Authorization": "Bearer sk-test-primary-token"}
        r = self.client.get("/v1/models", headers=headers)
        self.assertEqual(r.status_code, 200)
        ids = {item["id"] for item in r.json()["data"]}
        self.assertIn("MiniMaxAI/MiniMax-H3", ids)
        self.assertIn("MiniMaxAI/MiniMax-H3-Ref2VA", ids)
        self.assertIn("MiniMaxAI/MiniMax-H3-Image", ids)

        r_single = self.client.get("/v1/models/MiniMaxAI/MiniMax-H3-Ref2VA", headers=headers)
        self.assertEqual(r_single.status_code, 200)
        self.assertEqual(r_single.json()["id"], "MiniMaxAI/MiniMax-H3-Ref2VA")

        r_missing = self.client.get("/v1/models/nonexistent-model-xyz", headers=headers)
        self.assertEqual(r_missing.status_code, 404)

    def test_images_generations_edits_and_file_download(self):
        headers = {"Authorization": "Bearer sk-test-primary-token"}
        easyapp.LOADED_RUNTIMES["minimax_h3_image"] = {
            "name": "MiniMax-H3-Image",
            "modality": "image",
            "pipeline": object(),
        }

        def _fake_h3_img(active_pipe, prompt, width, height, seed, ref_images=None):
            color = (0, 255, 0) if ref_images else (0, 0, 255)
            return Image.new("RGB", (64, 64), color=color)

        with mock.patch.object(easyapp, "run_minimax_h3_image_inference", side_effect=_fake_h3_img) as m_inf:
            r_gen = self.client.post(
                "/v1/images/generations",
                headers=headers,
                json={"model": "MiniMaxAI/MiniMax-H3-Image", "prompt": "a blue mountain", "size": "1024x1024", "response_format": "url"},
            )
            self.assertEqual(r_gen.status_code, 200, r_gen.text)
            data = r_gen.json()["data"]
            self.assertEqual(len(data), 1)
            self.assertIn("/v1/images/files/", data[0]["url"])
            self.assertNotIn("b64_json", data[0])

            # Download saved image via query token (for external <img> tags)
            dl_path = data[0]["url"].split("testserver")[-1]
            r_dl = self.client.get(dl_path)
            self.assertEqual(r_dl.status_code, 200)
            self.assertEqual(r_dl.headers["content-type"], "image/png")

            # Multipart /v1/images/edits
            png_bytes = _make_tiny_png_bytes()
            r_edit = self.client.post(
                "/v1/images/edits",
                headers=headers,
                data={"model": "MiniMaxAI/MiniMax-H3-Image", "prompt": "turn it green", "size": "1024x1024"},
                files={"image": ("ref.png", png_bytes, "image/png")},
            )
            self.assertEqual(r_edit.status_code, 200, r_edit.text)
            self.assertIn("b64_json", r_edit.json()["data"][0])
            self.assertTrue(m_inf.call_args.kwargs["ref_images"])

    def test_videos_api_create_poll_content_delete(self):
        headers = {"Authorization": "Bearer sk-test-primary-token"}

        # 503 when video model not loaded
        r_unloaded = self.client.post(
            "/v1/videos",
            headers=headers,
            json={"model": "MiniMaxAI/MiniMax-H3-Ref2VA", "prompt": "A singing girl <Picture 1>"},
        )
        self.assertEqual(r_unloaded.status_code, 503)

        easyapp.LOADED_RUNTIMES["minimax_h3_ref2va"] = {
            "name": "MiniMax-H3-Ref2VA",
            "modality": "video",
            "pipeline": object(),
        }

        b64_img = base64.b64encode(_make_tiny_png_bytes()).decode("ascii")
        fake_mp4 = os.path.join(self.tmp_dir.name, "test_out.mp4")
        with open(fake_mp4, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42FAKEVIDEO")

        def _fake_bg_job(job_id, req):
            easyapp.JOBS[job_id]["status"] = "completed"
            easyapp.JOBS[job_id]["progress"] = 1.0
            easyapp.JOBS[job_id]["filename"] = "test_out.mp4"
            easyapp.JOBS[job_id]["completed_at"] = 1700000010
            easyapp.JOBS[job_id]["message"] = "生成完成"

        with mock.patch.object(easyapp, "_run_async_generation_job", side_effect=_fake_bg_job) as m_job:
            r_create = self.client.post(
                "/v1/videos",
                headers=headers,
                json={
                    "model": "MiniMaxAI/MiniMax-H3-Ref2VA",
                    "prompt": "<Picture 1> smiling in rain",
                    "size": "832x480",
                    "seconds": "5",
                    "ref_images": [f"data:image/png;base64,{b64_img}"],
                },
            )
            self.assertEqual(r_create.status_code, 200, r_create.text)
            vid_obj = r_create.json()
            self.assertEqual(vid_obj["object"], "video")
            self.assertEqual(vid_obj["task_mode"], "ref2va")
            vid_id = vid_obj["id"]
            self.assertTrue(vid_id.startswith("video_"))
            _, submitted_req = m_job.call_args[0]
            self.assertEqual(submitted_req.task_mode, "ref2va")
            self.assertEqual(len(submitted_req.ref_images_base64), 1)
            self.assertEqual(submitted_req.num_frames % 17, 5)

        # Wait briefly for daemon thread completion
        r_status = self.client.get(f"/v1/videos/{vid_id}", headers=headers)
        self.assertEqual(r_status.status_code, 200)
        self.assertEqual(r_status.json()["status"], "completed")
        self.assertEqual(r_status.json()["progress"], 100)

        r_list = self.client.get("/v1/videos", headers=headers)
        self.assertEqual(r_list.status_code, 200)
        self.assertEqual(r_list.json()["data"][0]["id"], vid_id)

        r_content = self.client.get(f"/v1/videos/{vid_id}/content?api_key=sk-test-secondary-token")
        self.assertEqual(r_content.status_code, 200)
        self.assertEqual(r_content.headers["content-type"], "video/mp4")
        self.assertIn(b"FAKEVIDEO", r_content.content)

        r_del = self.client.delete(f"/v1/videos/{vid_id}", headers=headers)
        self.assertEqual(r_del.status_code, 200)
        self.assertTrue(r_del.json()["deleted"])

    def test_chat_completions_routes_image_model(self):
        headers = {"Authorization": "Bearer sk-test-primary-token"}
        easyapp.LOADED_RUNTIMES["z_image_turbo"] = {
            "name": "Z-Image-Turbo",
            "modality": "image",
            "pipeline": lambda **kw: Image.new("RGB", (32, 32), color=(255, 255, 0)),
        }
        r = self.client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "Tongyi-MAI/Z-Image-Turbo",
                "messages": [{"role": "user", "content": "a golden sunset"}],
                "stream": False,
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        msg = r.json()["choices"][0]["message"]["content"]
        self.assertIn("/v1/images/files/", msg)

    def test_nginx_config_separates_login_and_inference_cors(self):
        cfg = runtime_auth.nginx_config("/tmp/authdir")
        # CSRF guards live inside location / rather than server root, so cross-site /v1/* works
        self.assertIn("location = /v1/videos {", cfg)
        self.assertIn("location = /v1/images/edits {", cfg)
        self.assertIn("Access-Control-Allow-Origin", cfg)
        self.assertIn("location ~ ^(?:/v1)?/videos/[A-Za-z0-9_-]+(?:/content)?$ {", cfg)


if __name__ == "__main__":
    unittest.main()
