"""Drive 持久化不得把 Colab 主账号令牌上传到 VM：python3 -m unittest tests.test_drive_conf_isolation"""

import ast
import json
import os
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_PATH = os.path.join(REPO_ROOT, "colab_bridge.py")
WANTED = {"_main_account_refresh_token", "_build_rclone_drive_conf"}


def _load(token_path, conf_path):
    with open(BRIDGE_PATH, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=BRIDGE_PATH)
    nodes = [
        n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in WANTED)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_DRIVE_RCLONE_KEYS" for t in n.targets))
    ]
    ns = {
        "os": os, "json": json,
        "TOKEN_CONFIG_PATH": token_path, "DRIVE_RCLONE_CONF_PATH": conf_path,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), BRIDGE_PATH, "exec"), ns)
    return ns["_build_rclone_drive_conf"]


MAIN_TOKEN = {
    "token": "ya29.MAIN", "refresh_token": "1//MAIN_REFRESH",
    "client_id": "main-client.apps.googleusercontent.com", "client_secret": "MAIN_SECRET",
    "scopes": ["https://www.googleapis.com/auth/cloud-platform",
               "https://www.googleapis.com/auth/drive.file"],
}


def _drive_conf(refresh, scope="drive.file", extra=""):
    tok = json.dumps({"access_token": "ya29.SIDE", "refresh_token": refresh, "token_type": "Bearer"})
    return f"[side]\ntype = drive\nscope = {scope}\ntoken = {tok}\n{extra}"


class DriveConfIsolationTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.token_path = os.path.join(self.dir.name, "token.json")
        self.conf_path = os.path.join(self.dir.name, "drive_rclone.conf")
        with open(self.token_path, "w", encoding="utf-8") as f:
            json.dump(MAIN_TOKEN, f)
        self.build = _load(self.token_path, self.conf_path)

    def tearDown(self):
        self.dir.cleanup()

    def _write(self, text):
        with open(self.conf_path, "w", encoding="utf-8") as f:
            f.write(text)

    def test_main_account_token_never_used_without_side_conf(self):
        text, note = self.build()
        self.assertEqual(text, "")
        self.assertIn("跳过", note)

    def test_side_account_conf_is_uploaded(self):
        self._write(_drive_conf("1//SIDE_REFRESH"))
        text, _ = self.build()
        self.assertIn("[easycolab_gdrive]", text)
        self.assertIn("1//SIDE_REFRESH", text)
        self.assertIn("scope = drive.file", text)
        for secret in ("MAIN_REFRESH", "ya29.MAIN", "MAIN_SECRET", "main-client"):
            self.assertNotIn(secret, text)

    def test_reused_main_refresh_token_rejected(self):
        self._write(_drive_conf("1//MAIN_REFRESH"))
        self.assertEqual(self.build()[0], "")

    def test_full_drive_scope_rejected(self):
        self._write(_drive_conf("1//SIDE_REFRESH", scope="drive"))
        self.assertEqual(self.build()[0], "")

    def test_unknown_keys_dropped(self):
        self._write(_drive_conf("1//SIDE_REFRESH", extra="service_account_file = /etc/passwd\n"))
        text, _ = self.build()
        self.assertTrue(text)
        self.assertNotIn("service_account_file", text)


if __name__ == "__main__":
    unittest.main()
