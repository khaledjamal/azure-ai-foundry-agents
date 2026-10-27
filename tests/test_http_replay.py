"""Tests for workshop_utils.http_replay: record calls to a local HTTP server, then
replay them with the server unused.

No Azure, no cost. Run from the repo root with the venv that has requests, httpx2, httpx
and aiohttp installed:
    uv run python -m unittest discover -s tests -v
"""

import asyncio
import gzip
import json
import os
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import requests

from workshop_utils import http_log, http_replay

try:
    import httpx2
except ImportError:
    httpx2 = None
try:
    import httpx
except ImportError:
    httpx = None
try:
    import aiohttp
except ImportError:
    aiohttp = None
try:
    from azure.identity import DefaultAzureCredential
    from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
except ImportError:
    DefaultAzureCredential = None


def setUpModule():
    """Use only the settings each test sets: not the shell's HTTP_REPLAY* values, nor
    the developer's .env."""
    kept = {name: value for name, value in os.environ.items() if not name.startswith("HTTP_REPLAY")}
    unittest.enterModuleContext(mock.patch.dict(os.environ, kept, clear=True))
    unittest.enterModuleContext(mock.patch.object(http_replay, "ENV_FILE", Path(os.devnull)))


REPLY = {"output_text": "Hello", "access_token": "secret-token-value"}
CONNECTION = {"name": "search", "credentials": {"type": "ApiKey", "key": "real-search-key"}}
SSE_BODY = 'data: {"delta": "Hel"}\n\ndata: {"delta": "lo"}\n\ndata: [DONE]\n\n'


class Handler(BaseHTTPRequestHandler):
    hits = 0
    polls = 0

    def log_message(self, *args):
        pass

    def send_body(self, body, content_type, headers=None):
        type(self).hits += 1
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/connection"):
            self.send_body(json.dumps(CONNECTION).encode(), "application/json")
        elif self.path.startswith("/run"):
            type(self).polls += 1
            status = "completed" if self.polls >= 3 else "in_progress"
            self.send_body(json.dumps({"id": "run_1", "status": status}).encode(), "application/json")
        elif self.path.startswith("/gzip"):
            body = gzip.compress(json.dumps(REPLY).encode())
            self.send_body(body, "application/json", {"Content-Encoding": "gzip"})
        elif self.path.startswith("/sse"):
            self.send_body(SSE_BODY.encode(), "text/event-stream")
        else:
            self.send_body(json.dumps(REPLY).encode(), "application/json")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        request = json.loads(body or b"{}") if "json" in self.headers.get("Content-Type", "") else {}
        self.send_body(json.dumps({"echo": request.get("input"), **REPLY}).encode(), "application/json")


class ReplayTestCase(unittest.TestCase):
    """A local server, and a fresh recording folder for each test."""

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.folder = self.new_folder()
        Handler.hits = Handler.polls = 0

    def new_folder(self):
        folder = tempfile.TemporaryDirectory(prefix="http-replay-test-")
        self.addCleanup(folder.cleanup)
        return Path(folder.name)

    def tearDown(self):
        self.stop()

    def stop(self):
        http_replay.uninstall()

    def record(self, calls, ignore_fields=()):
        http_replay.install(mode="record", directory=self.folder, ignore_fields=ignore_fields)
        http_log.enable(level="off")
        try:
            return calls()
        finally:
            self.stop()

    def replay(self, calls, ignore_fields=(), level="off"):
        """Run calls() in replay mode and check that the server got no request."""
        hits = Handler.hits
        http_replay.install(mode="replay", directory=self.folder, ignore_fields=ignore_fields)
        http_log.enable(level=level)
        try:
            return calls()
        finally:
            self.stop()
            self.assertEqual(Handler.hits, hits, "the server was called during replay")

    def recording_text(self):
        return "\n".join(path.read_text() for path in sorted(self.folder.glob("*.json")))


class RequestsReplayTest(ReplayTestCase):
    def test_record_then_replay_without_the_server(self):
        def calls():
            first = requests.get(f"{self.base}/reply?api-version=1", timeout=10)
            second = requests.post(f"{self.base}/chat", json={"input": "Hi", "model": "m"}, timeout=10)
            return first.json(), second.json()

        recorded = self.record(calls)
        self.assertEqual(len(list(self.folder.glob("*.json"))), 2)
        replayed = self.replay(calls)
        self.assertEqual(replayed, (_redacted(recorded[0]), _redacted(recorded[1])))
        self.assertEqual(replayed[1]["echo"], "Hi")

    def test_replayed_response_is_marked_and_logged(self):
        self.record(lambda: requests.get(f"{self.base}/reply", timeout=10))
        with self.assertLogs("http_log") as captured:
            response = self.replay(lambda: requests.get(f"{self.base}/reply", timeout=10), level="full")
        self.assertTrue(response.headers[http_replay.REPLAY_HEADER].endswith(".json"))
        self.assertIn("< x-http-replay: ", "\n".join(captured.output))

    def test_secrets_are_redacted_and_the_shape_is_kept(self):
        self.record(
            lambda: requests.get(
                f"{self.base}/connection?api-key=query-key", headers={"Authorization": "Bearer t0ken"}, timeout=10
            )
        )
        text = self.recording_text()
        for secret in ("real-search-key", "query-key", "t0ken"):
            self.assertNotIn(secret, text)
        replayed = self.replay(lambda: requests.get(f"{self.base}/connection?api-key=other", timeout=10).json())
        self.assertEqual(replayed["credentials"], {"type": "ApiKey", "key": "<redacted>"})

    def test_unrecorded_request_fails_and_never_reaches_the_server(self):
        self.record(lambda: requests.get(f"{self.base}/reply", timeout=10))
        with self.assertRaises(http_replay.HttpReplayMiss) as error:
            self.replay(lambda: requests.get(f"{self.base}/other", timeout=10))
        self.assertIn("/other", str(error.exception))

    def test_repeated_requests_get_their_answers_in_order(self):
        def poll():
            return [requests.get(f"{self.base}/run/run_1", timeout=10).json()["status"] for _ in range(3)]

        self.assertEqual(self.record(poll), ["in_progress", "in_progress", "completed"])

        def poll_once_more():
            return poll() + [requests.get(f"{self.base}/run/run_1", timeout=10)]

        self.assertEqual(self.replay(poll), ["in_progress", "in_progress", "completed"])
        with self.assertRaises(http_replay.HttpReplayMiss) as error:
            self.replay(poll_once_more)
        self.assertIn("call 4; 3 recorded", str(error.exception))

    def test_multipart_bodies_match_without_their_random_boundary(self):
        # requests picks a new random boundary for each upload, as azure-core does for
        # file uploads.
        def upload():
            files = {"file": ("notes.md", b"# Notes\n"), "purpose": (None, "assistants")}
            return requests.post(f"{self.base}/files", files=files, timeout=10).status_code

        self.record(upload)
        self.assertEqual(self.replay(upload), 200)
        with self.assertRaises(http_replay.HttpReplayMiss):
            self.replay(lambda: requests.post(f"{self.base}/files", files={"file": ("other.md", b"x")}, timeout=10))

    def test_json_bodies_match_by_value_and_ignored_fields(self):
        self.record(lambda: requests.post(f"{self.base}/chat", json={"input": "Hi", "ts": 1}, timeout=10))
        reordered = '{"ts": 1, "input": "Hi"}'
        headers = {"Content-Type": "application/json"}
        self.replay(lambda: requests.post(f"{self.base}/chat", data=reordered, headers=headers, timeout=10))
        with self.assertRaises(http_replay.HttpReplayMiss):
            self.replay(lambda: requests.post(f"{self.base}/chat", json={"input": "Hi", "ts": 2}, timeout=10))
        self.replay(
            lambda: requests.post(f"{self.base}/chat", json={"input": "Hi", "ts": 2}, timeout=10),
            ignore_fields=["ts"],
        )


@unittest.skipIf(httpx2 is None and httpx is None, "neither httpx2 nor httpx is installed")
class HttpxReplayTest(ReplayTestCase):
    def clients(self):
        return [module for module in (httpx2, httpx) if module is not None]

    def test_streamed_events_and_async_post(self):
        for module in self.clients():
            with self.subTest(module=module.__name__):
                self.folder = self.new_folder()

                def calls(module=module):
                    with module.Client() as client, client.stream("GET", f"{self.base}/sse") as response:
                        lines = [line for line in response.iter_lines() if line]

                    async def post():
                        async with module.AsyncClient() as client:
                            return (await client.post(f"{self.base}/chat", json={"input": "Hi"})).json()

                    return lines, asyncio.run(post())

                recorded = self.record(calls)
                self.assertEqual(len(recorded[0]), 3)
                self.assertEqual(self.replay(calls), (recorded[0], _redacted(recorded[1])))

    def test_replayed_body_is_logged_when_read(self):
        # The log must see a replayed body as the caller reads it, as with a real one,
        # not only at exit.
        for module in self.clients():
            with self.subTest(module=module.__name__):
                self.folder = self.new_folder()

                def calls(module=module):
                    with module.Client() as client:
                        sync_reply = client.get(f"{self.base}/reply").json()

                    async def post():
                        async with module.AsyncClient() as client:
                            return (await client.post(f"{self.base}/chat", json={"input": "Hi"})).json()

                    return sync_reply, asyncio.run(post())

                self.record(calls)
                with self.assertLogs("http_log") as captured:
                    self.replay(calls, level="full")
                log = "\n".join(captured.output)
                self.assertEqual(log.count("--- OUTBOUND RESPONSE #"), 2)
                self.assertIn('"output_text": "Hello"', log)
                self.assertIn('"echo": "Hi"', log)
                self.assertNotIn("body not read", log)


@unittest.skipIf(aiohttp is None, "aiohttp is not installed")
class AiohttpReplayTest(ReplayTestCase):
    def test_azure_core_style_reads(self):
        # azure-core: auto_decompress=False, read(), and decompression only when
        # Content-Encoding says so.
        def calls():
            async def main():
                async with aiohttp.ClientSession(auto_decompress=False) as session:
                    response = await session.request("GET", f"{self.base}/gzip")
                    body = await response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    response.release()
                    data = b'{"input": "Hi"}'
                    headers = {"Content-Type": "application/json"}
                    async with session.post(f"{self.base}/chat", data=data, headers=headers) as response:
                        return json.loads(body), await response.json()

            return asyncio.run(main())

        recorded = self.record(calls)
        replayed = self.replay(calls)
        self.assertEqual(replayed, (_redacted(recorded[0]), _redacted(recorded[1])))


class ReplaySafetyTest(ReplayTestCase):
    def test_network_is_blocked_but_loopback_works(self):
        self.record(lambda: requests.get(f"{self.base}/reply", timeout=10))

        def calls():
            with self.assertRaises(http_replay.HttpReplayNetworkBlocked):
                socket.create_connection(("192.0.2.1", 443), timeout=1)
            with self.assertRaises(http_replay.HttpReplayNetworkBlocked):
                socket.getaddrinfo("example.com", 443)
            with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=1):
                pass

        self.replay(calls)
        # The guard is gone after replay.
        self.assertTrue(socket.getaddrinfo("localhost", 80))

    @unittest.skipIf(DefaultAzureCredential is None, "azure-identity is not installed")
    def test_azure_credentials_give_a_placeholder_token(self):
        self.record(lambda: requests.get(f"{self.base}/reply", timeout=10))

        def calls():
            token = DefaultAzureCredential().get_token("https://ai.azure.com/.default").token

            async def get_async():
                async with AsyncDefaultAzureCredential() as credential:
                    return (await credential.get_token("https://ai.azure.com/.default")).token

            return token, asyncio.run(get_async())

        self.assertEqual(self.replay(calls), (http_replay.PLACEHOLDER_TOKEN, http_replay.PLACEHOLDER_TOKEN))

    def test_replay_needs_a_recording(self):
        with self.assertRaises(FileNotFoundError):
            http_replay.install(mode="replay", directory=self.folder)

    def test_off_by_default(self):
        http_log.enable(level="off")
        self.assertEqual(http_replay.mode(), "off")

    def with_env_file(self, text):
        """Point the module at a .env with this text and a repo root in the test
        folder."""
        env_file = self.folder / ".env"
        env_file.write_text(text)
        for name, value in (("ENV_FILE", env_file), ("REPO_ROOT", self.folder)):
            patcher = mock.patch.object(http_replay, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_env_file_turns_recording_on_with_a_default_folder(self):
        self.with_env_file("HTTP_REPLAY=record\n")
        http_log.enable(level="off")
        self.assertEqual(http_replay.mode(), "record")
        folders = list((self.folder / "runs" / "recordings").iterdir())
        self.assertEqual(len(folders), 1)
        self.assertRegex(folders[0].name, r"-\d{8}-\d{6}$")
        info = (folders[0] / "recording-info.txt").read_text()
        for field in ("date: ", "command: ", "git commit: ", "python: ", "packages:"):
            self.assertIn(field, info)

    def test_shell_setting_wins_over_env_file(self):
        self.with_env_file("HTTP_REPLAY=record\n")
        with mock.patch.dict(os.environ, {"HTTP_REPLAY": "off"}):
            http_log.enable(level="off")
        self.assertEqual(http_replay.mode(), "off")

    def test_record_into_a_given_folder_writes_recording_info(self):
        self.record(lambda: requests.get(f"{self.base}/reply", timeout=10))
        self.assertTrue((self.folder / "recording-info.txt").exists())
        self.assertEqual(len(list(self.folder.glob("*.json"))), 1)


def _redacted(reply):
    """A reply as replayed: secret fields are "<redacted>" in the recording."""
    return {key: "<redacted>" if key == "access_token" else value for key, value in reply.items()}


if __name__ == "__main__":
    unittest.main()
