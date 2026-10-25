"""Record web service calls to JSON files once, then replay them: the same code runs again without paying again.

Settings (environment variables, read when http_log.enable() runs):
    HTTP_REPLAY                off (default) | record | replay
    HTTP_REPLAY_DIR            folder of the recording (one folder per scenario); needed for record and replay
    HTTP_REPLAY_IGNORE_FIELDS  JSON body fields left out when requests are matched (comma-separated names, at any
                               depth); default none

record: every call is made for real, and each finished call is saved as one JSON file in HTTP_REPLAY_DIR: the
request (method, URL, headers, body) and the whole response (status, headers, body, never shortened; a server-sent
event stream is kept as its text). Secrets are redacted (authorization and key headers, key query parameters, token,
key and credential fields in JSON bodies); the shape of a redacted object is kept, so code that reads
`credentials.key` still works on replay. Real hosts and IDs stay as they are: recordings are private files.

replay: nothing leaves the machine. Each request is matched to a recorded one by method, URL (host, path and sorted
query) and body (JSON compared by value, HTTP_REPLAY_IGNORE_FIELDS left out; a multipart upload without its random
boundary); headers are not compared (auth, request IDs, dates, user agent). When the same request was made several
times, the 1st replay gets the 1st recorded answer, the 2nd the 2nd, and so on (an agent run polled until it ends, for
example). A request with no recorded answer raises
HttpReplayMiss naming it; there is no fall-back to the real service, so a replay can't cost money. Also in replay:
connections to anything but this machine (loopback) are refused (HttpReplayNetworkBlocked), and the azure-identity
credentials (DefaultAzureCredential, AzureCliCredential, sync and async) return a placeholder token, so no sign-in or
token request is needed. Replayed responses carry the header `x-http-replay: <file name>`.

Covered: requests, httpx2 and httpx, and aiohttp (bodies read with `read()`, as azure-core does). Not covered: other
protocols such as AMQP (Event Hubs) or gRPC, so a script that uses them must not run in record or replay mode.

Limits: a replay shows that the code still handles the recorded answers, not that the service still answers the same
way. A changed prompt, model or SDK version changes the requests and needs a new recording; so do random or
time-based values in request bodies (seed them, or list them in HTTP_REPLAY_IGNORE_FIELDS).

Started by http_log.enable(); `install()` is safe to call more than once.
"""

import base64
import hashlib
import io
import ipaddress
import json
import os
import re
import socket
import threading
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

from workshop_utils import http_log

MODES = ("off", "record", "replay")
REPLAY_HEADER = "x-http-replay"
# Response headers that describe the bytes on the wire; recordings keep the decompressed body, so they are dropped.
WIRE_HEADERS = {"content-length", "content-encoding", "transfer-encoding"}
PLACEHOLDER_TOKEN = "http-replay-placeholder-token"
KEPT_IN_SECRETS = {"type", "kind"}  # fields kept as they are inside a redacted object


class HttpReplayMiss(RuntimeError):
    """A request in replay mode that has no recorded answer."""


class HttpReplayNetworkBlocked(ConnectionRefusedError):
    """A connection to another machine attempted in replay mode."""


_state = {"mode": "off", "dir": None, "ignore": frozenset()}
_lock = threading.Lock()
_recorded = {}  # replay: match key -> list of recording files, in recorded order
_used = {}  # replay: match key -> how many of them have been replayed
_originals = {}  # (owner, name) -> original attribute, to restore in uninstall()


def mode():
    return _state["mode"]


def install(mode=None, directory=None, ignore_fields=None):
    """Start record or replay as HTTP_REPLAY (or the arguments) say; does nothing when the mode is off."""
    mode = (mode or os.environ.get("HTTP_REPLAY", "off")).strip().lower() or "off"
    if mode not in MODES:
        raise ValueError(f"HTTP_REPLAY must be one of {', '.join(MODES)}, not {mode!r}")
    if mode == "off":
        return
    if _state["mode"] != "off":
        if mode != _state["mode"]:
            raise RuntimeError(f"http_replay is already in {_state['mode']} mode")
        return
    directory = directory or os.environ.get("HTTP_REPLAY_DIR")
    if not directory:
        raise ValueError(f"HTTP_REPLAY={mode} needs HTTP_REPLAY_DIR (the recording's folder)")
    if ignore_fields is None:
        ignore_fields = [f for f in os.environ.get("HTTP_REPLAY_IGNORE_FIELDS", "").split(",") if f.strip()]
    _state.update(mode=mode, dir=Path(directory), ignore=frozenset(f.strip() for f in ignore_fields))
    if mode == "record":
        _state["dir"].mkdir(parents=True, exist_ok=True)
        http_log.add_observer(_save)
    else:
        _load()
        # The replay hooks go under the logging hooks, so a replayed call is logged like a real one.
        logging_on = bool(http_log._originals)
        http_log.disable()
        _guard_network()
        _patch_credentials()
        _patch_requests()
        for name in ("httpx2", "httpx"):
            _patch_httpx(name)
        _patch_aiohttp()
        if logging_on:
            http_log._install_hooks()
    print(f"[http_replay] {mode} mode, folder {_state['dir']}")


def uninstall():
    """Stop recording or replaying (used by the tests). Also stops logging, whose hooks sit on top of the replay."""
    http_log.disable()
    http_log.remove_observer(_save)
    for (owner, name), original in _originals.items():
        setattr(owner, name, original)
    _originals.clear()
    _recorded.clear()
    _used.clear()
    _state.update(mode="off", dir=None, ignore=frozenset())


# ---------------------------------------------------------------------------------------------------------------
# Matching: the same key in both modes.


def match_key(method, url, body):
    """Method, host, path, sorted query and the canonical body of a request, as one string."""
    parts = urlsplit(http_log._redact_url(str(url)))
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return f"{method.upper()} {parts.scheme}://{parts.netloc}{parts.path}?{query}\n{_canonical_body(body)}"


def _canonical_body(body):
    if not body:
        return ""
    if isinstance(body, str):  # a note such as "<streamed body, not logged>"
        return body
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        pass
    else:
        data = _drop_fields(_redact_secrets(data), _state["ignore"])
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return "sha256:" + hashlib.sha256(body).hexdigest()
    if re.fullmatch(r"[\w.~%+\-]+=[^&]*(&[\w.~%+\-]+=[^&]*)*", text):  # form-encoded
        return http_log._redact_params("&".join(sorted(text.split("&"))))
    multipart = re.match(r"--([0-9A-Za-z'()+_,./:=?-]{1,70})\r\n", text)
    if multipart:  # file uploads: the boundary is random for each request, so it is left out of the match
        return text.replace(multipart.group(1), "<boundary>")
    return text


def _drop_fields(value, names):
    if not names:
        return value
    if isinstance(value, dict):
        return {k: _drop_fields(v, names) for k, v in value.items() if k not in names}
    if isinstance(value, list):
        return [_drop_fields(v, names) for v in value]
    return value


def _redact_secrets(value, inside_secret=False, key=None):
    """Replace secret values with "<redacted>", keeping the shape of objects under a secret field name and their
    `type` / `kind` fields (an SDK picks the credential class from them)."""
    if isinstance(value, dict):
        return {
            k: _redact_secrets(v, inside_secret or k.lower() in http_log.SECRET_FIELDS, k) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(v, inside_secret, key) for v in value]
    if inside_secret and isinstance(value, str) and (key or "").lower() not in KEPT_IN_SECRETS:
        return http_log.REDACTED
    return value


# ---------------------------------------------------------------------------------------------------------------
# Record: http_log hands every finished call to _save.


def _save(call, body, note):
    content_type = http_log._header(call.headers, "content-type") or ""
    # A note means the body wasn't read to the end, except for answers that have no body.
    bodiless = call.status in (204, 304) or call.method == "HEAD"
    if call.status is None or (note and not (bodiless and not call.received)):
        print(f"[http_replay] NOT RECORDED (body not complete: {note}): {call.method} {call.url}")
        return
    record = {
        "number": call.number,
        "request": {
            "method": call.method,
            "url": call.url,
            "headers": [[n, http_log._redact_header(n, v)] for n, v in call.request_headers],
            **_body_fields(call.request_body, http_log._header(call.request_headers, "content-type") or ""),
        },
        "response": {
            "status": call.status,
            "reason": call.reason,
            "headers": [[n, http_log._redact_header(n, v)] for n, v in call.headers if n.lower() not in WIRE_HEADERS],
            **_body_fields(body, content_type),
        },
    }
    path = urlsplit(call.url).path.strip("/").replace("/", "_")[:60] or "root"
    name = f"{call.number:04d}-{call.method}-{re.sub(r'[^A-Za-z0-9_.-]', '-', path)}.json"
    with _lock:
        (_state["dir"] / name).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def _body_fields(body, content_type):
    """The body as JSON (body_json), text (body_text) or base64 (body_base64), secrets redacted."""
    if not body:
        return {}
    if isinstance(body, str):
        return {"body_note": body}
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return {"body_base64": base64.b64encode(body).decode()}
    if "json" in content_type.lower() or (not content_type and text.lstrip()[:1] in "{["):
        try:
            return {"body_json": _redact_secrets(json.loads(text))}
        except ValueError:
            pass
    if "x-www-form-urlencoded" in content_type.lower():
        text = http_log._redact_params(text)
    return {"body_text": text}


def _body_bytes(fields):
    if "body_note" in fields:
        return fields["body_note"]
    if "body_json" in fields:
        return json.dumps(fields["body_json"], ensure_ascii=False).encode()
    if "body_text" in fields:
        return fields["body_text"].encode()
    if "body_base64" in fields:
        return base64.b64decode(fields["body_base64"])
    return b""


# ---------------------------------------------------------------------------------------------------------------
# Replay: load the recording, then answer each request from it.


def _load():
    files = sorted(_state["dir"].glob("*.json"), key=lambda p: (json.loads(p.read_text())["number"], p.name))
    if not files:
        raise FileNotFoundError(f"no recorded calls in {_state['dir']}")
    for path in files:
        # The key is computed now, from the saved request, so HTTP_REPLAY_IGNORE_FIELDS can change after recording.
        request = json.loads(path.read_text())["request"]
        key = match_key(request["method"], request["url"], _body_bytes(request))
        _recorded.setdefault(hashlib.sha256(key.encode()).hexdigest(), []).append(path)


def _answer(method, url, body):
    """(status, reason, headers, body bytes) of the recorded answer to this request, or HttpReplayMiss."""
    key = match_key(method, url, body)
    digest = hashlib.sha256(key.encode()).hexdigest()
    with _lock:
        count = _used.get(digest, 0)
        files = _recorded.get(digest, [])
        if count >= len(files):
            occurrence = f" (call {count + 1}; {len(files)} recorded)" if files else ""
            message = f"no recorded answer for {method} {http_log._redact_url(str(url))}{occurrence} in {_state['dir']}"
            print(f"[http_replay] MISS: {message}")
            raise HttpReplayMiss(message)
        _used[digest] = count + 1
        path = files[count]
    response = json.loads(path.read_text())["response"]
    body_bytes = _body_bytes(response)
    headers = [(n, v) for n, v in response["headers"]] + [
        ("Content-Length", str(len(body_bytes))),
        (REPLAY_HEADER, path.name),
    ]
    return response["status"], response["reason"], headers, body_bytes


def _patch(owner, name, replacement):
    _originals[(owner, name)] = getattr(owner, name)
    setattr(owner, name, replacement)


def _guard_network():
    """Refuse connections and name lookups for anything but this machine."""

    def local(host):
        if host in (None, "", "localhost") or isinstance(host, bytes) and host == b"localhost":
            return True
        try:
            return ipaddress.ip_address(host.decode() if isinstance(host, bytes) else host).is_loopback
        except ValueError:
            return False

    def check(address):
        if isinstance(address, tuple) and not local(address[0]):
            raise HttpReplayNetworkBlocked(f"HTTP_REPLAY=replay: connection to {address[0]} refused")

    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def connect(self, address):
        check(address)
        return original_connect(self, address)

    def connect_ex(self, address):
        check(address)
        return original_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if not local(host):
            raise HttpReplayNetworkBlocked(f"HTTP_REPLAY=replay: name lookup of {host} refused")
        return original_getaddrinfo(host, *args, **kwargs)

    _patch(socket.socket, "connect", connect)
    _patch(socket.socket, "connect_ex", connect_ex)
    _patch(socket, "getaddrinfo", getaddrinfo)


def _patch_credentials():
    try:
        import azure.identity
        import azure.identity.aio
        from azure.core.credentials import AccessToken
    except ImportError:
        return
    try:
        from azure.core.credentials import AccessTokenInfo
    except ImportError:  # azure-core before 1.31
        AccessTokenInfo = None
    expires = 4102444800  # 2100-01-01

    def get_token(self, *scopes, **kwargs):
        return AccessToken(PLACEHOLDER_TOKEN, expires)

    def get_token_info(self, *scopes, **kwargs):
        return AccessTokenInfo(PLACEHOLDER_TOKEN, expires)

    async def get_token_async(self, *scopes, **kwargs):
        return get_token(self)

    async def get_token_info_async(self, *scopes, **kwargs):
        return get_token_info(self)

    for module, token, info in (
        (azure.identity, get_token, get_token_info),
        (azure.identity.aio, get_token_async, get_token_info_async),
    ):
        for name in ("DefaultAzureCredential", "AzureCliCredential"):
            cls = getattr(module, name)
            _patch(cls, "get_token", token)
            if AccessTokenInfo is not None and hasattr(cls, "get_token_info"):
                _patch(cls, "get_token_info", info)


def _patch_requests():
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.response import HTTPResponse
    except ImportError:
        return

    def send(self, request, **kwargs):
        body = request.body.encode() if isinstance(request.body, str) else request.body
        status, reason, headers, data = _answer(request.method, request.url, body)
        raw = HTTPResponse(body=io.BytesIO(data), headers=headers, status=status, reason=reason, preload_content=False)
        return self.build_response(request, raw)

    _patch(HTTPAdapter, "send", send)


def _patch_httpx(module_name):
    try:
        module = __import__(module_name)
    except ImportError:
        return

    class ReplayStream(module.SyncByteStream, module.AsyncByteStream):
        # A stream, not content=: httpx reads content= at once, before http_log wraps the stream, so the log would
        # never see the body being read. A stream is read when the caller reads it, as a real one is.
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            yield self.data

        async def __aiter__(self):
            yield self.data

    def respond(module, request, answer):
        status, reason, headers, data = answer
        extensions = {"reason_phrase": reason.encode()} if reason else {}
        return module.Response(
            status, headers=headers, stream=ReplayStream(data), request=request, extensions=extensions
        )

    def handle_request(self, request):
        return respond(module, request, _answer(request.method, request.url, request.read()))

    async def handle_async_request(self, request):
        return respond(module, request, _answer(request.method, request.url, await request.aread()))

    _patch(module.HTTPTransport, "handle_request", handle_request)
    _patch(module.AsyncHTTPTransport, "handle_async_request", handle_async_request)


def _patch_aiohttp():
    try:
        import aiohttp
        from multidict import CIMultiDict, CIMultiDictProxy
        from yarl import URL
    except ImportError:
        return

    class Content:
        """The part of aiohttp's StreamReader that azure-core uses for streamed bodies."""

        def __init__(self, data):
            self._data = io.BytesIO(data)

        async def read(self, n=-1):
            return self._data.read(n)

        async def readany(self):
            return self._data.read()

        def at_eof(self):
            return self._data.tell() == len(self._data.getbuffer())

    class ReplayedResponse:
        """Stands in for aiohttp.ClientResponse with a recorded answer."""

        def __init__(self, method, url, answer):
            status, reason, headers, data = answer
            self.method, self.url, self.status, self.reason = method, url, status, reason
            self.headers = CIMultiDictProxy(CIMultiDict(headers))
            self.raw_headers = tuple((n.encode("latin-1"), v.encode("latin-1")) for n, v in headers)
            self.content_type = self.headers.get("Content-Type", "application/octet-stream").split(";")[0]
            self.content = Content(data)
            self.ok = status < 400
            self._body = data

        async def read(self):
            return self._body

        async def text(self, encoding="utf-8"):
            return self._body.decode(encoding)

        async def json(self, **kwargs):
            return json.loads(self._body)

        def raise_for_status(self):
            if not self.ok:
                raise aiohttp.ClientResponseError(None, (), status=self.status, message=self.reason)

        def release(self):
            pass

        def close(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    async def _request(self, method, str_or_url, **kwargs):
        url = URL(str_or_url)
        if kwargs.get("params"):
            url = url.extend_query(kwargs["params"])
        data, headers = kwargs.get("data"), kwargs.get("headers")
        if kwargs.get("json") is not None:
            body = json.dumps(kwargs["json"]).encode()
        elif isinstance(data, str):
            body = data.encode()
        elif isinstance(data, (bytes, bytearray)):
            body = bytes(data)
        elif isinstance(data, dict):
            body = urlencode(data).encode()
        else:
            body = None if data is None else "<streamed body, not logged>"
        header_list = list((headers or {}).items())
        call = http_log._Call(method, url, header_list, body, http_log._header(header_list, "content-type"))
        try:
            answer = _answer(method, url, body)
        except HttpReplayMiss as error:
            call.failed(error)
            raise
        status, reason, answer_headers, answer_body = answer
        call.response(status, reason, answer_headers)
        call.record(answer_body)
        call.finish()
        return ReplayedResponse(method, url, answer)

    _patch(aiohttp.ClientSession, "_request", _request)
