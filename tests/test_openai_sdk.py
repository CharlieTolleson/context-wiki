#!/usr/bin/env python3
"""Offline smoke test of the real OpenAI SDK against a local stub of the LiteLLM endpoint.

test_ingest.py swaps openai.OpenAI for a fake, so it cannot notice an SDK release that
changes the client constructor, chat.completions.create, or the response shape. This file
leaves the SDK in place and points it at a throwaway HTTP server on 127.0.0.1 that answers
like LiteLLM's /v1/chat/completions. No network beyond loopback, no LLM call, no wiki touched.

Run:  python tests/test_openai_sdk.py        (or: pytest tests/)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Same isolation as test_ingest.py: no real .env location can be written by this test.
_ISOLATED = tempfile.mkdtemp(prefix="context-wiki-sdk-tests-")
for _name in ("WIKI_DATA_DIR", "WIKI_ARCHIVE_DIR", "WIKI_STATE_DIR"):
    os.environ[_name] = _ISOLATED
# The SDK honours proxy variables; loopback must never be sent to a proxy.
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingest"))
import ai_research as a  # noqa: E402


class _Stub(BaseHTTPRequestHandler):
    """Answers every POST with the next canned reply and records the request."""
    replies: list[str] = []
    seen: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        out = json.dumps({
            "id": "chatcmpl-stub", "object": "chat.completion", "created": 0, "model": "stub-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": type(self).replies.pop(0)}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


def _serve(replies: list[str]):
    _Stub.replies, _Stub.seen = list(replies), []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    creds = {"base_url": f"http://127.0.0.1:{server.server_address[1]}/v1", "api_key": "sk-test"}
    return server, creds


def test_merge_index_round_trips_through_the_sdk():
    """merge_index sends model, max_tokens and both messages, and reads the reply text back."""
    wiki = "## Retrieval & RAG\n\n- Use hybrid search. [2026-01-01-ab-01]\n"
    server, creds = _serve([wiki])
    try:
        out = a.merge_index(wiki, [], creds)
    finally:
        server.shutdown()
    assert out == wiki.strip()
    req = _Stub.seen[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["auth"] == "Bearer sk-test"
    assert req["body"]["model"] == a.MODEL and req["body"]["max_tokens"] > 0
    assert [m["role"] for m in req["body"]["messages"]] == ["system", "user"]


def test_prune_to_cap_reads_the_sdk_reply():
    """prune_to_cap parses the stub's JSON array and deletes those claims (a parse failure keeps all)."""
    filler = " ".join(["word"] * 60)
    lines = [f"- Claim {n} {filler} [2026-01-01-ab-{n:02d}]" for n in range(40)]
    wiki = "## S\n\n" + "\n".join(lines) + "\n"
    assert len(wiki.split()) > a.WORD_CAP
    saved = a.WATCHLIST_PATH
    a.WATCHLIST_PATH = Path(_ISOLATED) / "no-watchlist.txt"
    server, creds = _serve([json.dumps(list(range(20)))] * a.MAX_PRUNE_PASSES)
    try:
        out = a.prune_to_cap(wiki, creds)
    finally:
        server.shutdown()
        a.WATCHLIST_PATH = saved
    assert _Stub.seen, "prune made no request"
    assert len(out.split()) < len(wiki.split())


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"ok    {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
