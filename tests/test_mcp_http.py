"""MCP server: tool handlers and the Streamable HTTP transport (no index, no LLM, no network)."""
import asyncio
import json
import os
import unittest
from unittest.mock import patch

try:
    import mcp  # noqa: F401
    from starlette.testclient import TestClient
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

TOKEN = "test-token-123"
ACCEPT = "application/json, text/event-stream"


class FakeSearchEngine:
    def __init__(self):
        self.calls = []

    def search(self, query, limit=10, mode=None, subreddit=None):
        self.calls.append((query, limit, mode.value))
        return [{"subreddit": "Python", "score": 3, "text": f"hit for {query}", "file_path": "Python/POST_a.md"}]


def rpc(method, params=None, id_=1):
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}}


def sse_json(resp):
    """Return the JSON-RPC message carried in the SSE body."""
    data = [line[len("data:"):].strip() for line in resp.text.splitlines() if line.startswith("data:")]
    return json.loads(data[-1])


@unittest.skipUnless(HAS_MCP, "requires mcp extra")
class TestToolHandlers(unittest.TestCase):
    def setUp(self):
        import rsi.mcp_server as srv

        self.srv = srv
        self.engine = FakeSearchEngine()
        self.prev, srv._search_engine = srv._search_engine, self.engine

    def tearDown(self):
        self.srv._search_engine = self.prev

    def test_search_needs_no_llm(self):
        with patch("rsi.chat.providers.availability.find_available_provider") as find:
            out = asyncio.run(self.srv._handle_reddit_search({"query": "asyncio", "mode": "keyword"}))
        find.assert_not_called()
        self.assertIn("hit for asyncio", out[0].text)

    def test_limit_clamped(self):
        asyncio.run(self.srv._handle_reddit_search({"query": "q", "limit": 10_000}))
        asyncio.run(self.srv._handle_reddit_search({"query": "q", "limit": -5}))
        asyncio.run(self.srv._handle_reddit_search({"query": "q", "limit": "junk"}))
        self.assertEqual([c[1] for c in self.engine.calls], [50, 1, 10])

    def test_bad_mode_and_query_rejected(self):
        out = asyncio.run(self.srv._handle_reddit_search({"query": "q", "mode": "rm -rf"}))
        self.assertIn("mode", out[0].text)
        out = asyncio.run(self.srv._handle_reddit_search({"query": ""}))
        self.assertIn("query", out[0].text)
        self.assertEqual(self.engine.calls, [])

    def test_chat_without_llm_returns_text_error(self):
        with patch("rsi.chat.providers.availability.find_available_provider",
                   return_value=(None, None, "No LLM provider available.")):
            out = asyncio.run(self.srv._handle_reddit_chat({"query": "what did I save?"}))
        self.assertIn("unavailable", out[0].text)
        self.assertIn("reddit_search", out[0].text)

    def test_chat_builds_one_engine_per_call(self):
        from rsi.chat.engine import DirectEngine

        class FakeLLM:
            def generate(self, messages):
                return "answer"

        with patch("rsi.chat.providers.availability.find_available_provider",
                   return_value=("openai", "gpt-4o-mini", None)), \
                patch("rsi.chat.providers.base.create_provider", return_value=FakeLLM()) as create, \
                patch.object(DirectEngine, "__init__", wraps=DirectEngine.__init__, autospec=True) as init:
            for _ in range(2):
                out = asyncio.run(self.srv._handle_reddit_chat({"query": "q", "limit": 99}))
        self.assertEqual(init.call_count, 2)
        self.assertEqual(create.call_args.kwargs, {"provider": "openai", "model": "gpt-4o-mini"})
        self.assertIn("answer", out[0].text)
        self.assertEqual(self.engine.calls[-1][1], 50)


@unittest.skipUnless(HAS_MCP, "requires mcp extra")
class TestHttpTransport(unittest.TestCase):
    def setUp(self):
        import rsi.mcp_server as srv

        self.srv = srv
        self.engine = FakeSearchEngine()
        self.prev, srv._search_engine = srv._search_engine, self.engine

    def tearDown(self):
        self.srv._search_engine = self.prev

    def client(self, allowed_hosts=()):
        return TestClient(self.srv.create_http_app(token=TOKEN, allowed_hosts=list(allowed_hosts)))

    def post(self, c, body, token=TOKEN, **headers):
        h = {"Accept": ACCEPT, "Content-Type": "application/json", **headers}
        if token is not None:
            h["Authorization"] = f"Bearer {token}"
        return c.post("/mcp", json=body, headers=h, follow_redirects=False)

    def test_no_token_401(self):
        with self.client() as c:
            r = self.post(c, rpc("tools/list"), token=None)
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.headers["www-authenticate"], "Bearer")

    def test_wrong_token_401(self):
        with self.client() as c:
            self.assertEqual(self.post(c, rpc("tools/list"), token="nope").status_code, 401)

    def test_healthz_open(self):
        with self.client() as c:
            r = c.get("/healthz")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_exact_mcp_path_no_redirect_sse(self):
        init = rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                  "clientInfo": {"name": "t", "version": "1"}})
        with self.client() as c:
            r = self.post(c, init)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(sse_json(r)["result"]["serverInfo"]["name"], "reddit-stash-insights")

    def test_search_tool_over_http(self):
        with self.client() as c:
            r = self.post(c, rpc("tools/call", {"name": "reddit_search",
                                                "arguments": {"query": "asyncio", "mode": "keyword"}}))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("hit for asyncio", sse_json(r)["result"]["content"][0]["text"])

    def test_bad_host_rejected_good_host_accepted(self):
        with self.client(allowed_hosts=["rsi.example.com"]) as c:
            bad = self.post(c, rpc("tools/list"))
            good = self.post(c, rpc("tools/list"), Host="rsi.example.com")
        self.assertIn(bad.status_code, (400, 421))
        self.assertEqual(good.status_code, 200, good.text)


@unittest.skipUnless(HAS_MCP, "requires mcp extra")
class TestHttpStartup(unittest.TestCase):
    def test_refuses_without_token(self):
        from rsi.mcp_server import run_http

        with patch.dict(os.environ, {"RSI_MCP_TOKEN": ""}), self.assertRaises(RuntimeError):
            run_http("127.0.0.1", 0)

    def test_proxy_headers_off_by_default(self):
        from rsi.mcp_server import uvicorn_options

        with patch.dict(os.environ, {"RSI_FORWARDED_ALLOW_IPS": ""}):
            self.assertEqual(uvicorn_options("0.0.0.0", 8765),
                             {"host": "0.0.0.0", "port": 8765, "proxy_headers": False})

    def test_proxy_headers_from_allowlist(self):
        from rsi.mcp_server import uvicorn_options

        with patch.dict(os.environ, {"RSI_FORWARDED_ALLOW_IPS": "10.0.0.5, 10.0.0.6"}):
            opts = uvicorn_options("0.0.0.0", 8765)
        self.assertTrue(opts["proxy_headers"])
        self.assertEqual(opts["forwarded_allow_ips"], ["10.0.0.5", "10.0.0.6"])


if __name__ == "__main__":
    unittest.main()
