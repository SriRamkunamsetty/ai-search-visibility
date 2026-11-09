import contextlib
import csv
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from aivisibility.ai_visibility import (answer_details, estimate_calls, find_dates, get_ner_backend, names_in, release_verdict, run_visibility, set_ner_backend, visibility_report)
from aivisibility.catalog import import_facts_csv
from aivisibility.cli import main
from aivisibility.db import connect, get_or_create_title
from aivisibility.llm import AnthropicLLM, GeminiLLM, OllamaLLM, OpenAILLM, VLLMLLM, get_llm
from aivisibility.sample_data import seed_all
from aivisibility.web import App, make_handler


def tmp_csv(header, rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows); f.close()
    return f.name


class AIVisibility(unittest.TestCase):
    def test_date_parsing_and_verdicts(self):
        self.assertEqual(find_dates("March 5, 2027 and 7 April 2027 and 2027-06-01")[0], {"2027-03-05", "2027-04-07", "2027-06-01"})
        self.assertEqual(release_verdict("It opens March 5, 2027.", "2027-03-05"), "correct")
        self.assertEqual(release_verdict("Sometime in March 2027.", "2027-03-05"), "partial")
        self.assertEqual(release_verdict("It opens March 6, 2027.", "2027-03-05"), "wrong")
        self.assertEqual(release_verdict("I don't know.", "2027-03-05"), "unknown")
        self.assertEqual(find_dates("Feb 30, 2027")[0], set())                                  # impossible date is ignored

    def test_names_heuristic(self):
        self.assertEqual(names_in("Stars Ada Lovelace in March. The Film also.", "The Film"), {"Ada Lovelace"})

    def test_enhanced_entity_recognition_and_pluggable_ner(self):
        # False positives from studios, editorial phrases, roles, days of week should be rejected
        text = ("Warner Bros and Universal Pictures announced that Executive Producer Johnathan Smith "
                "and Cillian Murphy will star. However on Monday in Los Angeles, Rotten Tomatoes reported acclaim.")
        found = names_in(text, "The Film")
        self.assertIn("Cillian Murphy", found)
        self.assertNotIn("Warner Bros", found)
        self.assertNotIn("Universal Pictures", found)
        self.assertNotIn("Los Angeles", found)
        self.assertNotIn("Rotten Tomatoes", found)
        self.assertNotIn("Monday", found)

        # Test custom pluggable NER backend
        mock_backend = lambda t: ["Extracted Person One", "Non Person Org"]
        try:
            set_ner_backend(mock_backend)
            res = names_in("Any text", "The Film")
            self.assertIn("Extracted Person One", res)
        finally:
            set_ner_backend(None)

    def _db(self):
        c = connect()
        a = get_or_create_title(c, "Alpha Rising", "action"); b = get_or_create_title(c, "Beta Storm", "action")
        for t, d, cast in ((a, "2027-03-05", ["Ann Lee", "Bo Chan"]), (b, "2027-04-01", ["Cy Dunn"])):
            c.execute("INSERT INTO title_facts VALUES (?,?,?)", (t, "release_date", d))
            for n in cast:
                c.execute("INSERT INTO title_facts VALUES (?,?,?)", (t, "cast", n))
        c.execute("UPDATE titles SET release_date='2027-03-05' WHERE id=?", (a,)); c.commit()
        return c, a, b

    def test_report_numbers_with_scripted_model(self):
        c, a, b = self._db()
        calls = []
        def model(prompt, system=None, max_tokens=0):
            calls.append(prompt)
            if "release date of 'Alpha" in prompt: return "It opens March 5, 2027." if len(calls) % 2 else "It opens May 1, 2027."
            if "Who stars in 'Alpha" in prompt: return "Stars Ann Lee and Zed Quartz."
            if "anticipated action" in prompt: return "1. Beta Storm\n2. Alpha Rising" if len(calls) % 2 else "1. Alpha Rising"
            return "Looks great."
        n = estimate_calls(c, 4)
        self.assertEqual(run_visibility(c, model, "m", n_samples=4), n)
        self.assertEqual(run_visibility(c, model, "m", n_samples=4), 0)                      # cached: no repeat spend
        r = visibility_report(c, "m")["titles"][a]
        self.assertEqual(r["release"]["n"], 4); self.assertEqual(r["release"]["correct"] + r["release"]["wrong"], 4)
        self.assertEqual(r["cast"]["recall"], 0.5); self.assertEqual(r["cast"]["possible_fabricated_names_per_answer"], 1.0)
        cat = r["category"]; self.assertEqual(cat["mention_rate"], 1.0)
        self.assertEqual(cat["share_of_voice"], 4 / (4 + 2))                                  # Alpha 4 mentions, Beta 2
        self.assertIsNotNone(cat["mention_ci95"])

    def test_demo_pipeline_runs_offline(self):
        c = connect(); seed_all(c)
        v = visibility_report(c, "simulated-demo")["titles"]
        self.assertEqual(len(v), 12); self.assertTrue(all("release" in t and "cast" in t for t in v.values()))


def local_server(fn):
    """Real HTTP server on localhost. fn(method, path, headers, body) -> (status, json-able). Returns (base, log, stop)."""
    log = []
    class H(BaseHTTPRequestHandler):
        def _go(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
            log.append((self.command, self.path, dict(self.headers), body))
            status, out = fn(self.command, self.path, self.headers, body)
            data = (out if isinstance(out, str) else json.dumps(out)).encode()
            self.send_response(status); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        do_GET = do_POST = _go
        def log_message(self, *a): pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", log, lambda: (srv.shutdown(), srv.server_close())


class RealHttpTransports(unittest.TestCase):

    def test_anthropic_and_gemini_default_transport(self):
        calls = {"n": 0}
        def fn(method, path, headers, body):
            data = json.loads(body)
            if path == "/v1/messages":
                calls["n"] += 1
                if calls["n"] == 1:
                    return 529, "overloaded"
                assert headers["x-api-key"] == "sk-test" and headers["anthropic-version"] and data["messages"][0]["content"] == "hello" and data["system"] == "be brief"
                return 200, {"content": [{"type": "text", "text": "hi there"}]}
            assert path.endswith(":generateContent") and headers["x-goog-api-key"] == "g-test" and "g-test" not in path
            return 200, {"candidates": [{"content": {"parts": [{"text": "gem"}, {"text": "ini"}]}}]}
        base, log, stop = local_server(fn)
        try:
            a = AnthropicLLM("sk-test", sleep=lambda s: None); a.URL = base + "/v1/messages"
            self.assertEqual(a("hello", system="be brief"), "hi there"); self.assertEqual(calls["n"], 2)
            g = GeminiLLM("g-test", model="m1"); g.BASE = base + "/v1beta/models/"
            self.assertEqual(g("hello"), "gemini")
            self.assertEqual(json.loads(log[-1][3])["contents"][0]["parts"][0]["text"], "hello")
        finally:
            stop()

    def test_openai_ollama_vllm_default_transport(self):
        def fn(method, path, headers, body):
            data = json.loads(body)
            assert path.endswith("/chat/completions")
            assert headers["content-type"] == "application/json"
            if "Bearer sk-openai" in headers.get("authorization", ""):
                return 200, {"choices": [{"message": {"content": "hello from openai"}}]}
            if "Bearer ollama" in headers.get("authorization", ""):
                return 200, {"choices": [{"message": {"content": "hello from ollama"}}]}
            if "Bearer EMPTY" in headers.get("authorization", ""):
                return 200, {"choices": [{"message": {"content": "hello from vllm"}}]}
            return 401, "unauthorized"

        base, log, stop = local_server(fn)
        try:
            o = OpenAILLM("sk-openai", model="gpt-4o-mini", base_url=base + "/v1")
            self.assertEqual(o("hello", system="be brief"), "hello from openai")

            ol = OllamaLLM(base_url=base)  # tests auto /v1 normalization
            self.assertEqual(ol("hello"), "hello from ollama")

            vl = VLLMLLM(base_url=base + "/v1")
            self.assertEqual(vl("hello"), "hello from vllm")

            # Provider resolution smoke tests
            self.assertIsInstance(get_llm("openai", api_key="test"), OpenAILLM)
            self.assertIsInstance(get_llm("ollama", base_url=base), OllamaLLM)
            self.assertIsInstance(get_llm("vllm", base_url=base), VLLMLLM)
        finally:
            stop()


if __name__ == "__main__":
    unittest.main()
