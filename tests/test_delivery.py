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
from http.server import ThreadingHTTPServer

from aivisibility.ai_visibility import (answer_details, release_verdict, visibility_report)
from aivisibility.catalog import import_facts_csv
from aivisibility.cli import main
from aivisibility.db import connect, get_or_create_title
from aivisibility.sample_data import seed_all
from aivisibility.web import App, make_handler


def tmp_csv(header, rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows); f.close()
    return f.name


class FactsImport(unittest.TestCase):
    def test_facts_validation_and_release_date_replacement(self):
        c = connect(); tid = get_or_create_title(c, "Alpha")
        rows = [dict(title="Alpha", attribute="release_date", value="2027-03-05"), dict(title="Alpha", attribute="release_date", value="2027-04-01"),
                dict(title="Alpha", attribute="cast", value="Ann Lee"), dict(title="Alpha", attribute="cast", value="Bo Chan"),
                dict(title="Alpha", attribute="budget", value="1"), dict(title="Alpha", attribute="release_date", value="03/05/2027")]
        p = tmp_csv(["title", "attribute", "value"], rows); res = import_facts_csv(c, p); os.unlink(p)
        self.assertEqual((res["accepted"], len(res["rejected"])), (4, 2))
        self.assertEqual([r["value"] for r in c.execute("SELECT value FROM title_facts WHERE attribute='release_date'")], ["2027-04-01"])
        self.assertEqual(c.execute("SELECT release_date FROM titles WHERE id=?", (tid,)).fetchone()["release_date"], "2027-04-01")


class Inspector(unittest.TestCase):
    def test_answer_details_verdicts_match_ground_truth(self):
        c = connect(); seed_all(c)
        d = answer_details(c, 1, "simulated-demo")
        rel = [a for a in d["answers"] if a["prompt_key"] == "release"]
        self.assertEqual(len(rel), 8)
        truth = d["truth"]["release_date"][0]
        for a in rel:
            self.assertEqual(a["verdict"], release_verdict(a["answer"], truth))
        self.assertTrue(any(a["prompt_key"].startswith("category:") and "mentioned" in a for a in d["answers"]))
        self.assertIsNone(answer_details(c, 9999, "simulated-demo"))

    def test_counts_agree_with_the_report(self):
        c = connect(); seed_all(c)
        rep = visibility_report(c, "simulated-demo")["titles"][3]
        d = answer_details(c, 3, "simulated-demo")
        self.assertEqual(rep["release"]["correct"], sum(a.get("verdict") == "correct" for a in d["answers"] if a["prompt_key"] == "release"))


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        conn = connect(check_same_thread=False); seed_all(conn)
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App(conn)))
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.srv.server_close()

    def call(self, p, data=None):
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base + p, data=data)) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def test_api_page_and_hardening(self):
        s, h, b = self.call("/")
        self.assertEqual(s, 200); self.assertNotIn(b"innerHTML", b); self.assertIn("default-src 'none'", h["Content-Security-Policy"])
        self.assertTrue(json.loads(self.call("/api/meta")[2])["simulated"])
        ai = json.loads(self.call("/api/ai")[2]); self.assertEqual(len(ai["titles"]), 12)
        self.assertEqual(json.loads(self.call("/api/ai?model=nope")[2])["titles"], {})
        self.assertEqual(self.call("/api/answers/1")[0], 200)
        for p in ("/api/answers/9999", "/api/answers/abc", "/etc/passwd", "/api/answers/1?model=nope"):
            self.assertEqual(self.call(p)[0], 404)
        self.assertEqual(self.call("/api/ai", data=b"{}")[0], 405)

    def test_audit_api_endpoints(self):
        s, h, b = self.call("/api/audits")
        self.assertEqual(s, 200)
        audits = json.loads(b)
        self.assertTrue(len(audits) >= 1)
        run_id = audits[0]["id"]

        s, h, b = self.call(f"/api/audits/{run_id}")
        self.assertEqual(s, 200)
        single = json.loads(b)
        self.assertEqual(single["id"], run_id)
        self.assertIn("summary", single)

        s, h, b = self.call(f"/api/audits/diff?a={run_id}&b={run_id}")
        self.assertEqual(s, 200)
        diff = json.loads(b)
        self.assertIn("titles", diff)

        self.assertEqual(self.call("/api/audits/99999")[0], 404)
        self.assertEqual(self.call("/api/audits/diff?a=invalid&b=1")[0], 400)


class Cli(unittest.TestCase):
    def test_report_estimate_and_run_requires_confirmation(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["report", "--demo"])
        self.assertIn("simulated-demo", buf.getvalue()); self.assertIn("mention rate", buf.getvalue())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["estimate", "--demo", "--samples", "5"])
        self.assertIn("API calls", buf.getvalue())
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
                main(["run", "--demo", "--provider", "anthropic"])
            self.assertIn("--yes", str(cm.exception))                                      # never spends without explicit confirmation
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_audit_cli_commands(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["audit", "list", "--demo"])
        self.assertIn("simulated-demo", buf.getvalue())
        self.assertIn("REL. OK", buf.getvalue())

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["audit", "diff", "--demo", "--run1", "1", "--run2", "1"])
        self.assertIn("release ok", buf.getvalue())

        # Test schedule spend guard
        os.environ["OPENAI_API_KEY"] = "sk-test"
        try:
            with self.assertRaises(SystemExit) as cm:
                main(["audit", "schedule", "--demo", "--provider", "openai"])
            self.assertIn("--yes", str(cm.exception))
        finally:
            os.environ.pop("OPENAI_API_KEY", None)


class Browser(unittest.TestCase):
    def test_table_whiskers_and_inspector_in_a_real_browser(self):
        try:
            from playwright.sync_api import sync_playwright
            pw = sync_playwright().start(); browser = pw.chromium.launch()
        except Exception:
            self.skipTest("Playwright/Chromium not available")
        conn = connect(check_same_thread=False); seed_all(conn)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App(conn))); threading.Thread(target=srv.serve_forever, daemon=True).start()
        errors = []
        try:
            page = browser.new_page(); page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{srv.server_address[1]}/"); page.wait_for_selector("tbody tr")
            self.assertEqual(page.locator("tbody tr").count(), 12); self.assertTrue(page.locator("tbody svg").count() >= 20)
            page.locator("tbody tr").nth(4).click(); page.wait_for_selector("text=Ground truth you entered")
            txt = page.inner_text("main"); self.assertIn("ANSWER INSPECTOR", txt.upper())
            self.assertTrue(page.locator(".pill").count() > 5)
            for bad in ("[object", "undefined", "NaN"):
                self.assertNotIn(bad, txt)
        finally:
            browser.close(); pw.stop(); srv.shutdown(); srv.server_close()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
