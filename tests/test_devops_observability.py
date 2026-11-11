import contextlib
import io
import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from aivisibility.cli import main
from aivisibility.db import connect, get_or_create_title, now_iso
from aivisibility.observability import METRICS
from aivisibility.sample_data import seed_all
from aivisibility.web import App, make_handler
from aivisibility.worker import AuditWorker


class TestDevOpsObservability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = connect(check_same_thread=False)
        seed_all(cls.conn)
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App(cls.conn)))
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def _get(self, path):
        with urllib.request.urlopen(self.base + path) as r:
            return r.status, dict(r.headers), r.read()

    def test_prometheus_metrics_endpoint(self):
        status, headers, body = self._get("/metrics")
        self.assertEqual(status, 200)
        self.assertIn("text/plain", headers["Content-Type"])
        text = body.decode()
        self.assertIn("# HELP aivisibility_requests_total", text)
        self.assertIn("# TYPE aivisibility_requests_total counter", text)
        self.assertIn("aivisibility_up 1.0", text)

    def test_kubernetes_health_and_readiness_probes(self):
        s_live, _, b_live = self._get("/livez")
        self.assertEqual(s_live, 200)
        self.assertTrue(json.loads(b_live)["ok"])

        s_ready, _, b_ready = self._get("/readyz")
        self.assertEqual(s_ready, 200)
        data = json.loads(b_ready)
        self.assertEqual(data["status"], "ready")

    def test_api_eval_drift_endpoint(self):
        s, _, b = self._get("/api/eval/drift?a=1&b=1")
        self.assertEqual(s, 200)
        d = json.loads(b)
        self.assertIn("psi_mention_rate", d)
        self.assertIn("drift_level", d)
        self.assertEqual(d["drift_level"], "stable")

    def test_audit_worker_batch_execution(self):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name

        try:
            c = connect(db_path, check_same_thread=False)
            tid = get_or_create_title(c, "Worker Test Film", "action")
            c.execute(
                "INSERT INTO audit_tasks (title_id, model, prompt_key, prompt, sample_no, status, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (tid, "mock-model", "release", "When is it released?", 1, "pending", now_iso(), now_iso()),
            )
            c.commit()
            c.close()

            def fake_llm(p, **kw):
                return "Scheduled for release June 10, 2027."

            worker = AuditWorker(db_path, concurrency=2, llm_factory=lambda: fake_llm)
            processed = worker.process_next_batch()
            self.assertEqual(processed, 1)

            c_check = connect(db_path)
            row = c_check.execute("SELECT status FROM audit_tasks WHERE title_id=?", (tid,)).fetchone()
            self.assertEqual(row["status"], "completed")
            c_check.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_cli_eval_commands(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["eval", "semantic", "--reference", "Detective in space", "--candidate", "Astronaut in deep space"])
        self.assertIn("overall_faithfulness", buf.getvalue())

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["eval", "drift", "--demo", "--run1", "1", "--run2", "1"])
        self.assertIn("Distributional PSI", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
