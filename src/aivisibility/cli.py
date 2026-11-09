"""AI-Search Visibility command line. Run `aivisibility <command> -h` for options."""
import argparse
import json
import os

from . import __version__
from .ai_visibility import (audit_diff, estimate_calls, list_audit_runs, run_audit, run_visibility, visibility_report)
from .catalog import import_facts_csv
from .db import connect


def open_db(a):
    if getattr(a, "demo", False):
        from .sample_data import seed_all
        conn = connect(check_same_thread=False)
        seed_all(conn)
        return conn
    return connect(a.db, check_same_thread=False)


def _pc(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def _table(rep):
    head = f"{'title':<22}{'release ok':>11}{'cast recall':>12}{'fake names':>11}{'mention rate (95% CI)':>26}{'rank':>6}{'SOV':>6}"
    out = [f"model: {rep['model']}", head, "-" * len(head)]
    for r in rep["titles"].values():
        rel, cat, cast = r.get("release"), r.get("category"), r.get("cast")
        ci = f"{_pc(cat['mention_rate'])} ({_pc(cat['mention_ci95'][0])}-{_pc(cat['mention_ci95'][1])})" if cat else "n/a"
        out.append(f"{r['name']:<22}{(_pc(rel['correct'] / rel['n']) if rel else 'n/a'):>11}{(_pc(cast['recall']) if cast else 'n/a'):>12}"
                   f"{(format(cast['possible_fabricated_names_per_answer'], '.2f') if cast else 'n/a'):>11}{ci:>26}"
                   f"{(format(cat['avg_position'], '.1f') if cat and cat['avg_position'] else 'n/a'):>6}{(_pc(cat['share_of_voice']) if cat else 'n/a'):>6}")
    return "\n".join(out)


PROVIDERS = ["anthropic", "gemini", "openai", "ollama", "vllm"]


def build_parser():
    p = argparse.ArgumentParser(prog="aivisibility", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=os.environ.get("DATABASE_URL", "aivisibility.db"), help="SQLite path or postgresql:// DSN")
    common.add_argument("--demo", action="store_true", help="use in-memory SYNTHETIC demo data (simulated model)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", parents=[common], help="visibility report from stored answers"); r.add_argument("--model"); r.add_argument("--format", choices=["table", "json"], default="table")
    run = sub.add_parser("run", parents=[common], help="ask a real model (spends API calls)"); run.add_argument("--provider", choices=PROVIDERS)
    run.add_argument("--samples", type=int, default=5); run.add_argument("--model"); run.add_argument("--yes", action="store_true", help="confirm API spend")
    es = sub.add_parser("estimate", parents=[common], help="API calls a run would make"); es.add_argument("--samples", type=int, default=5)
    sub.add_parser("import-facts", parents=[common], help="title,attribute,value (release_date|cast|director|alias|studio)").add_argument("path")
    w = sub.add_parser("serve", parents=[common], help="read-only dashboard"); w.add_argument("--host", default="127.0.0.1"); w.add_argument("--port", type=int, default=8000)
    k = sub.add_parser("check-llm", help="live smoke test of an LLM key"); k.add_argument("--provider", choices=PROVIDERS)

    # Audits & scheduled runs
    au = sub.add_parser("audit", parents=[common], help="manage and schedule automated visibility audits")
    au_sub = au.add_subparsers(dest="audit_cmd", required=True)
    au_sub.add_parser("list", parents=[common], help="list historical audit runs")
    au_run = au_sub.add_parser("run", parents=[common], help="execute a tracked audit run")
    au_run.add_argument("--provider", choices=PROVIDERS); au_run.add_argument("--samples", type=int, default=5)
    au_run.add_argument("--model"); au_run.add_argument("--yes", action="store_true", help="confirm API spend")
    au_diff = au_sub.add_parser("diff", parents=[common], help="compare two audit runs")
    au_diff.add_argument("--run1", type=int, required=True, help="first audit run id")
    au_diff.add_argument("--run2", type=int, required=True, help="second audit run id")
    au_sch = au_sub.add_parser("schedule", parents=[common], help="run recurring automated audits")
    au_sch.add_argument("--provider", choices=PROVIDERS); au_sch.add_argument("--samples", type=int, default=5)
    au_sch.add_argument("--model"); au_sch.add_argument("--interval", type=int, default=3600, help="interval in seconds between audits")
    au_sch.add_argument("--iterations", type=int, default=0, help="max iterations (0 for unlimited)")
    au_sch.add_argument("--yes", action="store_true", help="confirm API spend")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        _dispatch(a)
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}")


def _dispatch(a):
    if a.cmd == "report":
        conn = open_db(a)
        models = [r["model"] for r in conn.execute("SELECT DISTINCT model FROM ai_answers ORDER BY model")]
        model = a.model or (models[0] if models else None)
        if not model:
            raise SystemExit("no stored answers: run `aivisibility run` first (or use --demo)")
        rep = visibility_report(conn, model)
        print(json.dumps(rep, indent=2, default=str) if a.format == "json" else _table(rep))
    elif a.cmd == "estimate":
        print(f"{estimate_calls(open_db(a), a.samples)} API calls for {a.samples} samples per prompt")
    elif a.cmd == "run":
        from .llm import get_llm
        conn, llm = open_db(a), get_llm(a.provider)
        print(f"{estimate_calls(conn, a.samples)} API calls to {llm.name} ({llm.model}); cached answers are skipped.")
        if not a.yes:
            raise SystemExit("re-run with --yes to spend the API calls")
        print("new calls made:", run_visibility(conn, llm, a.model or llm.model, a.samples))
    elif a.cmd == "import-facts":
        r = import_facts_csv(open_db(a), a.path); print(f"accepted={r['accepted']} rejected={len(r['rejected'])}")
        for x in r["rejected"][:20]:
            print(f"  line {x['line']}: {x['reason']}")
    elif a.cmd == "serve":
        from .web import serve
        serve(open_db(a), a.host, a.port)
    elif a.cmd == "check-llm":
        from .llm import get_llm
        llm = get_llm(a.provider)
        print(f"{llm.name} ({llm.model}) replied: {llm('Reply with the single word: ok', max_tokens=10).strip()!r}")
    elif a.cmd == "audit":
        conn = open_db(a)
        if a.audit_cmd == "list":
            runs = list_audit_runs(conn)
            if not runs:
                print("No audit runs recorded.")
                return
            head = f"{'ID':<6}{'DATE (UTC)':<22}{'MODEL':<18}{'PROVIDER':<12}{'SAMPLES':<9}{'CALLS':<8}{'REL. OK':<10}{'MENTION':<10}"
            print(head)
            print("-" * len(head))
            for r in runs:
                ms = r.get("metrics_summary") or {}
                rel = _pc(ms.get("avg_release_correct"))
                mr = _pc(ms.get("avg_mention_rate"))
                dt = (r.get("started_at") or "")[:19]
                print(f"{r['id']:<6}{dt:<22}{r['model']:<18}{r['provider']:<12}{r['samples_per_prompt']:<9}{r['calls_made']:<8}{rel:<10}{mr:<10}")
        elif a.audit_cmd == "run":
            from .llm import get_llm
            conn = open_db(a)
            llm = get_llm(a.provider)
            print(f"{estimate_calls(conn, a.samples)} API calls to {llm.name} ({llm.model}); cached answers are skipped.")
            if not a.yes:
                raise SystemExit("re-run with --yes to spend the API calls")
            res = run_audit(conn, llm, a.model or llm.model, a.samples, provider=llm.name)
            print(f"audit run #{res['audit_id']} completed: {res['calls_made']} new calls made.")
        elif a.audit_cmd == "diff":
            d = audit_diff(conn, a.run1, a.run2)
            ra, rb = d["run_a"], d["run_b"]
            print(f"Comparing Audit #{ra['id']} ({ra['model']} @ {ra['started_at'][:19]}) -> Audit #{rb['id']} ({rb['model']} @ {rb['started_at'][:19]}):")
            head = f"{'title':<24}{'release ok (A -> B)':<26}{'mention rate (A -> B)':<26}{'cast recall (A -> B)':<24}"
            print(head)
            print("-" * len(head))
            for t in d["titles"]:
                if t.get("status") == "compared":
                    r_a = _pc(t.get("release_rate_a"))
                    r_b = _pc(t.get("release_rate_b"))
                    r_d = f"({t['release_rate_delta']:+.0%})" if t.get("release_rate_delta") is not None else ""
                    rel_s = f"{r_a} -> {r_b} {r_d}"

                    m_a = _pc(t.get("mention_rate_a"))
                    m_b = _pc(t.get("mention_rate_b"))
                    m_d = f"({t['mention_rate_delta']:+.0%})" if t.get("mention_rate_delta") is not None else ""
                    mr_s = f"{m_a} -> {m_b} {m_d}"

                    c_a = _pc(t.get("cast_recall_a"))
                    c_b = _pc(t.get("cast_recall_b"))
                    c_d = f"({t['cast_recall_delta']:+.0%})" if t.get("cast_recall_delta") is not None else ""
                    cast_s = f"{c_a} -> {c_b} {c_d}"

                    print(f"{t['name']:<24}{rel_s:<26}{mr_s:<26}{cast_s:<24}")
                elif t.get("status") == "added_in_b":
                    print(f"{t['name']:<24}[ADDED in run {rb['id']}]")
                elif t.get("status") == "removed_in_b":
                    print(f"{t['name']:<24}[REMOVED in run {rb['id']}]")
        elif a.audit_cmd == "schedule":
            from .llm import get_llm
            import time
            from datetime import datetime, timezone
            llm = get_llm(a.provider)
            model = a.model or llm.model
            print(f"{estimate_calls(conn, a.samples)} API calls per audit to {llm.name} ({model})")
            if not a.yes:
                raise SystemExit("re-run with --yes to confirm scheduling API calls")
            print(f"Automated audit schedule: model={model}, interval={a.interval}s, iterations={'unlimited' if not a.iterations else a.iterations}")
            count = 0
            try:
                while True:
                    count += 1
                    t_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
                    print(f"[{t_str}] Starting scheduled audit iteration #{count}...")
                    res = run_audit(conn, llm, model, a.samples, provider=llm.name)
                    print(f"[{t_str}] Completed audit run #{res['audit_id']}: {res['calls_made']} calls made.")
                    if a.iterations and count >= a.iterations:
                        print(f"Completed {a.iterations} scheduled iterations. Exiting.")
                        break
                    time.sleep(a.interval)
            except KeyboardInterrupt:
                print("\nScheduler interrupted by user. Exiting.")


if __name__ == "__main__":
    main()
