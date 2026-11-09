"""Project 5: AI-search visibility. How do AI assistants describe your titles, and are they right?

Asks a model the same questions N times (answers vary), stores every answer, and measures: mention rate
and rank in category lists, share of voice vs other tracked titles, release-date correctness, cast recall,
possible fabricated names, sentiment. Proportions carry Wilson 95% intervals.

HONEST SCOPE: answers depend on model, date and phrasing; keep n_samples >= 5 for anything you report.
Name-extraction for 'unverified names' is a rough capitalisation heuristic and will have false positives."""
import json
import re
from datetime import datetime, timezone
from statistics import mean

from .db import now_iso
from .sentiment import lexicon_score
from .stats import wilson

PROMPTS = {
    "about": "In two sentences, what is the film '{title}' about?",
    "cast": "Who stars in '{title}'? List the main cast.",
    "release": "What is the release date of '{title}'?",
    "worth": "Is '{title}' worth watching? Give a short opinion.",
}
CATEGORY = "What are the most anticipated {genre} films of {year}? List up to 8 with one line each."
MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
MN = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
D_MDY = re.compile(rf"\b({MN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b")
D_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MN})\.?,?\s+(\d{{4}})\b")
D_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
D_MY = re.compile(rf"\b({MN})\.?,?\s+(\d{{4}})\b")
NAME = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}\b")
NAME_STOP = {
    # Pronouns / determiners / prepositions / conjunctions
    "The", "In", "It", "As", "He", "She", "This", "That", "With", "By", "For", "From", "To", "At", "On", "Of", "And",
    # Editorial & verbs
    "Starring", "Directed", "Main", "Cast", "Release", "Date", "According", "Based", "Stars", "Features",
    "Led", "Includes", "Also", "Film", "Movie", "Series", "Title", "Story", "Plot", "Genre", "Studio",
    # Common capitalized sentence transitions & adverbs
    "However", "Although", "Meanwhile", "Furthermore", "Therefore", "Moreover", "Additionally",
    "Interestingly", "Unsurprisingly", "Fortunately", "Unfortunately", "Specifically", "Notably",
    "Importantly", "Naturally", "Clearly", "Overall", "Instead", "Perhaps", "Obviously", "Honestly",
    "Actually", "First", "Second", "Third", "Next", "Finally", "Recently", "Currently", "Later",
    # Industry roles & accolades
    "Director", "Producer", "Executive", "Screenwriter", "Writer", "Composer", "Cinematographer",
    "Academy", "Award", "Golden", "Globe", "Box", "Office", "Rotten", "Tomatoes",
    # Days & Seasons
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "Spring", "Summer", "Autumn", "Fall", "Winter",
}

NON_PERSON_ENTITIES = {
    "Warner Bros", "Universal Pictures", "Walt Disney", "Sony Pictures", "Paramount Pictures",
    "20th Century", "Lionsgate Films", "Amazon Studios", "Apple TV", "HBO Max", "Marvel Studios",
    "Lucasfilm Ltd", "Focus Features", "Searchlight Pictures", "Columbia Pictures", "New Line Cinema",
    "New York", "Los Angeles", "United States", "United Kingdom", "Great Britain", "San Francisco",
    "North America", "South America", "Western Europe", "Rotten Tomatoes", "Box Office",
    "Executive Producer", "Academy Award", "Golden Globe", "Best Actor", "Best Actress",
    "Feature Film", "First Look", "Sneak Peek", "Official Trailer", "Critical Acclaim",
}

_NER_BACKEND = None


def set_ner_backend(backend):
    """Set custom NER callable: backend(text) -> list[str] of extracted person names."""
    global _NER_BACKEND
    _NER_BACKEND = backend


def get_ner_backend():
    """Returns active NER backend (or attempts to auto-load spaCy if installed)."""
    global _NER_BACKEND
    if _NER_BACKEND is not None:
        return _NER_BACKEND
    try:
        import os
        import spacy
        model_name = os.environ.get("NER_MODEL", "en_core_web_sm")
        try:
            nlp = spacy.load(model_name)
            return lambda text: [ent.text for ent in nlp(text).ents if ent.label_ == "PERSON"]
        except Exception:
            return None
    except ImportError:
        return None


def _m(name):
    return MONTHS[name[:3].lower()]


def find_dates(text):
    full, my = set(), set()
    def take(rx, f):
        nonlocal text
        for m in rx.finditer(text):
            try:
                full.add(f(m))
            except ValueError:
                pass
        text = rx.sub(" ", text)
    take(D_MDY, lambda m: datetime(int(m[3]), _m(m[1]), int(m[2])).strftime("%Y-%m-%d"))
    take(D_DMY, lambda m: datetime(int(m[3]), _m(m[2]), int(m[1])).strftime("%Y-%m-%d"))
    take(D_ISO, lambda m: datetime(int(m[1]), int(m[2]), int(m[3])).strftime("%Y-%m-%d"))
    for m in D_MY.finditer(text):
        my.add(f"{int(m[2]):04d}-{_m(m[1]):02d}")
    return full, my


def release_verdict(answer, truth_iso):
    full, my = find_dates(answer)
    if truth_iso in full:
        return "correct"
    if not full and truth_iso[:7] in my:
        return "partial"
    return "wrong" if (full or my) else "unknown"


def facts_for(conn, title_id):
    d = {}
    for r in conn.execute("SELECT attribute, value FROM title_facts WHERE title_id=?", (title_id,)):
        d.setdefault(r["attribute"], []).append(r["value"])
    return d


MONTH_WORDS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}


def names_in(text, title, ner_backend=None):
    backend = ner_backend or get_ner_backend()
    if callable(backend):
        try:
            candidates = backend(text)
            out = set()
            for raw in candidates:
                cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", raw).strip()
                words = cleaned.split()
                while words and words[0] in NAME_STOP:
                    words = words[1:]
                name = " ".join(words)
                if (len(words) < 2 or name.lower() in title.lower() or
                    any(x.lower() in MONTH_WORDS for x in words) or
                    name in NON_PERSON_ENTITIES):
                    continue
                out.add(name)
            return out
        except Exception:
            pass
    out = set()
    for m in NAME.finditer(text):
        w = m.group(0).split()
        while w and w[0] in NAME_STOP:  # sentence-initial filler glued onto a name ("Stars Ada Lovelace")
            w = w[1:]
        name = " ".join(w)
        if (len(w) < 2 or name.lower() in title.lower() or
            any(x.lower() in MONTH_WORDS for x in w) or
            name in NON_PERSON_ENTITIES or
            all(word in NAME_STOP for word in w)):
            continue
        out.add(name)
    return out


def _mention(name, aliases):
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in [name, *aliases]) + r")\b", re.I)


def ask(conn, llm, model, title_id, key, prompt, n_samples, audit_run_id=None):
    made = 0
    for s in range(1, n_samples + 1):
        if conn.execute("SELECT 1 FROM ai_answers WHERE title_id=? AND model=? AND prompt_key=? AND sample_no=?", (title_id, model, key, s)).fetchone():
            continue
        ans = llm(prompt, max_tokens=500)
        conn.execute("INSERT INTO ai_answers (title_id, model, prompt_key, prompt, sample_no, asked_at, answer, audit_run_id) VALUES (?,?,?,?,?,?,?,?) "
                     "ON CONFLICT (title_id, model, prompt_key, sample_no) DO NOTHING", (title_id, model, key, prompt, s, now_iso(), ans, audit_run_id))
        made += 1
    conn.commit()
    return made


def _genres(conn):
    g = {}
    for r in conn.execute("SELECT id, genre, release_date FROM titles WHERE genre IS NOT NULL ORDER BY id"):
        g.setdefault(r["genre"], []).append(r)
    return g


def estimate_calls(conn, n_samples):
    t = conn.execute("SELECT COUNT(*) c FROM titles").fetchone()["c"]
    return (t * len(PROMPTS) + len(_genres(conn))) * n_samples


def run_visibility(conn, llm, model, n_samples=5, audit_run_id=None):
    """Cached per (title, model, prompt, sample): re-running only fills gaps. Returns new calls made."""
    made = 0
    for t in conn.execute("SELECT id, name FROM titles ORDER BY id").fetchall():
        for key, tpl in PROMPTS.items():
            made += ask(conn, llm, model, t["id"], key, tpl.format(title=t["name"]), n_samples, audit_run_id=audit_run_id)
    for genre, ts in _genres(conn).items():
        year = (ts[0]["release_date"] or str(datetime.now(timezone.utc).year + 1))[:4]
        made += ask(conn, llm, model, ts[0]["id"], f"category:{genre}", CATEGORY.format(genre=genre, year=year), n_samples, audit_run_id=audit_run_id)
    return made


def _answers(conn, title_id, model, key):
    return [r["answer"] for r in conn.execute("SELECT answer FROM ai_answers WHERE title_id=? AND model=? AND prompt_key=? ORDER BY sample_no", (title_id, model, key))]


def visibility_report(conn, model):
    titles = conn.execute("SELECT id, name, genre FROM titles ORDER BY id").fetchall()
    genres = _genres(conn)
    per = {}
    for t in titles:
        f = facts_for(conn, t["id"])
        r = {"name": t["name"], "genre": t["genre"]}
        rel = _answers(conn, t["id"], model, "release")
        if rel and f.get("release_date"):
            c = {}
            for a in rel:
                v = release_verdict(a, f["release_date"][0]); c[v] = c.get(v, 0) + 1
            r["release"] = {"n": len(rel), **{k: c.get(k, 0) for k in ("correct", "partial", "wrong", "unknown")},
                            "correct_ci95": list(wilson(c.get("correct", 0), len(rel)))}
        cast = _answers(conn, t["id"], model, "cast")
        if cast and f.get("cast"):
            truth = f["cast"]
            r["cast"] = {"n": len(cast), "recall": mean(sum(n.lower() in a.lower() for n in truth) / len(truth) for a in cast),
                         "possible_fabricated_names_per_answer": mean(len({n for n in names_in(a, t["name"]) if not any(n.lower() in x.lower() or x.lower() in n.lower() for x in truth)}) for a in cast),
                         "caveat": "name extraction is a capitalisation heuristic"}
        worth = _answers(conn, t["id"], model, "worth")
        if worth:
            r["worth_net_sentiment"] = mean((lexicon_score(a) > 0) - (lexicon_score(a) < 0) for a in worth)
        per[t["id"]] = r
    for genre, ts in genres.items():
        answers = _answers(conn, ts[0]["id"], model, f"category:{genre}")
        if not answers:
            continue
        pats = {t["id"]: _mention(next(x["name"] for x in titles if x["id"] == t["id"]),
                                  facts_for(conn, t["id"]).get("alias", [])) for t in ts}
        hits = {i: 0 for i in pats}; ranks = {i: [] for i in pats}
        for a in answers:
            order = sorted(((m.start(), i) for i, p in pats.items() for m in [p.search(a)] if m), key=lambda x: x[0])
            for rank, (_, i) in enumerate(order, 1):
                hits[i] += 1; ranks[i].append(rank)
        total = sum(hits.values())
        for i in pats:
            lo, hi = wilson(hits[i], len(answers))
            per[i]["category"] = {"samples": len(answers), "mention_rate": hits[i] / len(answers), "mention_ci95": [lo, hi],
                                  "avg_position": mean(ranks[i]) if ranks[i] else None,
                                  "share_of_voice": hits[i] / total if total else None}
    return {"model": model, "titles": per}


def answer_details(conn, title_id, model):
    """Every stored answer for a title, side by side with ground truth and a per-answer verdict."""
    t = conn.execute("SELECT id, name, genre FROM titles WHERE id=?", (title_id,)).fetchone()
    if not t:
        return None
    f = facts_for(conn, title_id)
    anchor = conn.execute("SELECT MIN(id) m FROM titles WHERE genre=?", (t["genre"],)).fetchone()["m"] if t["genre"] else None
    pat = _mention(t["name"], f.get("alias", []))
    out = []
    rows = conn.execute("SELECT prompt_key, sample_no, answer FROM ai_answers WHERE title_id=? AND model=? ORDER BY prompt_key, sample_no", (title_id, model)).fetchall()
    if anchor is not None and t["genre"]:
        rows = list(rows) + list(conn.execute("SELECT prompt_key, sample_no, answer FROM ai_answers WHERE title_id=? AND model=? AND prompt_key=? ORDER BY sample_no",
                                              (anchor, model, f"category:{t['genre']}")).fetchall()) if anchor != title_id else rows
    for r in rows:
        d = {"prompt_key": r["prompt_key"], "sample": r["sample_no"], "answer": r["answer"]}
        if r["prompt_key"] == "release" and f.get("release_date"):
            d["verdict"] = release_verdict(r["answer"], f["release_date"][0])
        if r["prompt_key"] == "cast" and f.get("cast"):
            d["cast_found"] = [n for n in f["cast"] if n.lower() in r["answer"].lower()]
            d["unverified_names"] = sorted(n for n in names_in(r["answer"], t["name"]) if not any(n.lower() in x.lower() or x.lower() in n.lower() for x in f["cast"]))
        if r["prompt_key"].startswith("category:"):
            d["mentioned"] = bool(pat.search(r["answer"]))
        out.append(d)
    return {"title": t["name"], "genre": t["genre"], "truth": f, "answers": out}


def create_audit_run(conn, model, provider=None, n_samples=5):
    """Creates a new audit record in 'running' state and returns its ID."""
    from .db import insert_id
    return insert_id(conn,
                     "INSERT INTO audit_runs (model, provider, samples_per_prompt, started_at, status, calls_made) "
                     "VALUES (?,?,?,?, 'running', 0)",
                     (model, provider or "unknown", n_samples, now_iso()))


def complete_audit_run(conn, audit_id, calls_made, summary):
    """Finalizes an audit run with its computed visibility summary JSON."""
    summary_text = json.dumps(summary, default=str)
    conn.execute("UPDATE audit_runs SET completed_at=?, status='completed', calls_made=?, summary_json=? WHERE id=?",
                 (now_iso(), calls_made, summary_text, audit_id))
    conn.commit()


def list_audit_runs(conn, limit=50):
    """Returns past audit runs with high-level summary statistics."""
    rows = conn.execute("SELECT id, model, provider, samples_per_prompt, started_at, completed_at, status, calls_made, summary_json "
                        "FROM audit_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("summary_json"):
            try:
                s = json.loads(d["summary_json"])
                titles = list(s.get("titles", {}).values())
                rel_ok = [t["release"]["correct"] / t["release"]["n"] for t in titles if t.get("release") and t["release"].get("n")]
                cat_mr = [t["category"]["mention_rate"] for t in titles if t.get("category") and t["category"].get("mention_rate") is not None]
                d["metrics_summary"] = {
                    "titles_count": len(titles),
                    "avg_release_correct": mean(rel_ok) if rel_ok else None,
                    "avg_mention_rate": mean(cat_mr) if cat_mr else None,
                }
            except Exception:
                d["metrics_summary"] = None
        else:
            d["metrics_summary"] = None
        out.append(d)
    return out


def get_audit_run(conn, audit_id):
    """Returns details and full summary report for a single audit run."""
    row = conn.execute("SELECT id, model, provider, samples_per_prompt, started_at, completed_at, status, calls_made, summary_json "
                       "FROM audit_runs WHERE id=?", (audit_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    if d.get("summary_json"):
        try:
            d["summary"] = json.loads(d["summary_json"])
        except Exception:
            d["summary"] = None
    return d


def audit_diff(conn, run_id_a, run_id_b):
    """Computes title-by-title and aggregate differences between two audit runs."""
    ra = get_audit_run(conn, run_id_a)
    rb = get_audit_run(conn, run_id_b)
    if not ra or not rb:
        raise ValueError(f"Audit run not found (run_a={run_id_a}, run_b={run_id_b})")
    sa = ra.get("summary") or {}
    sb = rb.get("summary") or {}
    titles_a = {t.get("name", str(k)): t for k, t in (sa.get("titles") or {}).items()}
    titles_b = {t.get("name", str(k)): t for k, t in (sb.get("titles") or {}).items()}
    all_names = sorted(set(titles_a.keys()) | set(titles_b.keys()))
    diffs = []
    for name in all_names:
        ta, tb = titles_a.get(name), titles_b.get(name)
        if not ta:
            diffs.append({"name": name, "status": "added_in_b", "change": tb})
            continue
        if not tb:
            diffs.append({"name": name, "status": "removed_in_b", "change": ta})
            continue
        # Release accuracy delta
        rel_a = ta.get("release", {})
        rel_b = tb.get("release", {})
        rate_a = (rel_a.get("correct", 0) / rel_a["n"]) if rel_a.get("n") else None
        rate_b = (rel_b.get("correct", 0) / rel_b["n"]) if rel_b.get("n") else None
        rel_delta = (rate_b - rate_a) if (rate_a is not None and rate_b is not None) else None

        # Mention rate delta
        cat_a = ta.get("category", {})
        cat_b = tb.get("category", {})
        mr_a = cat_a.get("mention_rate")
        mr_b = cat_b.get("mention_rate")
        mr_delta = (mr_b - mr_a) if (mr_a is not None and mr_b is not None) else None

        # Cast recall delta
        cast_a = ta.get("cast", {}).get("recall")
        cast_b = tb.get("cast", {}).get("recall")
        cast_delta = (cast_b - cast_a) if (cast_a is not None and cast_b is not None) else None

        diffs.append({
            "name": name,
            "status": "compared",
            "release_rate_a": rate_a,
            "release_rate_b": rate_b,
            "release_rate_delta": rel_delta,
            "mention_rate_a": mr_a,
            "mention_rate_b": mr_b,
            "mention_rate_delta": mr_delta,
            "cast_recall_a": cast_a,
            "cast_recall_b": cast_b,
            "cast_recall_delta": cast_delta,
        })
    return {
        "run_a": {"id": ra["id"], "model": ra["model"], "started_at": ra["started_at"]},
        "run_b": {"id": rb["id"], "model": rb["model"], "started_at": rb["started_at"]},
        "titles": diffs
    }


def run_audit(conn, llm, model, n_samples=5, provider=None):
    """Executes a complete tracked audit run and stores the snapshot."""
    prov = provider or getattr(llm, "name", "unknown")
    run_id = create_audit_run(conn, model, provider=prov, n_samples=n_samples)
    try:
        calls = run_visibility(conn, llm, model, n_samples=n_samples, audit_run_id=run_id)
        rep = visibility_report(conn, model)
        complete_audit_run(conn, run_id, calls, rep)
        return {"audit_id": run_id, "calls_made": calls, "report": rep}
    except Exception as exc:
        conn.execute("UPDATE audit_runs SET status='failed', completed_at=? WHERE id=?", (now_iso(), run_id))
        conn.commit()
        raise exc
