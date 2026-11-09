"""SYNTHETIC titles, ground truth and a SEEDED SIMULATOR standing in for an AI model (never a real LLM)."""
import random
from datetime import datetime, timedelta, timezone

from .db import get_or_create_title


TITLES = [("Iron Meridian", "action"), ("Steel Tide", "action"), ("Last Bastion", "action"), ("Zero Hour Protocol", "action"),
          ("The Quiet Orchard", "drama"), ("Salt and Ember", "drama"), ("Paper Lanterns", "drama"), ("A Winter Letter", "drama"),
          ("Pause Button", "comedy"), ("Roommate Wanted", "comedy"), ("Bad Influence", "comedy"), ("Plus One", "comedy")]


FIRST = "Maren Teodor Ilsa Corin Veda Joss Anwen Bram Sorrel Dax Lior Petra".split()


LAST = "Voss Quill Marrow Hale Thorne Ashby Reeve Calloway Pryor Finch Lund Okoro".split()


def seed_ai_facts(conn, seed_value=9):
    """SYNTHETIC ground truth (fictional release dates, cast, directors) for the AI-visibility demo."""
    rng = random.Random(seed_value)
    base = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for ti, (name, genre) in enumerate(TITLES):
        tid = get_or_create_title(conn, name, genre)
        rel = (base + timedelta(days=ti * 9)).strftime("%Y-%m-%d")
        conn.execute("UPDATE titles SET release_date=? WHERE id=?", (rel, tid))
        rows = [("release_date", rel), ("director", f"{rng.choice(FIRST)} {rng.choice(LAST)}")]
        rows += [("cast", f"{FIRST[(ti + k) % len(FIRST)]} {LAST[(ti * 3 + k * 5) % len(LAST)]}") for k in range(3)]
        for attr, val in rows:
            conn.execute("INSERT INTO title_facts (title_id, attribute, value) VALUES (?,?,?) ON CONFLICT DO NOTHING", (tid, attr, val))
    conn.commit()


class SimulatedModel:
    """NOT a real LLM: a seeded stand-in whose answers are right ~60% of the time, so the AI-visibility
    pipeline can be demonstrated offline. Real runs use launchintel.llm.get_llm()."""
    def __init__(self, conn, seed_value=1):
        self.conn, self.seed, self.n = conn, seed_value, 0

    def __call__(self, prompt, system=None, max_tokens=500):
        import re
        from .ai_visibility import facts_for
        self.n += 1
        rng = random.Random(f"{self.seed}-{self.n}-{prompt}")
        m = re.search(r"'(.+?)'", prompt)
        if m is None:  # category prompt
            g = re.search(r"anticipated (\w+) films", prompt).group(1)
            names = [r["name"] for r in self.conn.execute("SELECT name FROM titles WHERE genre=?", (g,))]
            shown = [n for n in names if rng.random() < 0.35 + 0.6 * (sum(map(ord, n)) % 10) / 10]
            rng.shuffle(shown)
            return "\n".join(f"{i}. {n} - a much-discussed upcoming release." for i, n in enumerate(shown, 1)) or "No clear list available."
        t = self.conn.execute("SELECT id, name FROM titles WHERE name=?", (m.group(1),)).fetchone()
        f = facts_for(self.conn, t["id"])
        if "release date" in prompt:
            r = rng.random()
            d = datetime.strptime(f["release_date"][0], "%Y-%m-%d")
            if r < 0.6:
                return f"It is scheduled for {d.strftime('%B')} {d.day}, {d.year}."
            if r < 0.8:
                d += timedelta(days=21)
                return f"I believe it opens {d.strftime('%B')} {d.day}, {d.year}."
            return "I don't have reliable information about that release date."
        if "Who stars" in prompt:
            names = [n for n in f["cast"] if rng.random() < 0.8]
            extra = [" and Dorian Ashby"] if rng.random() < 0.25 else [""]
            return "The film stars " + ", ".join(names) + extra[0] + "."
        if "worth watching" in prompt:
            return "Yes, it looks great and the performances are fantastic." if rng.random() < 0.65 else "It looks boring and predictable to me."
        return f"'{t['name']}' is a feature film currently in the marketing phase."


def seed_all(conn):
    from .ai_visibility import run_audit
    seed_ai_facts(conn)
    run_audit(conn, SimulatedModel(conn), "simulated-demo", n_samples=8, provider="simulator")
    conn.commit()
