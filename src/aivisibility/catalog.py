"""Importer for ground-truth title facts (what the answers are checked against)."""
import csv
from datetime import datetime


FACT_ATTRS = ("release_date", "cast", "director", "alias", "studio")


def _read(path, required):
    fh = open(path, newline="", encoding="utf-8-sig")
    rd = csv.DictReader(fh)
    miss = [c for c in required if c not in (rd.fieldnames or [])]
    if miss:
        fh.close()
        raise ValueError(f"CSV is missing required columns: {miss}")
    return fh, rd


def import_facts_csv(conn, path):
    """Columns: title, attribute, value. attribute in release_date (YYYY-MM-DD), cast, director, alias, studio.
    A new release_date replaces the old one; other attributes accumulate (several cast members)."""
    ok, rejected = 0, []
    fh, rd = _read(path, ("title", "attribute", "value"))
    with fh:
        for n, r in enumerate(rd, start=2):
            try:
                t = conn.execute("SELECT id FROM titles WHERE name=?", (r["title"].strip(),)).fetchone()
                if not t:
                    raise ValueError(f"unknown title '{r['title'].strip()}'")
                attr, val = r["attribute"].strip().lower(), r["value"].strip()
                if attr not in FACT_ATTRS:
                    raise ValueError(f"attribute must be one of {FACT_ATTRS}")
                if not val:
                    raise ValueError("empty value")
                if attr == "release_date":
                    datetime.strptime(val, "%Y-%m-%d")
                    conn.execute("DELETE FROM title_facts WHERE title_id=? AND attribute='release_date'", (t["id"],))
                    conn.execute("UPDATE titles SET release_date=? WHERE id=?", (val, t["id"]))
                conn.execute("INSERT INTO title_facts (title_id, attribute, value) VALUES (?,?,?) ON CONFLICT DO NOTHING", (t["id"], attr, val))
                ok += 1
            except (ValueError, KeyError) as exc:
                rejected.append({"line": n, "reason": str(exc)})
    conn.commit()
    return {"accepted": ok, "rejected": rejected}
