-- One template, two dialects ({PK} is substituted in db.py). Timestamps are ISO-8601 UTC text.
CREATE TABLE IF NOT EXISTS titles (
  id {PK},
  name TEXT NOT NULL UNIQUE,
  genre TEXT,
  studio TEXT,
  release_date TEXT
);
CREATE TABLE IF NOT EXISTS title_facts (
  title_id BIGINT NOT NULL REFERENCES titles(id),
  attribute TEXT NOT NULL,
  value TEXT NOT NULL,
  PRIMARY KEY (title_id, attribute, value)
);
CREATE TABLE IF NOT EXISTS ai_answers (
  id {PK},
  title_id BIGINT NOT NULL REFERENCES titles(id),
  model TEXT NOT NULL,
  prompt_key TEXT NOT NULL,
  prompt TEXT NOT NULL,
  sample_no INTEGER NOT NULL,
  asked_at TEXT NOT NULL,
  answer TEXT NOT NULL,
  audit_run_id BIGINT,
  UNIQUE (title_id, model, prompt_key, sample_no)
);
CREATE TABLE IF NOT EXISTS audit_runs (
  id {PK},
  model TEXT NOT NULL,
  provider TEXT,
  samples_per_prompt INTEGER NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  status TEXT NOT NULL DEFAULT 'running',
  calls_made INTEGER DEFAULT 0,
  summary_json TEXT
);
CREATE TABLE IF NOT EXISTS audit_tasks (
  id {PK},
  audit_run_id BIGINT,
  title_id BIGINT NOT NULL REFERENCES titles(id),
  model TEXT NOT NULL,
  prompt_key TEXT NOT NULL,
  prompt TEXT NOT NULL,
  sample_no INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  retries INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
