SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  duration_ms INTEGER NOT NULL,
  fps REAL,
  width INTEGER,
  height INTEGER,
  has_audio INTEGER NOT NULL DEFAULT 1,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS segments (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  kind TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  FOREIGN KEY(video_id) REFERENCES assets(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_segments_video_time
  ON segments(video_id, start_ms, end_ms);

CREATE TABLE IF NOT EXISTS evidence (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  segment_id TEXT,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  modality TEXT NOT NULL,
  content TEXT NOT NULL,
  confidence REAL NOT NULL,
  source TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  FOREIGN KEY(video_id) REFERENCES assets(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_evidence_video_time
  ON evidence(video_id, start_ms, end_ms);
CREATE INDEX IF NOT EXISTS idx_evidence_modality_time
  ON evidence(modality, start_ms, end_ms);

CREATE TABLE IF NOT EXISTS evidence_embeddings (
  id TEXT PRIMARY KEY,
  evidence_id TEXT NOT NULL,
  modality TEXT NOT NULL,
  model TEXT NOT NULL,
  vector_json TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  metadata_json TEXT NOT NULL,
  FOREIGN KEY(evidence_id) REFERENCES evidence(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_evidence_embeddings_namespace
  ON evidence_embeddings(modality, model);

CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
  video_id TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  status TEXT NOT NULL,
  stats_json TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(video_id, fingerprint)
);
"""
