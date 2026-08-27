"""PostgreSQL schema for traceable evidence and pgvector embeddings."""

POSTGRES_SCHEMA_VERSION = "1.0.0"

POSTGRES_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS pl_assets (
  id TEXT PRIMARY KEY,
  source_uri TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  duration_ms BIGINT NOT NULL CHECK (duration_ms >= 0),
  fps DOUBLE PRECISION,
  width INTEGER,
  height INTEGER,
  has_audio BOOLEAN NOT NULL DEFAULT TRUE,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pl_segments (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES pl_assets(id) ON DELETE CASCADE,
  start_ms BIGINT NOT NULL CHECK (start_ms >= 0),
  end_ms BIGINT NOT NULL CHECK (end_ms >= start_ms),
  kind TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS pl_segments_video_time
  ON pl_segments(video_id, start_ms, end_ms);

CREATE TABLE IF NOT EXISTS pl_evidence (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES pl_assets(id) ON DELETE CASCADE,
  segment_id TEXT REFERENCES pl_segments(id) ON DELETE SET NULL,
  start_ms BIGINT NOT NULL CHECK (start_ms >= 0),
  end_ms BIGINT NOT NULL CHECK (end_ms >= start_ms),
  modality TEXT NOT NULL,
  content TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  source TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  evidence_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS pl_evidence_video_time
  ON pl_evidence(video_id, start_ms, end_ms);
CREATE INDEX IF NOT EXISTS pl_evidence_modality_time
  ON pl_evidence(modality, start_ms, end_ms);
CREATE INDEX IF NOT EXISTS pl_evidence_fts
  ON pl_evidence USING GIN (to_tsvector('simple', content));

-- The embedding row intentionally repeats the complete provenance needed for
-- retrieval/audit. evidence_id remains a foreign key to the canonical row.
-- The INSERT path populates these columns with INSERT ... SELECT from
-- pl_evidence JOIN pl_assets, so an orphan or mismatched embedding is rejected.
CREATE TABLE IF NOT EXISTS pl_embeddings (
  id TEXT PRIMARY KEY,
  evidence_id TEXT NOT NULL REFERENCES pl_evidence(id) ON DELETE CASCADE,
  video_id TEXT NOT NULL REFERENCES pl_assets(id) ON DELETE CASCADE,
  segment_id TEXT REFERENCES pl_segments(id) ON DELETE SET NULL,
  start_ms BIGINT NOT NULL CHECK (start_ms >= 0),
  end_ms BIGINT NOT NULL CHECK (end_ms >= start_ms),
  modality TEXT NOT NULL,
  source_uri TEXT NOT NULL,
  evidence_text TEXT NOT NULL,
  evidence_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  evidence_source TEXT NOT NULL,
  model_version TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  evidence_hash TEXT NOT NULL,
  source_sha256 TEXT NOT NULL,
  embedding_hash TEXT NOT NULL,
  embedding vector NOT NULL,
  dimensions INTEGER NOT NULL CHECK (dimensions > 0),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (evidence_id, modality, model_version)
);
CREATE INDEX IF NOT EXISTS pl_embeddings_lookup
  ON pl_embeddings(modality, model_version, dimensions);
CREATE INDEX IF NOT EXISTS pl_embeddings_evidence
  ON pl_embeddings(evidence_id);

-- Protect the duplicated audit fields even if a future writer bypasses the
-- Python store. Updates to canonical evidence are reflected by the same
-- transaction and must still produce a matching embedding row.
CREATE OR REPLACE FUNCTION pl_validate_embedding_provenance()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE canonical RECORD;
BEGIN
  SELECT e.video_id,
         e.segment_id,
         e.start_ms,
         e.end_ms,
         e.modality,
         e.content,
         e.metadata,
         e.source,
         e.confidence,
         e.evidence_hash,
         COALESCE(NULLIF(e.metadata->>'source_uri', ''),
                  NULLIF(e.metadata->>'media_path', ''),
                  NULLIF(e.metadata->>'frame_path', ''), a.source_uri) AS source_uri,
         a.sha256 AS source_sha256
    INTO canonical
    FROM pl_evidence e
    JOIN pl_assets a ON a.id = e.video_id
   WHERE e.id = NEW.evidence_id;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'embedding % has no canonical evidence %', NEW.id, NEW.evidence_id;
  END IF;

  IF NEW.video_id IS DISTINCT FROM canonical.video_id
     OR NEW.segment_id IS DISTINCT FROM canonical.segment_id
     OR NEW.start_ms IS DISTINCT FROM canonical.start_ms
     OR NEW.end_ms IS DISTINCT FROM canonical.end_ms
     OR NEW.modality IS DISTINCT FROM canonical.modality
     OR NEW.source_uri IS DISTINCT FROM canonical.source_uri
     OR NEW.evidence_text IS DISTINCT FROM canonical.content
     OR NEW.evidence_metadata IS DISTINCT FROM canonical.metadata
     OR NEW.evidence_source IS DISTINCT FROM canonical.source
     OR NEW.confidence IS DISTINCT FROM canonical.confidence
     OR NEW.evidence_hash IS DISTINCT FROM canonical.evidence_hash
     OR NEW.source_sha256 IS DISTINCT FROM canonical.source_sha256 THEN
    RAISE EXCEPTION 'embedding % provenance does not match evidence %', NEW.id, NEW.evidence_id;
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS pl_embeddings_provenance_guard ON pl_embeddings;
CREATE TRIGGER pl_embeddings_provenance_guard
  BEFORE INSERT OR UPDATE ON pl_embeddings
  FOR EACH ROW EXECUTE FUNCTION pl_validate_embedding_provenance();

CREATE OR REPLACE FUNCTION pl_sync_embedding_provenance()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
  UPDATE pl_embeddings p
     SET video_id = NEW.video_id,
         segment_id = NEW.segment_id,
         start_ms = NEW.start_ms,
         end_ms = NEW.end_ms,
         modality = NEW.modality,
         source_uri = COALESCE(NULLIF(NEW.metadata->>'source_uri', ''),
                               NULLIF(NEW.metadata->>'media_path', ''),
                               NULLIF(NEW.metadata->>'frame_path', ''),
                               (SELECT source_uri FROM pl_assets WHERE id = NEW.video_id)),
         evidence_text = NEW.content,
         evidence_metadata = NEW.metadata,
         evidence_source = NEW.source,
         confidence = NEW.confidence,
         evidence_hash = NEW.evidence_hash,
         source_sha256 = (SELECT sha256 FROM pl_assets WHERE id = NEW.video_id)
   WHERE p.evidence_id = NEW.id;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS pl_evidence_embedding_sync ON pl_evidence;
CREATE TRIGGER pl_evidence_embedding_sync
  AFTER UPDATE ON pl_evidence
  FOR EACH ROW EXECUTE FUNCTION pl_sync_embedding_provenance();

CREATE OR REPLACE FUNCTION pl_sync_asset_embedding_provenance()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
  UPDATE pl_embeddings p
     SET source_uri = COALESCE(NULLIF(e.metadata->>'source_uri', ''),
                               NULLIF(e.metadata->>'media_path', ''),
                               NULLIF(e.metadata->>'frame_path', ''), NEW.source_uri),
         source_sha256 = NEW.sha256
    FROM pl_evidence e
   WHERE p.evidence_id = e.id AND e.video_id = NEW.id;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS pl_assets_embedding_sync ON pl_assets;
CREATE TRIGGER pl_assets_embedding_sync
  AFTER UPDATE ON pl_assets
  FOR EACH ROW EXECUTE FUNCTION pl_sync_asset_embedding_provenance();

CREATE TABLE IF NOT EXISTS pl_metadata (
  key TEXT PRIMARY KEY,
  value JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS pl_ingestion_runs (
  video_id TEXT NOT NULL REFERENCES pl_assets(id) ON DELETE CASCADE,
  fingerprint TEXT NOT NULL,
  status TEXT NOT NULL,
  stats JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(video_id, fingerprint)
);
"""
