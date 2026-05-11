-- ResearchDock schema upgrade for Phase 1 hybrid retrieval
-- Apply this on top of an existing database initialized by 01_schema.sql / 02_seed.sql / 03_milestone4.sql.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE paper_chunks
    ADD COLUMN search_vector TSVECTOR;

ALTER TABLE paper_chunks
    ADD COLUMN embedding_next VECTOR(1024);

UPDATE paper_chunks
SET embedding_next = CASE
    WHEN embedding IS NULL THEN NULL
    WHEN jsonb_typeof(embedding) = 'array' AND jsonb_array_length(embedding) = 1024
        THEN replace(embedding::text, ' ', '')::vector(1024)
    ELSE NULL
END;

ALTER TABLE paper_chunks
    DROP COLUMN embedding;

ALTER TABLE paper_chunks
    RENAME COLUMN embedding_next TO embedding;

UPDATE paper_chunks
SET search_vector = to_tsvector('simple', coalesce(content, ''))
WHERE search_vector IS NULL;

CREATE INDEX IF NOT EXISTS idx_paper_chunks_search_vector
    ON paper_chunks USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS idx_paper_chunks_embedding_cosine
    ON paper_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

COMMIT;
