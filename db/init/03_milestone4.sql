-- ResearchDock schema upgrade for Milestone 4
-- Apply this on top of an existing database initialized by 01_schema.sql / 02_seed.sql.

BEGIN;

ALTER TABLE IF EXISTS paper_chunks
    ALTER COLUMN embedding TYPE JSONB
    USING CASE
        WHEN embedding IS NULL THEN NULL
        ELSE (embedding::text)::jsonb
    END;

CREATE INDEX IF NOT EXISTS idx_paper_chunks_paper_chunk
    ON paper_chunks (paper_id, chunk_index);

CREATE TABLE IF NOT EXISTS chat_topics (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL DEFAULT '新话题',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_topics_user_updated_at
    ON chat_topics (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    topic_id BIGINT NOT NULL REFERENCES chat_topics (id) ON DELETE CASCADE,
    role VARCHAR(32) NOT NULL,
    content TEXT NOT NULL,
    model VARCHAR(255),
    answer_mode VARCHAR(32),
    used_knowledge_base BOOLEAN NOT NULL DEFAULT FALSE,
    citations_json JSONB,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_topic_created_at
    ON chat_messages (topic_id, created_at ASC);

COMMIT;
