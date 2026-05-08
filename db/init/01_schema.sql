-- ResearchDock MVP schema (Milestone 4)
-- Requires PostgreSQL with pgvector (e.g. pgvector/pgvector image)

-- CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE app_settings (
    id BIGSERIAL PRIMARY KEY,
    openai_base_url TEXT,
    openai_api_key_encrypted TEXT,
    chat_model VARCHAR(255),
    embedding_model VARCHAR(255),
    default_summary_language VARCHAR(32),
    default_chunk_size INTEGER,
    default_chunk_overlap INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE sources (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    url TEXT NOT NULL,
    source_type VARCHAR(64),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    schedule_cron VARCHAR(128),
    max_items_per_run INTEGER,
    last_run_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE papers (
    id BIGSERIAL PRIMARY KEY,
    title TEXT,
    authors TEXT,
    abstract_raw TEXT,
    source_url TEXT,
    pdf_url TEXT,
    doi VARCHAR(255),
    published_at TIMESTAMPTZ,
    content_hash VARCHAR(128),
    ingest_type VARCHAR(32),
    status VARCHAR(32),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

CREATE TABLE paper_assets (
    id BIGSERIAL PRIMARY KEY,
    paper_id BIGINT NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
    asset_type VARCHAR(64),
    storage_path TEXT,
    mime_type VARCHAR(128),
    raw_text TEXT,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE paper_summaries (
    id BIGSERIAL PRIMARY KEY,
    paper_id BIGINT NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
    summary_language VARCHAR(32),
    abstract_zh TEXT,
    summary_points JSONB,
    research_problem TEXT,
    method TEXT,
    findings TEXT,
    limitations TEXT,
    model_name VARCHAR(255),
    prompt_version VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE paper_chunks (
    id BIGSERIAL PRIMARY KEY,
    paper_id BIGINT NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding JSONB,
    token_count INTEGER,
    page_from INTEGER,
    page_to INTEGER,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_paper_chunks_paper_id ON paper_chunks (paper_id);
CREATE INDEX idx_paper_chunks_paper_chunk ON paper_chunks (paper_id, chunk_index);

CREATE TABLE chat_topics (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL DEFAULT '新话题',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_chat_topics_user_updated_at ON chat_topics (user_id, updated_at DESC);

CREATE TABLE chat_messages (
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

CREATE INDEX idx_chat_messages_topic_created_at ON chat_messages (topic_id, created_at ASC);

CREATE TABLE jobs (
    id BIGSERIAL PRIMARY KEY,
    job_type VARCHAR(64),
    source_id BIGINT REFERENCES sources (id) ON DELETE SET NULL,
    paper_id BIGINT REFERENCES papers (id) ON DELETE SET NULL,
    status VARCHAR(32),
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
