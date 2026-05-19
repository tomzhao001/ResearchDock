-- ResearchDock bootstrap schema for fresh database initialization
-- Includes the current schema and initial admin seed data.
-- Requires PostgreSQL with pgvector (e.g. pgvector/pgvector image)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE organizations (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    slug VARCHAR(128) NOT NULL UNIQUE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    organization_id BIGINT NOT NULL REFERENCES organizations (id),
    username VARCHAR(64) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(32) NOT NULL DEFAULT 'org_member',
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

CREATE TABLE organization_settings (
    id BIGSERIAL PRIMARY KEY,
    organization_id BIGINT NOT NULL UNIQUE REFERENCES organizations (id) ON DELETE CASCADE,
    auto_extraction_questions_json JSONB,
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
    organization_id BIGINT NOT NULL REFERENCES organizations (id),
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

CREATE TABLE paper_document_pages (
    id BIGSERIAL PRIMARY KEY,
    paper_id BIGINT NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
    asset_id BIGINT NOT NULL REFERENCES paper_assets (id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    text TEXT,
    width INTEGER,
    height INTEGER,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE paper_document_blocks (
    id BIGSERIAL PRIMARY KEY,
    paper_id BIGINT NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
    page_id BIGINT REFERENCES paper_document_pages (id) ON DELETE CASCADE,
    block_index INTEGER NOT NULL,
    block_type VARCHAR(64) NOT NULL DEFAULT 'paragraph',
    docling_label VARCHAR(128),
    heading_level INTEGER,
    section_path TEXT,
    text TEXT NOT NULL,
    bbox_json JSONB,
    provenance_json JSONB,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE paper_document_tables (
    id BIGSERIAL PRIMARY KEY,
    paper_id BIGINT NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
    page_from INTEGER,
    page_to INTEGER,
    table_index INTEGER NOT NULL,
    caption TEXT,
    markdown TEXT,
    data_json JSONB,
    bbox_json JSONB,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE paper_document_pictures (
    id BIGSERIAL PRIMARY KEY,
    paper_id BIGINT NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
    page_number INTEGER,
    picture_index INTEGER NOT NULL,
    caption TEXT,
    description TEXT,
    description_model VARCHAR(255),
    description_prompt_version VARCHAR(64),
    bbox_json JSONB,
    image_asset_path TEXT,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_paper_document_pages_paper_page ON paper_document_pages (paper_id, page_number);
CREATE INDEX idx_paper_document_blocks_paper_block ON paper_document_blocks (paper_id, block_index);
CREATE INDEX idx_paper_document_blocks_paper_type ON paper_document_blocks (paper_id, block_type);
CREATE INDEX idx_paper_document_tables_paper_table ON paper_document_tables (paper_id, table_index);
CREATE INDEX idx_paper_document_pictures_paper_picture ON paper_document_pictures (paper_id, picture_index);

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
    embedding VECTOR(1024),
    search_vector TSVECTOR,
    token_count INTEGER,
    page_from INTEGER,
    page_to INTEGER,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_paper_chunks_paper_id ON paper_chunks (paper_id);
CREATE INDEX idx_paper_chunks_paper_chunk ON paper_chunks (paper_id, chunk_index);
CREATE INDEX idx_paper_chunks_search_vector ON paper_chunks USING GIN (search_vector);
CREATE INDEX idx_paper_chunks_embedding_cosine ON paper_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

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
    celery_task_id VARCHAR(255),
    status VARCHAR(32),
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    cancel_requested_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO organizations (name, slug, is_active)
VALUES ('Default Organization', 'default', TRUE)
ON CONFLICT (slug) DO NOTHING;

-- Initial admin user (password: 123456) — bcrypt hash generated at project setup
INSERT INTO users (organization_id, username, password_hash, role, is_active)
VALUES (
    (SELECT id FROM organizations WHERE slug = 'default'),
    'admin',
    '$2b$12$FWCFMmz/kramxYvmhhW8e.Icx3D/TOEeoknZAffydgnEai/G6OEny',
    'org_owner',
    TRUE
)
ON CONFLICT (username) DO NOTHING;

CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL PRIMARY KEY
);

INSERT INTO alembic_version (version_num)
VALUES ('20260519_03')
ON CONFLICT (version_num) DO NOTHING;
