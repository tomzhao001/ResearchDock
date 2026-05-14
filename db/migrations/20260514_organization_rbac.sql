-- ============================================================================
-- Migration: 组织 + RBAC（多租户粗粒度隔离）
-- - 新增 organizations、organization_settings
-- - users 增加 organization_id、role 并回填
-- - papers 增加 organization_id 并回填
-- - 种子数据：默认组织 + admin（密码与 db/init/01_schema.sql 一致：123456）
--
-- 环境：PostgreSQL（建议已安装 pgvector 扩展；与本项目 docker 镜像一致）
-- 特性：尽量幂等，可对「旧版无组织字段」的库执行，也可在已对齐 01_schema 的库上重复执行
-- 用法：psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f db/migrations/20260514_organization_rbac.sql
-- ============================================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS organizations (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    slug VARCHAR(128) NOT NULL UNIQUE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS organization_settings (
    id BIGSERIAL PRIMARY KEY,
    organization_id BIGINT NOT NULL UNIQUE REFERENCES organizations (id) ON DELETE CASCADE,
    auto_extraction_questions_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO organizations (name, slug, is_active)
VALUES ('Default Organization', 'default', TRUE)
ON CONFLICT (slug) DO NOTHING;

-- ---------------------------------------------------------------------------
-- users: organization_id
-- ---------------------------------------------------------------------------
ALTER TABLE users ADD COLUMN IF NOT EXISTS organization_id BIGINT;

UPDATE users u
SET organization_id = (SELECT id FROM organizations o WHERE o.slug = 'default' LIMIT 1)
WHERE u.organization_id IS NULL;

ALTER TABLE users ALTER COLUMN organization_id SET NOT NULL;

DO $users_fk$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON c.conrelid = t.oid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = 'public'
          AND t.relname = 'users'
          AND c.conname = 'users_organization_id_fkey'
    ) THEN
        ALTER TABLE users
            ADD CONSTRAINT users_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations (id);
    END IF;
END;
$users_fk$;

-- ---------------------------------------------------------------------------
-- users: role（与 backend/app/permissions.py 中 org_owner / org_admin / org_member 一致）
-- ---------------------------------------------------------------------------
ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(32) NOT NULL DEFAULT 'org_member';

-- ---------------------------------------------------------------------------
-- papers: organization_id
-- ---------------------------------------------------------------------------
ALTER TABLE papers ADD COLUMN IF NOT EXISTS organization_id BIGINT;

UPDATE papers p
SET organization_id = (SELECT id FROM organizations o WHERE o.slug = 'default' LIMIT 1)
WHERE p.organization_id IS NULL;

ALTER TABLE papers ALTER COLUMN organization_id SET NOT NULL;

DO $papers_fk$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON c.conrelid = t.oid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = 'public'
          AND t.relname = 'papers'
          AND c.conname = 'papers_organization_id_fkey'
    ) THEN
        ALTER TABLE papers
            ADD CONSTRAINT papers_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations (id);
    END IF;
END;
$papers_fk$;

-- ---------------------------------------------------------------------------
-- 默认管理员（bcrypt 与 01_schema.sql 相同；首次登录后请立即修改密码）
-- ---------------------------------------------------------------------------
INSERT INTO users (organization_id, username, password_hash, role, is_active)
VALUES (
    (SELECT id FROM organizations WHERE slug = 'default' LIMIT 1),
    'admin',
    '$2b$12$FWCFMmz/kramxYvmhhW8e.Icx3D/TOEeoknZAffydgnEai/G6OEny',
    'org_owner',
    TRUE
)
ON CONFLICT (username) DO NOTHING;

UPDATE users u
SET
    organization_id = o.id,
    role = 'org_owner'
FROM organizations o
WHERE o.slug = 'default'
  AND u.username = 'admin';

CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL PRIMARY KEY
);

INSERT INTO alembic_version (version_num)
VALUES ('20260514_01')
ON CONFLICT (version_num) DO NOTHING;

COMMIT;
