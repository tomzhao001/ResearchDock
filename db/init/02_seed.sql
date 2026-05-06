-- Initial admin user (password: 123456) — bcrypt hash generated at project setup
INSERT INTO users (username, password_hash, is_active)
VALUES (
    'admin',
    '$2b$12$FWCFMmz/kramxYvmhhW8e.Icx3D/TOEeoknZAffydgnEai/G6OEny',
    TRUE
)
ON CONFLICT (username) DO NOTHING;
