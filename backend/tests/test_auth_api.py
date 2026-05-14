def test_auth_me_returns_organization_role_and_permissions(client, user, organization) -> None:
    login_response = client.post("/api/auth/login", json={"username": "admin", "password": "123456"})
    assert login_response.status_code == 200

    body = login_response.json()
    assert body["username"] == "admin"
    assert body["role"] == "org_owner"
    assert body["organization"]["id"] == organization.id
    assert body["organization"]["name"] == organization.name
    assert "papers:write" in body["permissions"]
    assert "org_settings:write" in body["permissions"]

    me_response = client.get("/api/auth/me")
    assert me_response.status_code == 200
    me = me_response.json()
    assert me["id"] == user.id
    assert me["organization"]["slug"] == organization.slug
    assert sorted(me["permissions"]) == sorted(body["permissions"])
