from sqlalchemy.orm import Session

from app.models import OrganizationSettings


def login(client, *, username: str, password: str) -> None:
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200


def test_org_question_set_defaults_to_empty(client, user) -> None:
    login(client, username="admin", password="123456")

    response = client.get("/api/org-settings/questions")

    assert response.status_code == 200
    body = response.json()
    assert body["questions"] == []


def test_org_question_set_can_be_updated(client, user, db_session: Session, organization) -> None:
    login(client, username="admin", password="123456")

    response = client.put(
        "/api/org-settings/questions",
        json={
            "questions": [
                {"id": "q1", "question": "这篇论文解决了什么问题？"},
                {"id": "q2", "question": "主要方法是什么？"},
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["organization_id"] == organization.id
    assert [item["id"] for item in body["questions"]] == ["q1", "q2"]

    settings = db_session.query(OrganizationSettings).filter_by(organization_id=organization.id).one()
    assert settings.auto_extraction_questions_json == [
        {"id": "q1", "question": "这篇论文解决了什么问题？"},
        {"id": "q2", "question": "主要方法是什么？"},
    ]


def test_org_question_set_is_scoped_by_organization(client, user, second_user, db_session: Session, organization, second_organization) -> None:
    login(client, username="admin", password="123456")
    write_response = client.put(
        "/api/org-settings/questions",
        json={"questions": [{"id": "org-a", "question": "组织 A 的问题"}]},
    )
    assert write_response.status_code == 200

    client.post("/api/auth/logout")
    login(client, username="other-admin", password="654321")
    read_response = client.get("/api/org-settings/questions")

    assert read_response.status_code == 200
    assert read_response.json()["organization_id"] == second_organization.id
    assert read_response.json()["questions"] == []

    settings = db_session.query(OrganizationSettings).filter_by(organization_id=organization.id).one()
    assert settings.auto_extraction_questions_json == [{"id": "org-a", "question": "组织 A 的问题"}]


def test_org_question_set_rejects_duplicate_ids(client, user) -> None:
    login(client, username="admin", password="123456")

    response = client.put(
        "/api/org-settings/questions",
        json={
            "questions": [
                {"id": "dup", "question": "问题一"},
                {"id": "dup", "question": "问题二"},
            ]
        },
    )

    assert response.status_code == 400
    assert "Duplicate question id" in response.json()["detail"]
