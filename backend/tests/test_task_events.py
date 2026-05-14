from app.models import Paper
from app.services.task_events import get_task_status_channel, publish_task_status_event


def test_publish_task_status_event_uses_organization_channel(db_session, organization) -> None:
    paper = Paper(organization_id=organization.id, title="Org Paper", status="queued")
    db_session.add(paper)
    db_session.commit()
    db_session.refresh(paper)

    published: list[tuple[str, str]] = []

    class FakePublisher:
        def publish(self, channel: str, payload: str) -> None:
            published.append((channel, payload))

    import app.services.task_events as task_events

    original_get_publisher = task_events._get_publisher
    task_events._get_publisher = lambda: FakePublisher()
    try:
        publish_task_status_event(db_session, paper_id=paper.id)
    finally:
        task_events._get_publisher = original_get_publisher

    assert published
    assert published[0][0] == get_task_status_channel(organization.id)
