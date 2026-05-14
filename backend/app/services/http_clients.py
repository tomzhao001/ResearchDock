from __future__ import annotations

from threading import Lock

import httpx

_DEFAULT_LIMITS = httpx.Limits(
    max_connections=20,
    max_keepalive_connections=10,
    keepalive_expiry=30.0,
)

_client_lock = Lock()
_shared_clients: dict[tuple[str, bool], httpx.Client] = {}


def get_shared_http_client(*, name: str, verify_ssl: bool) -> httpx.Client:
    key = (name, bool(verify_ssl))
    with _client_lock:
        client = _shared_clients.get(key)
        if client is None or client.is_closed:
            client = httpx.Client(
                http2=True,
                verify=verify_ssl,
                limits=_DEFAULT_LIMITS,
                headers={"Accept": "application/json"},
            )
            _shared_clients[key] = client
        return client


def close_shared_http_clients() -> None:
    with _client_lock:
        clients = list(_shared_clients.values())
        _shared_clients.clear()
    for client in clients:
        client.close()
