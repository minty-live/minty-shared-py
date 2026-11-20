"""pytest plugin (v0.8): no test reaches minty-accounts-api's session touch.

``pytest_plugins = ["minty_shared.testing.sessions"]`` in a service's root ``conftest.py``: every
test starts with an empty :data:`minty_shared.tokens.SESSIONS` and a fake touch in which every
sign-in lives - unless the test ends it with ``hub_sessions.end(sid)``. ``hub_sessions.calls``
lists the sids that were asked about. (v0.9) ``hub_sessions.unreachable()`` makes every touch fail
as an unreachable accounts-api would (``SessionCheckError``); ``reachable()`` undoes it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from minty_shared import tokens


class HubSessions:
    """The fake behind ``minty_shared.tokens.touch_over_http`` in a test."""

    def __init__(self) -> None:
        self.ended: set[str] = set()
        self.calls: list[str] = []
        self.down = False

    def end(self, sid: str) -> None:
        self.ended.add(sid)

    def unreachable(self) -> None:
        self.down = True

    def reachable(self) -> None:
        self.down = False

    def touch(self, accounts_api_url: str, secret: str, sid: str, **kwargs: Any) -> bool:
        self.calls.append(sid)
        if self.down:
            raise tokens.SessionCheckError("accounts-api unreachable (hub_sessions.unreachable())")
        return sid not in self.ended


@pytest.fixture(autouse=True)
def hub_sessions(monkeypatch: pytest.MonkeyPatch) -> Iterator[HubSessions]:
    fake = HubSessions()
    monkeypatch.setattr(tokens, "touch_over_http", fake.touch)
    tokens.SESSIONS.forget()
    yield fake
    tokens.SESSIONS.forget()
