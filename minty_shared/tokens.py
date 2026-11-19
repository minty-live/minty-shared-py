"""The two JWTs Minty's backends exchange, both HS256 on the shared ``SECRET_KEY`` (extra ``jwt``).

* A **bearer** is minted by Flask for a signed-in person and verified by every service.
  :func:`decode_bearer` allows the same clock skew everywhere (60 s): Flask mints, another host
  verifies, and PyJWT's default of zero refuses a token whose ``iat`` is a second ahead of this
  host's clock - which the wizard read as an expired session (measured 2026-09-14: a laptop ran
  6.5 s ahead of Render).
* A **service assertion** is how a Django API calls one of Flask's internal routes: 60 s long,
  scoped to one route (``scope``) and one company (``entity_id``), optionally naming the person.
  Flask's ``_authenticate_internal_caller`` refuses any other scope.
* **The sign-in behind a token** (v0.8, owner 2026-11-19: sign-out after 60 minutes without
  activity): a hand-off token's ``sid`` claim names the hub session it was minted from.
  :func:`session_alive` asks minty-accounts-api (``POST /api/internal/session/touch``, a service
  assertion scoped ``session-touch``) whether that session still lives, which also counts as
  activity and keeps it alive. At most once a minute per ``sid`` per process
  (:class:`SessionGate`); a session that has ended answers 410 and the caller refuses the token
  (401), so signing out of Minty, or 60 idle minutes, ends every app. A token with no ``sid``
  (minted before v0.8) is alive until it expires. accounts-api unreachable or answering anything
  but 200/410 is logged at ERROR and the token honoured (it is still a signed, unexpired
  credential), then asked about again a minute later.
* **Telling the two 401s apart** (v0.8, owner 2026-11-19: "Instead of an Unauthorized error, show
  a pop-up that the session has expired"): :func:`session_ended` - the token is ours (expiry
  ignored) and its sign-in has ended - lets an API add ``"reason": "session_expired"`` to its 401
  (:func:`mark_session_ended`, :func:`unauthorized_reason`). Without the reason, the app renews
  its token quietly; with it, it says the session has expired and sends the person to sign in.

The secret is passed in rather than read from a framework's settings, so this module has no
framework import.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from minty_shared import logctx

ALGORITHM = "HS256"
CLOCK_SKEW_LEEWAY_SECONDS = 60
SERVICE_ASSERTION_SECONDS = 60
#: The hub-session touch (v0.8): the route, its assertion's scope, how often one ``sid`` is asked
#: about per process, and how long a call may take.
SESSION_TOUCH_PATH = "/api/internal/session/touch"
SESSION_TOUCH_SCOPE = "session-touch"
SESSION_CHECK_SECONDS = 60
SESSION_TOUCH_TIMEOUT_SECONDS = 5
#: Past this many remembered ``sid``s, the ones older than a check are dropped.
_SESSION_CACHE_LIMIT = 4096

logger = logging.getLogger("minty_shared.tokens")


def decode_bearer(
    token: str, secret: str, *, leeway: int = CLOCK_SKEW_LEEWAY_SECONDS, **options: Any
) -> dict[str, Any]:
    """The verified claims of a bearer, or a ``jwt.PyJWTError`` (expired, bad signature,
    malformed). ``options`` are PyJWT's (e.g. ``options={"require": [...]}``)."""
    # PyJWT's own annotations are partial; the claims are a plain JSON object.
    claims: dict[str, Any] = jwt.decode(  # pyright: ignore[reportUnknownMemberType]
        token, secret, algorithms=[ALGORITHM], leeway=leeway, **options
    )
    return claims


def service_assertion(
    secret: str,
    scope: str,
    entity_id: object,
    *,
    ttl: int = SERVICE_ASSERTION_SECONDS,
    **claims: Any,
) -> str:
    """A short-lived assertion for one internal route and one company; extra ``claims``
    (``user_id=...``) ride along."""
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "scope": scope,
        "entity_id": str(entity_id),
        "iat": now,
        "exp": now + timedelta(seconds=ttl),
        **claims,
    }
    token: str = jwt.encode(payload, secret, algorithm=ALGORITHM)  # pyright: ignore[reportUnknownMemberType]
    return token


def assertion_headers(assertion: str) -> dict[str, str]:
    """The headers of a call to another Minty backend: the bearer, and the request's trace
    (``X-Request-ID``) so both services' log lines for it share one id."""
    return {"Authorization": f"Bearer {assertion}", **logctx.outgoing_headers()}


# ---- the hub session behind a token (v0.8) ------------------------------------------------------


class SessionCheckError(Exception):
    """minty-accounts-api could not say whether a session lives (unreachable, timed out, or an
    answer other than 200/410). Never read as "ended"."""


def touch_over_http(
    accounts_api_url: str, secret: str, sid: str, *, timeout: float = SESSION_TOUCH_TIMEOUT_SECONDS
) -> bool:
    """Ask minty-accounts-api whether the hub session ``sid`` names still lives, keeping it alive
    if so: True on 200, False on 410 (it has ended), else :class:`SessionCheckError` - a 404
    included, so an accounts-api without the route never signs anyone out."""
    if not accounts_api_url:
        raise SessionCheckError("ACCOUNTS_API_URL is not set")
    assertion = service_assertion(secret, SESSION_TOUCH_SCOPE, "")
    request = urllib.request.Request(
        accounts_api_url.rstrip("/") + SESSION_TOUCH_PATH,
        data=json.dumps({"sid": sid}).encode(),
        method="POST",
        headers={"Content-Type": "application/json", **assertion_headers(assertion)},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):  # noqa: S310 - our own configured address
            return True
    except urllib.error.HTTPError as exc:
        if exc.code == 410:
            return False
        raise SessionCheckError(f"accounts-api answered {exc.code}") from exc
    except (OSError, TimeoutError) as exc:  # URLError is an OSError
        raise SessionCheckError(f"accounts-api unreachable: {exc}") from exc


class SessionGate:
    """Whether a token's hub session lives, asked at most once per ``interval`` per ``sid`` in this
    process. ``touch(sid)`` answers True (alive, kept alive), False (ended) or raises
    :class:`SessionCheckError`."""

    def __init__(
        self,
        *,
        interval: float = SESSION_CHECK_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._interval = interval
        self._clock = clock
        self._seen: dict[str, tuple[float, bool]] = {}
        self._lock = threading.Lock()

    def alive(self, sid: str | None, touch: Callable[[str], bool]) -> bool:
        if not sid:
            return True  # minted before v0.8: lives until it expires
        now = self._clock()
        with self._lock:
            seen = self._seen.get(sid)
        if seen is not None and now - seen[0] < self._interval:
            return seen[1]
        try:
            answer = bool(touch(sid))
        except SessionCheckError as exc:
            logger.error(
                "session check: minty-accounts-api could not answer (%s); the token is honoured "
                "and asked about again in %d s",
                exc,
                self._interval,
            )
            answer = True
        with self._lock:
            if len(self._seen) >= _SESSION_CACHE_LIMIT:
                self._seen = {k: v for k, v in self._seen.items() if now - v[0] < self._interval}
            self._seen[sid] = (now, answer)
        return answer

    def forget(self) -> None:
        with self._lock:
            self._seen.clear()


#: This process's gate (every API's bearer auth goes through it).
SESSIONS = SessionGate()


def session_alive(sid: str | None, accounts_api_url: str, secret: str) -> bool:
    """Whether the hub session behind a token's ``sid`` lives (see the module note): False means
    refuse the token with a 401, which each app answers by handing the person over again."""
    return SESSIONS.alive(sid, lambda s: touch_over_http(accounts_api_url, secret, s))


# ---- telling the two 401s apart (v0.8, owner 2026-11-19) ----------------------------------------

#: The ``reason`` a 401 carries when the hub sign-in behind the token has ended (signed out, or 60
#: minutes without activity): each app shows "Your session has expired" and sends the person to
#: sign in, instead of renewing the token quietly. A 401 without it (a token that merely ran out
#: while its sign-in lives, or one with no ``sid``) is renewed quietly as before.
SESSION_EXPIRED_REASON = "session_expired"
_ENDED_ATTR = "minty_session_ended"


def session_ended(
    token: str,
    accounts_api_url: str,
    secret: str,
    *,
    touch: Callable[[str], bool] | None = None,
) -> bool:
    """True when ``token`` is genuinely ours (its signature verifies - its EXPIRY is ignored, for
    this question only) and carries a ``sid`` whose hub sign-in has ended. A forged or malformed
    token, or one with no ``sid``, is never a session end. Asked through the process cache
    (:data:`SESSIONS`), by HTTP, or by ``touch`` (minty-accounts-api's own store)."""
    try:
        claims = decode_bearer(token, secret, options={"verify_exp": False})
    except jwt.PyJWTError:
        return False
    sid = claims.get("sid")
    if not isinstance(sid, str) or not sid:
        return False
    if touch is None:
        return not session_alive(sid, accounts_api_url, secret)
    return not SESSIONS.alive(sid, touch)


def mark_session_ended(request: Any) -> None:
    """Note on ``request`` that its token was refused because its sign-in has ended, for the 401
    handler (:func:`unauthorized_reason`)."""
    setattr(request, _ENDED_ATTR, True)


def unauthorized_reason(request: Any) -> dict[str, str]:
    """``{"reason": "session_expired"}`` to add to a 401's body when :func:`mark_session_ended`
    marked ``request``, else nothing."""
    return {"reason": SESSION_EXPIRED_REASON} if getattr(request, _ENDED_ATTR, False) else {}
