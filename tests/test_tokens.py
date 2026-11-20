"""minty_shared.tokens: the bearer decode (60 s skew everywhere) and the service assertion."""

import time
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from minty_shared import logctx, tokens

SECRET = "test-secret-key-shared-with-flask-and-long-enough"


@pytest.fixture(autouse=True)
def _clean():
    logctx.reset()
    yield
    logctx.reset()


def _mint(**claims):
    return jwt.encode(claims, SECRET, algorithm="HS256")


def test_a_good_bearer_decodes():
    now = int(time.time())
    assert (
        tokens.decode_bearer(_mint(user_id="u1", iat=now, exp=now + 60), SECRET)["user_id"] == "u1"
    )


def test_a_token_minted_slightly_in_the_future_is_accepted():
    # Flask's clock 30 s ahead of this host: within the skew, so not "not yet valid".
    now = int(time.time())
    claims = tokens.decode_bearer(
        _mint(user_id="u1", iat=now + 30, nbf=now + 30, exp=now + 600), SECRET
    )
    assert claims["user_id"] == "u1"


def test_a_token_expired_within_the_skew_still_decodes_and_beyond_it_does_not():
    now = int(time.time())
    assert tokens.decode_bearer(_mint(exp=now - 30), SECRET) == {"exp": now - 30}
    with pytest.raises(jwt.ExpiredSignatureError):
        tokens.decode_bearer(_mint(exp=now - 120), SECRET)


def test_a_forged_token_is_refused():
    forged = jwt.encode(
        {"user_id": "u1"}, "not-the-key-not-the-key-not-the-key!!", algorithm="HS256"
    )
    with pytest.raises(jwt.InvalidSignatureError):
        tokens.decode_bearer(forged, SECRET)


def test_another_algorithm_is_refused():
    unsigned = jwt.encode({"user_id": "u1"}, None, algorithm="none")
    with pytest.raises(jwt.InvalidAlgorithmError):
        tokens.decode_bearer(unsigned, SECRET)


def test_required_claims_pass_through():
    with pytest.raises(jwt.MissingRequiredClaimError):
        tokens.decode_bearer(_mint(user_id="u1"), SECRET, options={"require": ["exp"]})


def test_a_service_assertion_is_scoped_short_lived_and_names_the_company():
    before = datetime.now(UTC)
    token = tokens.service_assertion(SECRET, "xero-access-token", 42, user_id="u9")
    claims = jwt.decode(token, SECRET, algorithms=["HS256"])
    assert (claims["scope"], claims["entity_id"], claims["user_id"]) == (
        "xero-access-token",
        "42",
        "u9",
    )
    expires = datetime.fromtimestamp(claims["exp"], UTC)
    assert timedelta(seconds=55) <= expires - before <= timedelta(seconds=62)


def test_assertion_headers_carry_the_trace():
    assert tokens.assertion_headers("t") == {"Authorization": "Bearer t"}
    logctx.start_trace("a1b2c3d4")
    assert tokens.assertion_headers("t") == {
        "Authorization": "Bearer t",
        "X-Request-ID": "a1b2c3d4",
    }


# ---- the hub session behind a token (v0.8: sign-out after 60 minutes without activity) ---------


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _touches(answers):
    calls = []

    def touch(sid):
        calls.append(sid)
        answer = answers[sid]
        if isinstance(answer, Exception):
            raise answer
        return answer

    return touch, calls


def test_a_token_with_no_sid_is_alive_without_asking():
    touch, calls = _touches({})
    gate = tokens.SessionGate(clock=_Clock())
    assert gate.alive(None, touch) and gate.alive("", touch)
    assert calls == []


def test_a_live_session_is_asked_about_at_most_once_a_minute():
    clock = _Clock()
    touch, calls = _touches({"s1": True})
    gate = tokens.SessionGate(clock=clock)
    assert gate.alive("s1", touch)
    clock.now += 59
    assert gate.alive("s1", touch)
    assert calls == ["s1"]
    clock.now += 2
    assert gate.alive("s1", touch)
    assert calls == ["s1", "s1"]


def test_an_ended_session_is_dead_and_each_sid_is_asked_about_separately():
    touch, calls = _touches({"gone": False, "s2": True})
    gate = tokens.SessionGate(clock=_Clock())
    assert gate.alive("gone", touch) is False
    assert gate.alive("gone", touch) is False
    assert gate.alive("s2", touch) is True
    assert calls == ["gone", "s2"]


def test_accounts_api_unreachable_honours_a_sign_in_confirmed_in_the_last_five_minutes(caplog):
    """v0.9 (owner, 2026-11-19): a blip goes unnoticed - the token is honoured, logged at ERROR, and
    asked about again a minute later - but only within 5 minutes of the last time accounts-api
    said this sign-in lives. Past that the request is refused (SessionCheckUnavailableError, a 503)."""
    clock = _Clock()
    answers = {"s1": True}
    touch, calls = _touches(answers)
    gate = tokens.SessionGate(clock=clock)
    assert gate.alive("s1", touch) is True  # confirmed at 1000
    answers["s1"] = tokens.SessionCheckError("connection refused")
    clock.now += 61
    with caplog.at_level("ERROR", logger="minty_shared.tokens"):
        assert gate.alive("s1", touch) is True
    assert "connection refused" in caplog.text and "s1" not in caplog.text
    assert gate.alive("s1", touch) is True and calls == ["s1", "s1"]  # not asked again yet
    clock.now += 61
    assert gate.alive("s1", touch) is True and len(calls) == 3
    clock.now = 1000 + tokens.SESSION_CHECK_GRACE_SECONDS - 1
    assert gate.alive("s1", touch) is True
    clock.now = 1000 + tokens.SESSION_CHECK_GRACE_SECONDS + 1
    with pytest.raises(tokens.SessionCheckUnavailableError) as refused:
        gate.alive("s1", touch)
    assert str(refused.value) == "Minty can't check your sign-in just now. Try again in a minute."


def test_a_sign_in_never_confirmed_is_refused_at_once_while_accounts_api_is_unreachable():
    touch, calls = _touches({"s1": tokens.SessionCheckError("timed out")})
    gate = tokens.SessionGate(clock=_Clock())
    with pytest.raises(tokens.SessionCheckUnavailableError):
        gate.alive("s1", touch)
    with pytest.raises(tokens.SessionCheckUnavailableError):
        gate.alive("s1", touch)
    assert calls == ["s1", "s1"], "a refusal is not remembered: the next request asks again"


def test_once_accounts_api_answers_again_the_sign_in_is_confirmed_afresh():
    clock = _Clock()
    answers = {"s1": tokens.SessionCheckError("down")}
    touch, _ = _touches(answers)
    gate = tokens.SessionGate(clock=clock)
    with pytest.raises(tokens.SessionCheckUnavailableError):
        gate.alive("s1", touch)
    answers["s1"] = True
    assert gate.alive("s1", touch) is True
    answers["s1"] = tokens.SessionCheckError("down again")
    clock.now += tokens.SESSION_CHECK_GRACE_SECONDS - 1
    assert gate.alive("s1", touch) is True


def test_an_ended_sign_in_stays_ended_while_accounts_api_is_unreachable():
    """An ended sign-in never comes back, so it is still refused as ended (401, "Your session has
    expired") - not a 503."""
    clock = _Clock()
    answers = {"gone": False}
    touch, _ = _touches(answers)
    gate = tokens.SessionGate(clock=clock)
    assert gate.alive("gone", touch) is False
    answers["gone"] = tokens.SessionCheckError("down")
    clock.now += tokens.SESSION_CHECK_GRACE_SECONDS + 61
    assert gate.alive("gone", touch) is False


def test_forget_empties_the_cache():
    touch, calls = _touches({"s1": True})
    gate = tokens.SessionGate(clock=_Clock())
    gate.alive("s1", touch)
    gate.forget()
    gate.alive("s1", touch)
    assert calls == ["s1", "s1"]


class _Answer:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http(monkeypatch, outcome):
    import urllib.error

    seen = {}

    def urlopen(request, timeout):
        seen.update(
            url=request.full_url,
            method=request.get_method(),
            body=request.data,
            headers=dict(request.header_items()),
            timeout=timeout,
        )
        if isinstance(outcome, int) and outcome >= 400:
            raise urllib.error.HTTPError(request.full_url, outcome, "x", {}, None)
        if isinstance(outcome, Exception):
            raise outcome
        return _Answer(outcome)

    monkeypatch.setattr(tokens.urllib.request, "urlopen", urlopen)
    return seen


def test_the_touch_posts_the_sid_with_a_scoped_assertion(monkeypatch):
    import json

    seen = _http(monkeypatch, 200)
    assert tokens.touch_over_http("https://accounts.test/", SECRET, "s1") is True
    assert (seen["url"], seen["method"]) == (
        "https://accounts.test/api/internal/session/touch",
        "POST",
    )
    assert json.loads(seen["body"]) == {"sid": "s1"}
    assert seen["timeout"] == tokens.SESSION_TOUCH_TIMEOUT_SECONDS
    assertion = seen["headers"]["Authorization"].removeprefix("Bearer ")
    claims = jwt.decode(assertion, SECRET, algorithms=["HS256"])
    assert claims["scope"] == tokens.SESSION_TOUCH_SCOPE and claims["exp"] - claims["iat"] <= 60


def test_410_means_the_session_has_ended(monkeypatch):
    _http(monkeypatch, 410)
    assert tokens.touch_over_http("https://accounts.test", SECRET, "s1") is False


@pytest.mark.parametrize("outcome", [404, 401, 500, OSError("refused"), TimeoutError()])
def test_anything_else_is_unavailable_never_ended(monkeypatch, outcome):
    # A 404 too: an accounts-api without the route (deployed after this API) must not sign
    # everyone out.
    _http(monkeypatch, outcome)
    with pytest.raises(tokens.SessionCheckError):
        tokens.touch_over_http("https://accounts.test", SECRET, "s1")


def test_no_address_is_unavailable():
    with pytest.raises(tokens.SessionCheckError):
        tokens.touch_over_http("", SECRET, "s1")


def test_session_alive_goes_through_the_process_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tokens,
        "touch_over_http",
        lambda url, secret, sid: calls.append((url, sid)) or sid != "gone",
    )
    tokens.SESSIONS.forget()
    try:
        assert tokens.session_alive("s1", "https://a.test", SECRET) is True
        assert tokens.session_alive("s1", "https://a.test", SECRET) is True
        assert tokens.session_alive("gone", "https://a.test", SECRET) is False
        assert tokens.session_alive(None, "https://a.test", SECRET) is True
        assert calls == [("https://a.test", "s1"), ("https://a.test", "gone")]
    finally:
        tokens.SESSIONS.forget()


# ---- telling the two 401s apart (v0.8, owner 2026-11-19: "Your session has expired") ----------


@pytest.fixture
def _sessions(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tokens,
        "touch_over_http",
        lambda url, secret, sid: calls.append(sid) or sid != "gone",
    )
    tokens.SESSIONS.forget()
    yield calls
    tokens.SESSIONS.forget()


def _at(seconds_from_now, **claims):
    now = datetime.now(UTC)
    return _mint(
        iat=now - timedelta(minutes=31), exp=now + timedelta(seconds=seconds_from_now), **claims
    )


def test_an_expired_token_whose_sign_in_has_ended_is_a_session_end(_sessions):
    assert tokens.session_ended(_at(-3600, sid="gone"), "https://a.test", SECRET) is True
    assert _sessions == ["gone"]


def test_a_valid_token_whose_sign_in_has_ended_is_a_session_end(_sessions):
    assert tokens.session_ended(_at(600, sid="gone"), "https://a.test", SECRET) is True


def test_an_expired_token_whose_sign_in_lives_is_not(_sessions):
    assert tokens.session_ended(_at(-3600, sid="s1"), "https://a.test", SECRET) is False


def test_an_expired_token_is_no_session_end_while_accounts_api_cannot_answer(monkeypatch):
    """Already refused for its expiry: unable to ask, it is a plain 401 (renewed quietly), never
    a 503 and never "your session has expired"."""

    def down(url, secret, sid):
        raise tokens.SessionCheckError("down")

    monkeypatch.setattr(tokens, "touch_over_http", down)
    tokens.SESSIONS.forget()
    try:
        assert tokens.session_ended(_at(-3600, sid="s1"), "https://a.test", SECRET) is False
    finally:
        tokens.SESSIONS.forget()


def test_no_sid_a_forged_or_a_malformed_token_is_never_a_session_end(_sessions):
    assert tokens.session_ended(_at(-3600), "https://a.test", SECRET) is False
    forged = jwt.encode(
        {"sid": "gone", "exp": 1}, "another-key-long-enough-for-hs256-x", algorithm="HS256"
    )
    assert tokens.session_ended(forged, "https://a.test", SECRET) is False
    assert tokens.session_ended("not-a-token", "https://a.test", SECRET) is False
    assert _sessions == []


def test_a_session_end_can_be_asked_with_the_callers_own_touch(_sessions):
    asked = []
    touch = lambda sid: asked.append(sid) or False  # noqa: E731
    assert tokens.session_ended(_at(-60, sid="s9"), "", SECRET, touch=touch) is True
    assert asked == ["s9"] and _sessions == []


def test_the_reason_rides_on_a_marked_request_only():
    class Request:
        pass

    marked, plain = Request(), Request()
    tokens.mark_session_ended(marked)
    assert tokens.unauthorized_reason(marked) == {"reason": "session_expired"}
    assert tokens.unauthorized_reason(plain) == {}
    assert tokens.SESSION_EXPIRED_REASON == "session_expired"
