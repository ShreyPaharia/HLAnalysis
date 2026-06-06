"""The reconcile loop must treat expected, self-recovering venue read failures
(HL 429 bursts, connection blips, 5xx) as transient — logging them concisely
and retrying next cycle — rather than emitting a full `reconcile crashed`
traceback that reads like a bug. `_is_transient_venue_error` is the classifier
that distinguishes the two; this pins its behaviour for both venues."""
from __future__ import annotations

import requests

from hlanalysis.engine.hl_client import RateLimitError, RestError
from hlanalysis.engine.runtime import _is_transient_venue_error


def _http_error(status_code: int) -> requests.exceptions.HTTPError:
    resp = requests.Response()
    resp.status_code = status_code
    return requests.exceptions.HTTPError(response=resp)


def test_hl_rate_limit_is_transient():
    # The exact failure that produced the scary "reconcile crashed alias=v31"
    # traceback: an exhausted HL 429 retry.
    assert _is_transient_venue_error(RateLimitError("(429, ...)")) is True


def test_connection_and_timeout_are_transient():
    assert _is_transient_venue_error(ConnectionError("conn reset")) is True
    assert _is_transient_venue_error(TimeoutError("timed out")) is True
    assert _is_transient_venue_error(requests.exceptions.ConnectionError()) is True
    assert _is_transient_venue_error(requests.exceptions.Timeout()) is True


def test_pm_5xx_and_429_are_transient():
    assert _is_transient_venue_error(_http_error(503)) is True
    assert _is_transient_venue_error(_http_error(429)) is True


def test_non_transient_errors_are_not_swallowed():
    # A real business/client error (4xx) and an outright bug must still get the
    # full "crashed" traceback path — they are not transient.
    assert _is_transient_venue_error(_http_error(400)) is False
    assert _is_transient_venue_error(RestError("malformed response")) is False
    assert _is_transient_venue_error(ValueError("logic bug")) is False
    assert _is_transient_venue_error(KeyError("missing field")) is False
