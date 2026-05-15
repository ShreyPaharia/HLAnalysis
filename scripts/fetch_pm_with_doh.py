"""Bootstrap script: route polymarket DNS lookups through Cloudflare's DoH so
the `hl-bt fetch` PM path works on networks where the system resolver blocks
polymarket.com (e.g. Pi-hole / NextDNS deny rules). Drop-in wrapper around
`hlanalysis.backtest.cli.main`.

Also retries transient timeouts on the PM data-api so a one-off network blip
doesn't kill a multi-hour fetch.
"""
from __future__ import annotations

import socket
import sys
import time

import dns.resolver
import requests

_CF = dns.resolver.Resolver(configure=False)
_CF.nameservers = ["1.1.1.1", "1.0.0.1", "8.8.8.8"]
_CF.lifetime = 30.0
_CF.timeout = 10.0

_real_getaddrinfo = socket.getaddrinfo
_dns_cache: dict[str, str] = {}


def _resolve_via_cf(host: str) -> str:
    if host in _dns_cache:
        return _dns_cache[host]
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            ans = _CF.resolve(host, "A")
            ip = ans[0].address
            _dns_cache[host] = ip
            return ip
        except Exception as e:
            last_exc = e
            time.sleep(2 ** attempt)
    raise last_exc  # type: ignore[misc]


def _patched_getaddrinfo(host, port, *args, **kwargs):
    try:
        return _real_getaddrinfo(host, port, *args, **kwargs)
    except socket.gaierror:
        if not isinstance(host, str) or not host.endswith("polymarket.com"):
            raise
        ip = _resolve_via_cf(host)
        return _real_getaddrinfo(ip, port, *args, **kwargs)


socket.getaddrinfo = _patched_getaddrinfo


# Wrap requests.get with retry on transient timeouts. PM data-api occasionally
# stalls past the 30s timeout under load; without retry the whole multi-hour
# fetch dies.
_real_get = requests.get
_MAX_RETRIES = 5
_BACKOFF_SECONDS = (1, 2, 5, 10, 20)


_TRANSIENT_EXC = (
    requests.Timeout,
    requests.ConnectionError,
    dns.resolver.LifetimeTimeout,
    dns.exception.DNSException,
    socket.gaierror,
    OSError,
)


def _retrying_get(url, *args, **kwargs):
    kwargs.setdefault("timeout", 60)
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return _real_get(url, *args, **kwargs)
        except _TRANSIENT_EXC as e:
            last_exc = e
            backoff = _BACKOFF_SECONDS[attempt]
            print(
                f"[retry] {type(e).__name__} on {url[:80]} (attempt "
                f"{attempt + 1}/{_MAX_RETRIES}); sleeping {backoff}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(backoff)
    raise last_exc  # type: ignore[misc]


requests.get = _retrying_get  # type: ignore[assignment]


if __name__ == "__main__":
    from hlanalysis.backtest.cli import main as cli_main

    sys.exit(cli_main(sys.argv[1:]))
