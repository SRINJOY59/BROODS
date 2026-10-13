"""Thin stdlib client for the proxy's control plane.

Shared by the runner (drives probes from outside) and by probes themselves
(arm a scenario, read timing back). One implementation so both call sites
agree on what "reset" and "stats" mean.
"""
from __future__ import annotations

import json
import urllib.request


class ProxyClient:
    def __init__(self, port: int) -> None:
        self.port = port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=30))

    def _get(self, path: str) -> dict:
        return json.load(urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}{path}", timeout=30))

    def reset(self, scenario: str, cap: int = 200, hang_s: float = 20.0,
              fail_at: int | None = None, delay_s: float = 0.0,
              fail_status: int = 400, fail_repeat: int = 1,
              fail_match: str | None = None, fanout: int = 4,
              upstream: str | None = None, api_key: str | None = None,
              usage: str = "fixed") -> None:
        self._post("/__reset", {"scenario": scenario, "cap": cap, "hang_s": hang_s,
                                "fail_at": fail_at, "delay_s": delay_s,
                                "fail_status": fail_status, "fail_repeat": fail_repeat,
                                "fail_match": fail_match, "fanout": fanout,
                                "upstream": upstream, "api_key": api_key,
                                "usage": usage})

    def quiesce(self) -> None:
        self._post("/__quiesce", {})

    def cancel(self, t_cancel: float | None = None) -> float:
        """Mark the cancellation instant. Pass the framework's own timestamp
        where available -- the proxy's clock is a round-trip later."""
        return self._post("/__cancel", {"t_cancel": t_cancel})["t_cancel"]

    def stats(self) -> dict:
        return self._get("/__stats")
