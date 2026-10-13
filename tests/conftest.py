import json
import socket
import urllib.request

import pytest

from broods.proxy import serve
from broods.proxy_client import ProxyClient


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def proxy() -> ProxyClient:
    """One injector for the whole session. Its state is a module global."""
    port = _free_port()
    serve(port)
    return ProxyClient(port)


@pytest.fixture
def chat(proxy):
    """Send one chat request to the proxy and return the parsed reply."""
    def _chat(messages, tools=None):
        body = {"model": "mock", "messages": messages}
        if tools is not None:
            body["tools"] = tools
        req = urllib.request.Request(
            f"{proxy.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=10))
    return _chat
