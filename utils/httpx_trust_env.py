from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


def trust_env_for_http_url(url: str) -> bool:
    """Use proxy env vars for public hosts, but keep local/private endpoints direct."""
    try:
        host = urlparse(url).hostname
    except Exception:
        return True
    if not host:
        return True

    h = host.strip().lower()
    if h == "localhost" or h.endswith(".local"):
        return False

    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return True
    return not (ip.is_loopback or ip.is_private or ip.is_link_local)
