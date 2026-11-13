"""Bounded HTTP transport for camera-model requests."""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Mapping, Protocol

MAX_RESPONSE_BYTES = 1_000_000


class TransportError(Exception):
    """Raised by an injected transport.  ``transient`` controls retries."""

    def __init__(self, message: str, *, transient: bool = True):
        super().__init__(message)
        self.transient = bool(transient)


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)


class Transport(Protocol):
    def request(
        self,
        url: str,
        *,
        method: str = "POST",
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:  # pragma: no cover - protocol definition
        ...


class UrllibTransport:
    """Default stdlib transport.  HTTP error responses become normal results."""

    def __init__(self, max_response_bytes: int = MAX_RESPONSE_BYTES):
        self.max_response_bytes = int(max_response_bytes)

    def request(
        self,
        url: str,
        *,
        method: str = "POST",
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
                return TransportResponse(
                    status=int(getattr(response, "status", 200)),
                    body=raw,
                    headers=dict(getattr(response, "headers", None) or {}),
                )
        except urllib.error.HTTPError as exc:
            raw = b""
            try:
                raw = exc.read(self.max_response_bytes + 1)
            except OSError:
                pass
            return TransportResponse(
                status=int(exc.code),
                body=raw,
                headers=dict(exc.headers or {}),
            )

    def close(self) -> None:
        return None
