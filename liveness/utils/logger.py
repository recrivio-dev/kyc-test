"""Logging for the liveness package.

Uses the stdlib logger the rest of this service already logs through
(`uvicorn.error`), rather than live-mini's loguru + rotating file sink: in a
container the file sink writes to a throwaway layer and duplicates what the
Docker json-file driver already captures.

Loguru's brace-style call signature — logger.info("{} frames", n) — is
preserved via a thin adapter so the ported call sites need no edits.
"""
from __future__ import annotations

import logging

_base = logging.getLogger("uvicorn.error")


class _BraceLogger:
    """Adapts loguru's `logger.info("{}", x)` onto stdlib `%`-style logging."""

    @staticmethod
    def _fmt(msg: str, args: tuple) -> str:
        if not args:
            return str(msg)
        try:
            return str(msg).format(*args)
        except (IndexError, KeyError, ValueError):
            return f"{msg} {args}"

    def debug(self, msg, *a, **kw):    _base.debug(self._fmt(msg, a))
    def info(self, msg, *a, **kw):     _base.info(self._fmt(msg, a))
    def success(self, msg, *a, **kw):  _base.info(self._fmt(msg, a))
    def warning(self, msg, *a, **kw):  _base.warning(self._fmt(msg, a))
    def error(self, msg, *a, **kw):    _base.error(self._fmt(msg, a))
    def exception(self, msg, *a, **kw): _base.exception(self._fmt(msg, a))


logger = _BraceLogger()


def configure_logging() -> None:
    """No-op — gunicorn/uvicorn own the handler configuration."""


__all__ = ["logger", "configure_logging"]
