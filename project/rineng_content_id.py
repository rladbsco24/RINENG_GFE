"""Small non-cryptographic content IDs for ephemeral cache names.

These IDs are not provenance records or integrity claims.  Scientific inputs
are checked by shape, dtype, finite values, and direct numerical comparisons.
"""
from __future__ import annotations

import zlib


class _ContentIdentity:
    def __init__(self, data=b""):
        self._cycle = 0
        self._rolling = 1
        self._size = 0
        if data:
            self.update(data)

    def update(self, data):
        block = bytes(data)
        self._cycle = zlib.crc32(block, self._cycle) & 0xFFFFFFFF
        self._rolling = zlib.adler32(block, self._rolling) & 0xFFFFFFFF
        self._size += len(block)
        return self

    def digest(self):
        return (self._size.to_bytes(8, "big") + self._cycle.to_bytes(4, "big")
                + self._rolling.to_bytes(4, "big"))

    def hexdigest(self):
        return self.digest().hex()

    def readable(self):
        return self.hexdigest()


def content_identity(data=b""):
    return _ContentIdentity(data)


content_id = content_identity
fingerprint = content_identity
