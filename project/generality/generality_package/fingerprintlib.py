"""Compatibility wrapper for compact, non-cryptographic cache IDs."""
from rineng_content_id import content_identity


def fingerprint(data=b""):
    return content_identity(data)
