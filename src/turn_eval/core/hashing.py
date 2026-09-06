"""Canonical serialization and content hashing.

These live apart from :mod:`turn_eval.artifacts` so that isolated model
environments can import them without pulling in the artifact writers. Several
of those environments are pinned to older Python versions than the core
package, and the writers depend on newer standard-library features.
"""

from __future__ import annotations

import hashlib
import json


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()
