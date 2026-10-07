"""Versioned prompt registry — provenance for Council / Justice / Argus.

SCOPE §22.2 binding: "Every system prompt has a hash and version string.
Runs record prompt versions used. Changing a prompt mid-project invalidates
prior runs for that agent unless re-verified."

This module provides the seam that makes that contract enforceable:

  - ``Prompt(name, version, body)`` — immutable record of one prompt revision.
    Computes a stable sha256 of the body so the JSONL row's
    ``system_prompt_hash`` (Phase 02 / D-14) can be back-resolved to the
    exact prompt revision that produced a call.

  - ``register(prompt)`` — adds it to the global registry. **Idempotent for
    identical re-registration; raises ``PromptVersionConflict`` if
    (name, version) already exists with a different body.** This catches
    the most common accidental violation: a Council agent's prompt gets
    edited in place, all old runs become unreplayable, but the version
    string never bumps.

  - ``get(name, version)`` — lookup at agent construction.

  - ``all_prompts()`` — snapshot for run-config JSON dump. Phase 16.x
    ablation runs MUST capture this into the run config so replay against
    the exact prompt revision is possible.

Why this is non-retrofittable:

Once a Council run lands JSONL rows with system_prompt_hash=H1, that hash
is the only durable record of what the agent saw. If you later edit the
prompt and re-run, the new hash H2 cannot be cross-referenced back to a
file unless someone went through this registry first. Bolt-on after the
fact loses the chain.

Why versioning AND hashing:

The hash detects accidental drift (different body → different hash).
The version string is human-meaningful and gets cited in the paper's
methods section ("Plato v1.2 used the system prompt at commit
abc1234..."). The version is what reviewers ask for; the hash is what
proves the version did not silently change.

Usage (Phase 11.0 Council agent registration):

    from ascension.common.prompts import Prompt, register

    plato_prompt = Prompt(
        name="plato",
        version="v1",
        body=Path("prompts/plato_v1.md").read_text(encoding="utf-8"),
    )
    register(plato_prompt)
    # ... later:
    response = await client.generate(
        role="plato", prompt=user_question, system=plato_prompt.body,
    )

Run-config capture (Phase 16.x ablation):

    from ascension.common.prompts import all_prompts

    config = {
        "seed": seed,
        "model_versions": {role: settings.model_for(role) for role in roles},
        "prompts": [
            {"name": p.name, "version": p.version, "body_sha256": p.body_sha256}
            for p in all_prompts()
        ],
        ...
    }
    Path(f"runs/{run_id}/config.json").write_text(json.dumps(config, indent=2))
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prompt:
    """Immutable record of one prompt revision.

    `name` is the agent role (e.g., "plato", "judge"). `version` is a
    human-meaningful semver-ish string (e.g., "v1", "v1.2.0-experimental").
    `body` is the full system-prompt text that will be passed verbatim to
    LLMClient.generate(..., system=body).
    """

    name: str
    version: str
    body: str

    @property
    def body_sha256(self) -> str:
        """Stable hash of the body — matches the `system_prompt_hash` field
        written by `ascension.llm.sink.build_call_record` for any call that
        used this prompt as `system`.
        """
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()

    @property
    def body_len_chars(self) -> int:
        """Cheap pre-call estimate input — agents using long prompts can
        check this against the cost ceiling before issuing a call.
        """
        return len(self.body)


class PromptVersionConflict(ValueError):
    """(name, version) already registered with a different body.

    Per SCOPE §22.2: prompt edits invalidate prior runs unless re-verified.
    Bumping the version string is the documented escape hatch.
    """


_REGISTRY: dict[tuple[str, str], Prompt] = {}


def register(prompt: Prompt) -> Prompt:
    """Add a Prompt to the global registry.

    Idempotent for identical re-registration (returns the existing entry).
    Raises ``PromptVersionConflict`` if (name, version) already exists with
    a different body — this catches accidental edits that would silently
    retcon a prompt's history.

    Returns the canonical registered Prompt instance — callers should use
    the return value rather than the input so identity comparisons work
    across registration sites.
    """
    key = (prompt.name, prompt.version)
    existing = _REGISTRY.get(key)
    if existing is not None:
        if existing.body == prompt.body:
            return existing
        raise PromptVersionConflict(
            f"Prompt (name={prompt.name!r}, version={prompt.version!r}) is "
            "already registered with a different body. Bump the version "
            "string instead of editing in place — see SCOPE §22.2 "
            f"(existing body sha256={existing.body_sha256[:12]}, "
            f"new body sha256={prompt.body_sha256[:12]})."
        )
    _REGISTRY[key] = prompt
    return prompt


def get(name: str, version: str) -> Prompt:
    """Lookup a registered Prompt or raise KeyError.

    Use at agent construction so the prompt's version + hash are guaranteed
    to be in the registry when run-config dump captures the snapshot.
    """
    key = (name, version)
    if key not in _REGISTRY:
        raise KeyError(
            f"Prompt (name={name!r}, version={version!r}) not registered. "
            "Call ascension.common.prompts.register() at agent construction."
        )
    return _REGISTRY[key]


def all_prompts() -> list[Prompt]:
    """Snapshot of the registry, ordered by (name, version) for stable
    JSON dumps. Used by the run-config capture path (Phase 16.x).
    """
    return [_REGISTRY[k] for k in sorted(_REGISTRY.keys())]


def reset_for_tests() -> None:
    """Drop the global registry. Test-only escape hatch — production code
    MUST NOT call this; it invalidates every captured Prompt reference.
    """
    _REGISTRY.clear()
