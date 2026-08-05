"""Resolve a complete graph cache without trusting stale serialized paths."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def resolve_complete_graph_cache(
    protein_ids: Iterable[str],
    candidates: Iterable[tuple[str, str | Path | None]],
) -> Path:
    """Return the first candidate containing every required protein graph.

    Dataset pickles preserve an absolute graph directory, which can become stale
    after moving a checkout. Callers therefore provide ordered candidate roots.
    Every candidate is audited against all required protein IDs; a partial cache
    is never accepted merely because the directory exists.
    """
    required = sorted({str(protein_id) for protein_id in protein_ids})
    if not required:
        raise ValueError("Cannot resolve a graph cache for an empty protein set")

    seen: set[Path] = set()
    audits: list[str] = []
    for label, raw_root in candidates:
        if raw_root is None or not str(raw_root).strip():
            continue
        root = Path(raw_root).expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        if not root.is_dir():
            audits.append(f"{label}: {root} [directory missing]")
            continue

        missing_count = 0
        missing_examples: list[str] = []
        for protein_id in required:
            if not (root / f"{protein_id}.pt").is_file():
                missing_count += 1
                if len(missing_examples) < 10:
                    missing_examples.append(protein_id)
        if missing_count == 0:
            return root
        audits.append(
            f"{label}: {root} [missing {missing_count}/{len(required)} graphs; "
            f"examples: {', '.join(missing_examples)}]"
        )

    detail = "\n  ".join(audits) if audits else "no candidate paths were supplied"
    raise FileNotFoundError(
        f"No complete graph cache contains all {len(required)} required proteins. "
        f"Candidate audit:\n  {detail}"
    )
