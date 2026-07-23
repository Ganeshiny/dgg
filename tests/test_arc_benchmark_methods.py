from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from arc_benchmark_methods import (  # noqa: E402
    canonicalize_similarity_id,
    parse_similarity_hits,
    transfer_scores,
)


def test_canonicalize_similarity_id_handles_blast_and_structure_ids():
    assert canonicalize_similarity_id("pdb|7FHK|A") == "7FHK_A"
    assert canonicalize_similarity_id("PDB|9SDP|BA") == "9SDP_BA"
    assert canonicalize_similarity_id("/tmp/7FHK_A.pdb") == "7FHK_A"
    assert canonicalize_similarity_id("/tmp/7FHK_A.pdb.gz") == "7FHK_A"
    assert canonicalize_similarity_id("7FHK_A") == "7FHK_A"


def test_blast_pdb_target_ids_transfer_locked_training_labels(tmp_path):
    hit_file = tmp_path / "blast.tsv"
    hit_file.write_text(
        "QUERY_A\tpdb|7FHK|A\t99.9\t100\t100\t100\t0.0\t250\t100\n"
    )

    hits = parse_similarity_hits(hit_file)
    scores = transfer_scores(
        ["QUERY_A"],
        ["7FHK_A"],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        hits,
        top_k=10,
        min_qcov=0.5,
        min_tcov=0.5,
    )

    np.testing.assert_array_equal(scores, np.asarray([[1.0, 0.0]], dtype=np.float32))
