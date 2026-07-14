"""
preprocessing/pdb_clusters/build_entity_map.py
==============================================
Builds a cached JSON mapping:
    pdb_id -> { auth_chain -> entity_num }
by parsing the _pdbx_poly_seq_scheme block in all local CIF.gz files.

Run once before split_by_pdb_clusters.py.

Usage:
    python3 preprocessing/pdb_clusters/build_entity_map.py
"""

import gzip, json, os, sys
from pathlib import Path

PROJECT_DIR   = Path(__file__).resolve().parent.parent.parent
STRUCT_DIR    = PROJECT_DIR / 'preprocessing' / 'data' / 'structure_files'
OUT_JSON      = PROJECT_DIR / 'preprocessing' / 'data' / 'entity_map.json'


def parse_chain_entity(cif_gz_path: Path) -> dict[str, str]:
    """Return {auth_chain: entity_id_str} for one CIF file."""
    mapping: dict[str, str] = {}
    try:
        with gzip.open(cif_gz_path, 'rt', errors='replace') as fh:
            content = fh.read()
    except Exception:
        return mapping

    lines = content.split('\n')
    i = 0
    while i < len(lines):
        if lines[i].strip() == 'loop_':
            j = i + 1
            block_headers: list[str] = []
            while j < len(lines) and lines[j].strip().startswith('_pdbx_poly_seq_scheme'):
                block_headers.append(lines[j].strip())
                j += 1

            if not block_headers:
                i = j
                continue

            # Locate column indices
            entity_col = next(
                (k for k, h in enumerate(block_headers)
                 if h == '_pdbx_poly_seq_scheme.entity_id'), None)
            strand_col = next(
                (k for k, h in enumerate(block_headers)
                 if h == '_pdbx_poly_seq_scheme.pdb_strand_id'), None)

            if entity_col is None or strand_col is None:
                i = j
                continue

            # Collect unique (auth_chain -> entity) pairs
            k = j
            while k < len(lines):
                row = lines[k].strip()
                if not row or row.startswith('#') or row.startswith('_') or row == 'loop_':
                    break
                parts = row.split()
                needed = max(entity_col, strand_col)
                if len(parts) > needed:
                    auth_chain = parts[strand_col]
                    entity_id  = parts[entity_col]
                    mapping[auth_chain] = entity_id
                k += 1
            return mapping   # only one such block per file
        i += 1
    return mapping


def main() -> None:
    cif_files = sorted(STRUCT_DIR.glob('*.cif.gz'))
    print(f'Found {len(cif_files)} CIF.gz files in {STRUCT_DIR}')

    result: dict[str, dict[str, str]] = {}
    missing_block: list[str] = []

    for idx, cif_path in enumerate(cif_files):
        pdb_id = cif_path.name.replace('.cif.gz', '').upper()
        chain_map = parse_chain_entity(cif_path)
        if chain_map:
            result[pdb_id] = chain_map
        else:
            missing_block.append(pdb_id)

        if (idx + 1) % 500 == 0:
            print(f'  Parsed {idx+1}/{len(cif_files)} …')

    print(f'Successfully parsed: {len(result)} PDB IDs')
    print(f'No pdbx_poly_seq_scheme block: {len(missing_block)} (e.g. {missing_block[:5]})')

    with open(OUT_JSON, 'w') as fh:
        json.dump(result, fh)
    print(f'Saved entity map → {OUT_JSON}')


if __name__ == '__main__':
    main()
