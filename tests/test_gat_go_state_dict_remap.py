"""Verifies scripts/run_gat_go_arc.py's remap of GAT-GO's released checkpoint
(saved under torch_geometric < 2.0's GATConv/GraphConv parameter names) onto
the pinned torch_geometric==2.5.3's current names.

Skipped where torch_geometric is not installed; run it in an environment that
has it (e.g. the project's own conda env) to exercise these checks.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
    from torch_geometric.nn import GATConv, SAGPooling, global_mean_pool
    from torch import nn

    _HAS_TORCH_GEOMETRIC = True
except ImportError:
    _HAS_TORCH_GEOMETRIC = False


if _HAS_TORCH_GEOMETRIC:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_gat_go_arc import remap_legacy_gat_go_state_dict

    class _GnnPF(torch.nn.Module):
        """A structural copy of GAT-GO's src/GnnPF.py, only what the state-dict
        remap needs to be tested against: the same GATConv/SAGPooling layer
        construction with the same shapes."""

        def __init__(self):
            super().__init__()
            embed_channels = 512
            hidden_channels = [512, 512, 1024, 1024]
            self.esm_in = nn.Sequential(nn.Conv1d(1280, embed_channels, kernel_size=1), nn.ReLU(inplace=True))
            self.seq_in = nn.Sequential(nn.Conv1d(25, embed_channels, kernel_size=1), nn.ReLU(inplace=True))
            self.pssm_in = nn.Sequential(nn.Conv1d(20, embed_channels, kernel_size=1), nn.ReLU(inplace=True))
            self.rep_dim = hidden_channels[-1] + 1280
            self.classifier = nn.Sequential(
                nn.Linear(self.rep_dim, 2752), nn.ReLU(inplace=True), nn.Linear(2752, 2752)
            )
            self.gc1 = GATConv(embed_channels, hidden_channels[0], heads=12, dropout=0.5, bias=False, concat=False)
            self.gc2 = GATConv(hidden_channels[0], hidden_channels[1], heads=12, dropout=0.5, bias=False, concat=False)
            self.gc3 = GATConv(hidden_channels[1], hidden_channels[2], heads=12, dropout=0.5, bias=False, concat=False)
            self.gc4 = GATConv(hidden_channels[2], hidden_channels[3], heads=12, dropout=0.5, bias=False, concat=False)
            self.gp1 = SAGPooling(in_channels=hidden_channels[0])
            self.gp2 = SAGPooling(in_channels=hidden_channels[1])
            self.gp3 = SAGPooling(in_channels=hidden_channels[2])
            self.gp4 = SAGPooling(in_channels=hidden_channels[3])

        def forward(self, esm_rep, seq, pssm, A, seq_embed, batch):
            esm = self.esm_in(esm_rep)
            seq = self.seq_in(seq)
            pssm = self.pssm_in(pssm)
            embed = (seq + pssm + esm).T.squeeze(2)
            out = self.gc1(embed, A).relu()
            out, A, _, batch, _, _ = self.gp1(out, A, None, batch)
            out = self.gc2(out, A).relu()
            out, A, _, batch, _, _ = self.gp2(out, A, None, batch)
            out = self.gc3(out, A).relu()
            out, A, _, batch, _, _ = self.gp3(out, A, None, batch)
            out = self.gc4(out, A).relu()
            out, A, _, batch, _, _ = self.gp4(out, A, None, batch)
            out = global_mean_pool(out, batch)
            return self.classifier(torch.cat([out, seq_embed], dim=1))

    def _to_legacy_naming(new_state: dict) -> dict:
        """The inverse of remap_legacy_gat_go_state_dict: build a synthetic
        pre-2.0-PyG checkpoint FROM a real model's current-naming weights, so
        a correct remap must recover them exactly."""
        legacy = {}
        for key, value in new_state.items():
            if ".gnn.lin_rel.weight" in key:
                legacy[key.replace(".gnn.lin_rel.weight", ".gnn.lin_l.weight")] = value
            elif ".gnn.lin_rel.bias" in key:
                legacy[key.replace(".gnn.lin_rel.bias", ".gnn.lin_l.bias")] = value
            elif ".gnn.lin_root.weight" in key:
                legacy[key.replace(".gnn.lin_root.weight", ".gnn.lin_r.weight")] = value
            elif key.endswith(".lin.weight"):
                base = key[: -len("lin.weight")]
                legacy[base + "lin_l.weight"] = value
                legacy[base + "lin_r.weight"] = value
            elif key.endswith(".att_src"):
                legacy[key[: -len("att_src")] + "att_l"] = value
            elif key.endswith(".att_dst"):
                legacy[key[: -len("att_dst")] + "att_r"] = value
            elif key.endswith(".select.weight"):
                pass
            else:
                legacy[key] = value
        return legacy


@unittest.skipUnless(_HAS_TORCH_GEOMETRIC, "torch_geometric is not installed in this environment")
class GatGoStateDictRemapTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _GnnPF()
        self.new_state = self.model.state_dict()
        self.legacy_state = _to_legacy_naming(self.new_state)

    def test_remap_recovers_exact_key_set(self):
        remapped = remap_legacy_gat_go_state_dict(self.legacy_state, self.model)
        self.assertEqual(set(remapped), set(self.new_state))

    def test_remap_recovers_bit_identical_weight_values(self):
        remapped = remap_legacy_gat_go_state_dict(self.legacy_state, self.model)
        for key, value in self.new_state.items():
            if key.endswith(".select.weight"):
                self.assertTrue(torch.equal(remapped[key], torch.ones(1, 1)))
            else:
                self.assertTrue(torch.equal(remapped[key], value), key)

    def test_remap_loads_with_strict_true(self):
        remapped = remap_legacy_gat_go_state_dict(self.legacy_state, self.model)
        fresh = _GnnPF()
        fresh.load_state_dict(remapped, strict=True)  # raises on any mismatch

    def test_forward_pass_is_bit_identical_after_remap(self):
        remapped = remap_legacy_gat_go_state_dict(self.legacy_state, self.model)
        fresh = _GnnPF()
        fresh.load_state_dict(remapped, strict=True)
        self.model.eval()
        fresh.eval()

        length = 40
        esm_rep = torch.randn(1, 1280, length)
        seq = torch.randn(1, 25, length)
        pssm = torch.randn(1, 20, length)
        seq_embed = torch.randn(1, 1280)
        src = torch.arange(length)
        dst = (src + 1) % length
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        batch = torch.zeros(length, dtype=torch.long)

        with torch.inference_mode():
            out_original = self.model(esm_rep=esm_rep, seq=seq, pssm=pssm, seq_embed=seq_embed, A=edge_index, batch=batch)
            out_remapped = fresh(esm_rep=esm_rep, seq=seq, pssm=pssm, seq_embed=seq_embed, A=edge_index, batch=batch)

        self.assertTrue(torch.equal(out_original, out_remapped))

    def test_remap_raises_when_a_target_key_is_left_unexplained(self):
        """If the model has a real parameter the renaming rules don't
        account for (e.g. the architecture changed again upstream), the
        function must raise rather than silently loading a partial model.
        select.weight is the one legitimate exception; deleting a GATConv
        key instead must not be silently tolerated."""
        broken = dict(self.legacy_state)
        del broken["gc1.lin_l.weight"]
        del broken["gc1.lin_r.weight"]
        with self.assertRaises(RuntimeError):
            remap_legacy_gat_go_state_dict(broken, self.model)


if __name__ == "__main__":
    unittest.main()
