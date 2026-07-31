"""Verifies scripts/run_heal_arc.py's remap of HEAL's released checkpoint
(saved under torch_geometric < 2.0's GCNConv layout) onto the pinned
torch_geometric==2.5.3's layout.

The load-bearing check here is the **transpose**. Old GCNConv stored
`weight` as (in, out) and computed `x @ weight`; new PyG stores
`lin.weight` as (out, in) and computes `x @ lin.weight.T`. Every affected
HEAL layer is 512->512 square, so load_state_dict(strict=True) will happily
accept a non-transposed matrix and silently produce wrong scores. These
tests pin the orientation down empirically instead of trusting it.

Skipped where torch_geometric is not installed.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
    from torch import nn
    from torch_geometric.nn import GCNConv

    _HAS_TORCH_GEOMETRIC = True
except ImportError:
    _HAS_TORCH_GEOMETRIC = False


if _HAS_TORCH_GEOMETRIC:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_heal_arc import remap_legacy_heal_state_dict

    class _MiniHeal(nn.Module):
        """Structural stand-in for the parts of HEAL's CL_protNET the remap
        touches: GCNConv layers reached by nested attribute paths, matching
        the real key shapes (gcn.gcn.N, gcn.pool.pools.N.mab.layer_{k,v})."""

        def __init__(self, channels: int = 512):
            super().__init__()
            self.gcn = nn.Module()
            self.gcn.gcn = nn.ModuleList([GCNConv(channels, channels, bias=True) for _ in range(3)])
            pools = []
            for _ in range(2):
                pool = nn.Module()
                pool.mab = nn.Module()
                pool.mab.layer_k = GCNConv(channels, channels)
                pool.mab.layer_v = GCNConv(channels, channels)
                pool.mab.fc_q = nn.Linear(channels, channels)  # plain nn.Linear: must be untouched
                pools.append(pool)
            self.gcn.pool = nn.Module()
            self.gcn.pool.pools = nn.ModuleList(pools)


@unittest.skipUnless(_HAS_TORCH_GEOMETRIC, "torch_geometric is not installed in this environment")
class HealGcnConvOrientationTests(unittest.TestCase):
    """Pins down that old-GCNConv `weight` is the transpose of new
    `lin.weight`, by reproducing the documented pre-2.0 forward pass."""

    def test_old_forward_matches_new_conv_when_weight_is_transposed(self):
        torch.manual_seed(0)
        in_ch, out_ch, num_nodes = 6, 4, 12  # deliberately NON-square, so
        # a wrong orientation cannot hide behind a square matrix
        conv = GCNConv(in_ch, out_ch, bias=True)
        conv.eval()

        x = torch.randn(num_nodes, in_ch)
        src = torch.arange(num_nodes)
        dst = (src + 1) % num_nodes
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])

        with torch.inference_mode():
            out_new = conv(x, edge_index)

        # The legacy weight, as HEAL's checkpoint stores it: (in, out).
        legacy_weight = conv.lin.weight.detach().t().contiguous()
        self.assertEqual(tuple(legacy_weight.shape), (in_ch, out_ch))

        # Reproduce PyG <= 1.7.2's GCNConv.forward: x = x @ self.weight,
        # then the identical normalized propagation, then + bias.
        from torch_geometric.nn.conv.gcn_conv import gcn_norm

        with torch.inference_mode():
            x_legacy = x @ legacy_weight
            edge_index_n, edge_weight_n = gcn_norm(
                edge_index, None, num_nodes, improved=False, add_self_loops=True, dtype=x.dtype
            )
            out_legacy = conv.propagate(edge_index_n, x=x_legacy, edge_weight=edge_weight_n)
            out_legacy = out_legacy + conv.bias

        torch.testing.assert_close(out_new, out_legacy, rtol=1e-5, atol=1e-6)

    def test_non_transposed_weight_gives_a_different_answer(self):
        """Guards the guard: if transposing were a no-op, the test above
        would prove nothing. On a non-square layer the wrong orientation
        must not even be loadable."""
        conv = GCNConv(6, 4, bias=True)
        wrong = conv.lin.weight.detach().clone()  # already (out, in); "forgetting"
        self.assertNotEqual(tuple(wrong.shape), (6, 4))


@unittest.skipUnless(_HAS_TORCH_GEOMETRIC, "torch_geometric is not installed in this environment")
class HealStateDictRemapTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _MiniHeal()
        self.target = self.model.state_dict()
        # Synthesize a legacy checkpoint FROM the real model: every
        # `<path>.lin.weight` becomes `<path>.weight`, transposed.
        self.legacy = {}
        for key, value in self.target.items():
            if key.endswith(".lin.weight"):
                self.legacy[key[: -len(".lin.weight")] + ".weight"] = value.t().contiguous()
            else:
                self.legacy[key] = value

    def test_remap_reconciles_the_state_dict_exactly(self):
        remapped = remap_legacy_heal_state_dict(self.legacy, self.model)
        self.assertEqual(set(remapped), set(self.target))

    def test_remap_restores_original_orientation_and_values(self):
        remapped = remap_legacy_heal_state_dict(self.legacy, self.model)
        for key, value in self.target.items():
            self.assertEqual(tuple(remapped[key].shape), tuple(value.shape), key)
            self.assertTrue(torch.equal(remapped[key], value), key)

    def test_remap_loads_with_strict_true(self):
        remapped = remap_legacy_heal_state_dict(self.legacy, self.model)
        _MiniHeal().load_state_dict(remapped, strict=True)

    def test_plain_linear_layers_are_left_alone(self):
        """fc_q is a torch.nn.Linear, unchanged across PyG versions; the
        remap must not touch it."""
        remapped = remap_legacy_heal_state_dict(self.legacy, self.model)
        key = "gcn.pool.pools.0.mab.fc_q.weight"
        self.assertIn(key, remapped)
        self.assertTrue(torch.equal(remapped[key], self.target[key]))

    def test_remap_raises_on_unreconciled_keys(self):
        broken = dict(self.legacy)
        del broken["gcn.gcn.0.weight"]
        with self.assertRaises(RuntimeError):
            remap_legacy_heal_state_dict(broken, self.model)

    def test_remap_rejects_a_wrongly_shaped_legacy_weight(self):
        """A non-transpose-shaped legacy matrix must be refused, not
        reshaped or silently accepted."""
        model = _MiniHeal(channels=512)
        legacy = {}
        for key, value in model.state_dict().items():
            if key.endswith(".lin.weight"):
                legacy[key[: -len(".lin.weight")] + ".weight"] = torch.zeros(8, 3)
            else:
                legacy[key] = value
        with self.assertRaises(RuntimeError):
            remap_legacy_heal_state_dict(legacy, model)


if __name__ == "__main__":
    unittest.main()
