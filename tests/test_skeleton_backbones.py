import unittest

import torch

from pose2equip.models.skeleton_backbones import (
    MLPBaselineNet,
    SkeletonTransformerNet,
    STGCNQueryNet,
    TCNBaselineNet,
    build_skeleton_backbone,
)


class SkeletonBackboneTest(unittest.TestCase):
    def _assert_sequence_output(self, model):
        x = torch.randn(2, 5, 15, 3)
        out = model(x)
        self.assertIn("object_3d", out)
        self.assertEqual(tuple(out["object_3d"].shape), (2, 5, 4, 2, 3))

    def _assert_single_frame_output(self, model):
        x = torch.randn(2, 15, 3)
        out = model(x)
        self.assertEqual(tuple(out["object_3d"].shape), (2, 4, 2, 3))

    def test_mlp_baseline_forward_shapes(self):
        model = MLPBaselineNet(num_joints=15, hidden_dim=32, num_equip_kpts=8)
        self._assert_sequence_output(model)
        self._assert_single_frame_output(model)

    def test_tcn_baseline_forward_shapes(self):
        model = TCNBaselineNet(num_joints=15, hidden_dim=32, num_equip_kpts=8, num_layers=2)
        self._assert_sequence_output(model)
        self._assert_single_frame_output(model)

    def test_skeleton_transformer_forward_shapes(self):
        model = SkeletonTransformerNet(
            num_joints=15,
            hidden_dim=32,
            num_equip_kpts=8,
            num_layers=1,
            num_heads=4,
        )
        self._assert_sequence_output(model)
        self._assert_single_frame_output(model)

    def test_stgcn_query_forward_shapes(self):
        model = STGCNQueryNet(
            num_joints=15,
            hidden_dim=32,
            num_equip_kpts=8,
            decoder_layers=1,
            num_heads=4,
        )
        self._assert_sequence_output(model)
        self._assert_single_frame_output(model)

    def test_factory_builds_all_comparison_backbones(self):
        for name in ["mlp", "tcn", "skeleton_transformer", "stgcn_query"]:
            with self.subTest(name=name):
                model = build_skeleton_backbone(
                    name=name,
                    num_joints=15,
                    hidden_dim=32,
                    num_equip_kpts=8,
                    num_layers=1,
                    num_heads=4,
                )
                self._assert_sequence_output(model)


if __name__ == "__main__":
    unittest.main()
