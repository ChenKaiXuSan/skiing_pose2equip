import unittest

from pathlib import Path

from omegaconf import OmegaConf

from pose2equip.dataloader.data_loader import UnityDataModule
from pose2equip.main import resolve_test_ckpt_path, resolve_trainer_device_kwargs


class TrainingFlowConfigTest(unittest.TestCase):
    def _minimal_cfg(self):
        return OmegaConf.create(
            {
                "data": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "img_size": 224,
                    "load_frames": False,
                    "load_2d_kpt": False,
                    "load_3d_kpt": True,
                    "time_window": 4,
                },
                "experiment": "unit",
                "train": {"gpu": 0},
                "trainer": {
                    "accelerator": "auto",
                    "devices": "auto",
                    "test_ckpt_path": "best",
                },
            }
        )

    def test_val_and_test_dataloaders_keep_partial_batches(self):
        data_module = UnityDataModule(
            self._minimal_cfg(),
            {"train": [object()], "val": [object()], "test": [object()]},
        )
        data_module.setup()

        self.assertTrue(data_module.train_dataloader().drop_last)
        self.assertFalse(data_module.val_dataloader().drop_last)
        self.assertFalse(data_module.test_dataloader().drop_last)


    def test_default_experiment_tag_encodes_matrix_axes(self):
        if not OmegaConf.has_resolver("now"):
            OmegaConf.register_new_resolver("now", lambda fmt: "20260620-153000")
        cfg = OmegaConf.load(Path("configs/pose2equip.yaml"))
        cfg.model.backbone = "stgcn_query"
        cfg.data.human_3d_source = "sam3d"
        cfg.data.time_window = 32
        cfg.data.load_frames = False
        cfg.pose2equip.hidden_dim = 128
        cfg.pose2equip.backbone_layers = 4
        cfg.pose2equip.decoder_layers = 2
        cfg.pose2equip.num_heads = 4
        OmegaConf.resolve(cfg)

        self.assertEqual(
            cfg.experiment_tag,
            "stgcn_query__pose-sam3d__h128__bl4__dl2__heads4__tw32__frames-False",
        )
        self.assertIn(cfg.experiment_tag, cfg.log_path)
        self.assertIn(str(cfg.run_id), cfg.log_path)

    def test_trainer_device_and_checkpoint_defaults_are_experiment_friendly(self):
        cfg = self._minimal_cfg()

        self.assertEqual(
            resolve_trainer_device_kwargs(cfg),
            {"accelerator": "auto", "devices": "auto"},
        )
        self.assertEqual(resolve_test_ckpt_path(cfg), "best")


if __name__ == "__main__":
    unittest.main()
