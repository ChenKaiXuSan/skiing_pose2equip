import json
import tempfile
import unittest
from pathlib import Path

from pose2equip.tools.cross_validation.index_io import (
    existing_fold_files,
    remove_legacy_aggregate_file,
    save_fold_files,
)


class DummySample:
    def __init__(self, sample_id: str):
        self.sample_id = sample_id

    def to_dict(self):
        return {"sample_id": self.sample_id}


class IndexIOTest(unittest.TestCase):
    def test_save_fold_files_writes_one_json_per_fold_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            folds = {
                0: {
                    "train": [DummySample("train-0")],
                    "val": [{"sample_id": "val-0"}],
                    "test": [DummySample("test-0")],
                    "ratio": "7/2/1",
                }
            }

            saved_files = save_fold_files(
                folds=folds,
                strategy="by_action",
                index_mapping_dir=tmp_path,
            )

            expected_file = (
                tmp_path / "camera_pairs_by_action_folds" / "fold_00.json"
            ).resolve()
            self.assertEqual(saved_files, [expected_file])

            with saved_files[0].open("r", encoding="utf-8") as f:
                fold_data = json.load(f)

            self.assertEqual(fold_data["train"], [{"sample_id": "train-0"}])
            self.assertEqual(fold_data["val"], [{"sample_id": "val-0"}])
            self.assertEqual(fold_data["test"], [{"sample_id": "test-0"}])
            self.assertEqual(fold_data["ratio"], "7/2/1")
            self.assertEqual(
                fold_data["_metadata"],
                {
                    "strategy": "by_action",
                    "fold_idx": 0,
                    "num_train": 1,
                    "num_val": 1,
                    "num_test": 1,
                    "total": 3,
                },
            )

    def test_existing_fold_files_reuses_only_json_fold_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fold_dir = tmp_path / "camera_pairs_by_camera_pair_folds"
            fold_dir.mkdir()
            (fold_dir / "fold_01.json").write_text("{}", encoding="utf-8")
            (fold_dir / "fold_00.json").write_text("{}", encoding="utf-8")
            (fold_dir / "notes.txt").write_text("ignore", encoding="utf-8")

            self.assertEqual(
                existing_fold_files(tmp_path, "by_camera_pair"),
                [
                    (fold_dir / "fold_00.json").resolve(),
                    (fold_dir / "fold_01.json").resolve(),
                ],
            )

    def test_remove_legacy_aggregate_file_deletes_old_single_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            legacy_file = tmp_path / "camera_pairs_by_action.json"
            legacy_file.write_text("{}", encoding="utf-8")

            self.assertEqual(
                remove_legacy_aggregate_file(tmp_path, "by_action"), legacy_file
            )
            self.assertFalse(legacy_file.exists())


if __name__ == "__main__":
    unittest.main()
