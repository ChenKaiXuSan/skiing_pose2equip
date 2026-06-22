import builtins

import pytest

from pose2equip.dataloader.unity_dataset_single_view import LabeledUnityDataset


def test_load_cv2_reports_frame_loading_dependency(monkeypatch):
    original_import = builtins.__import__

    def import_without_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("missing cv2")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_cv2)

    with pytest.raises(ImportError, match="cv2 is required only when loading RGB frames"):
        LabeledUnityDataset._load_cv2()
