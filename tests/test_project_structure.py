from pathlib import Path


def test_shared_data_index_api_is_available():
    from pose2equip.data_index import (
        detect_available_folds,
        load_fold_dataset_idx_from_fold_json,
        remap_unity_dataset_paths,
    )

    assert callable(detect_available_folds)
    assert callable(load_fold_dataset_idx_from_fold_json)
    assert callable(remap_unity_dataset_paths)


def test_cross_validation_lives_under_tools_package():
    import pose2equip.tools.cross_validation.main as cv_main

    assert callable(cv_main.generate_index_files)


def test_old_package_test_directory_was_moved():
    assert Path("tests/test_equipment_geometry.py").exists()
    assert not Path("pose2equip/test/test_equipment_geometry.py").exists()


def test_equipment_trainers_share_base_class():
    from pose2equip.trainer.base_equipment_trainer import BaseEquipmentTrainer
    from pose2equip.trainer.train_pose2equip import Pose2EquipTrainer
    from pose2equip.trainer.train_stgcn import Pose2Equip_STGCN_Trainer

    assert issubclass(Pose2EquipTrainer, BaseEquipmentTrainer)
    assert issubclass(Pose2Equip_STGCN_Trainer, BaseEquipmentTrainer)


def test_model_components_are_split_into_focused_modules():
    from pose2equip.models.equipment_decoder import EquipmentQueryDecoder
    from pose2equip.models.image_encoder import DinoPatchEncoder
    from pose2equip.models.pose2equip_net import Pose2EquipNet
    from pose2equip.models.pose_encoder import PoseEncoder
    from pose2equip.models.stgcn_baseline import STGCNBaselineNet

    assert DinoPatchEncoder.__name__ == "DinoPatchEncoder"
    assert PoseEncoder.__name__ == "PoseEncoder"
    assert EquipmentQueryDecoder.__name__ == "EquipmentQueryDecoder"
    assert Pose2EquipNet.__name__ == "Pose2EquipNet"
    assert STGCNBaselineNet.__name__ == "STGCNBaselineNet"
