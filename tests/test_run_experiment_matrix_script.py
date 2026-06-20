import subprocess
from pathlib import Path


SCRIPT = Path("scripts/run_experiment_matrix.sh")


def test_run_experiment_matrix_script_help_lists_experiments():
    result = subprocess.run(["bash", str(SCRIPT), "--help"], text=True, capture_output=True)

    assert result.returncode == 0
    assert "E01" in result.stdout
    assert "E12" in result.stdout
    assert "--dry-run" in result.stdout


def test_run_experiment_matrix_script_dry_run_recommended():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "recommended"],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "model.backbone=stgcn" in result.stdout
    assert "model.backbone=skeleton_transformer" in result.stdout
    assert "data.human_3d_source=unity" in result.stdout
    assert "data.human_3d_source=sam3d" in result.stdout
    assert "trainer.devices=1" in result.stdout


def test_run_experiment_matrix_script_dry_run_single_experiment():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "E02"],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "Running E02" in result.stdout
    assert "model.backbone=stgcn" in result.stdout
    assert "data.human_3d_source=sam3d" in result.stdout
    assert "data.sam3d_human_key=character_cam1" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=0" in result.stdout


def test_run_experiment_matrix_script_assigns_recommended_jobs_across_two_gpus():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "recommended"],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "CUDA_VISIBLE_DEVICES=0" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=1" in result.stdout
    assert "Running E01" in result.stdout
    assert "Running E02" in result.stdout
    assert "Running E07" in result.stdout
    assert "Running E08" in result.stdout


def test_run_experiment_matrix_script_respects_gpus_env_list():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "E01", "E02", "E03"],
        text=True,
        capture_output=True,
        env={"GPUS": "1,0", **__import__("os").environ},
    )

    assert result.returncode == 0
    assert "Running E01" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=1" in result.stdout
    assert "Running E02" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=0" in result.stdout
