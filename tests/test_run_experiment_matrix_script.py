import subprocess
from pathlib import Path


SCRIPT = Path("scripts/run_experiment_matrix.sh")
PEGASUS_ARRAY_SCRIPT = Path("pegasus/run_experiment_matrix_all.sh")


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


def test_run_experiment_matrix_script_passes_hydra_overrides_through():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "E01",
            "data.unity.root_path=/tmp/skiing_data",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "Running E01" in result.stdout
    assert "data.unity.root_path=/tmp/skiing_data" in result.stdout



def test_pegasus_array_script_maps_array_index_to_single_experiment():
    text = PEGASUS_ARRAY_SCRIPT.read_text()

    assert "#PBS -t 0-11" in text
    assert "EXP_IDS=(E01 E02 E03 E04 E05 E06 E07 E08 E09 E10 E11 E12)" in text
    assert "TASK_ID=${PBS_SUBREQNO:-${PBS_ARRAY_INDEX:-${TASK_ID:-0}}}" in text
    assert "EXP_ID=${EXP_IDS[${TASK_ID}]}" in text
    assert "MODEL_BACKBONE=${MODEL_BACKBONES[${TASK_ID}]}" in text
    assert "HUMAN_3D_SOURCE=${HUMAN_3D_SOURCES[${TASK_ID}]}" in text
    assert "python -m pose2equip.main" in text
    assert "data.unity.root_path=${DATA_ROOT}" in text
    assert "data.index_mapping_path=${INDEX_MAPPING_PATH}" in text


def test_pegasus_array_script_dry_run_uses_one_experiment_from_index():
    result = subprocess.run(
        ["bash", str(PEGASUS_ARRAY_SCRIPT)],
        text=True,
        capture_output=True,
        env={
            "SKIP_CONDA": "1",
            "DRY_RUN": "1",
            "PBS_SUBREQNO": "1",
            **__import__("os").environ,
        },
    )

    assert result.returncode == 0
    assert "Task ID: 1" in result.stdout
    assert "Experiment: E02" in result.stdout
    assert "Backbone: stgcn" in result.stdout
    assert "data.human_3d_source=sam3d" in result.stdout
