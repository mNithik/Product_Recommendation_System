"""Tests for lightweight model artifact persistence."""

from pathlib import Path

from src.utils.model_artifacts import artifact_path, load_model_artifact, save_model_artifact


class _ToyModel:
    def __init__(self, value: int):
        self.value = value


def test_model_artifact_round_trip():
    output_dir = Path("tests_artifacts_tmp")
    model = _ToyModel(value=7)

    save_model_artifact(
        model,
        output_dir=str(output_dir),
        experiment_name="exp1",
        model_name="Toy Model",
    )

    loaded = load_model_artifact(
        output_dir=str(output_dir),
        experiment_name="exp1",
        model_name="Toy Model",
    )

    assert loaded.value == 7
    assert artifact_path(str(output_dir), "exp1", "Toy Model").exists()
