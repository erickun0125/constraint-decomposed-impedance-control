"""The recorded ensembles of examples/data and the example notebook."""

from pathlib import Path

import numpy as np
import pytest
import torch

from con_dec_imp.estimator import twist_matrix
from con_dec_imp.metrics import subspace_error
from con_dec_imp.policy.actions import GRIPPER_CLOSED_THRESHOLD, chunk_to_transforms
from con_dec_imp.subspace import estimate_subspace
from con_dec_imp_sim import OBJECTS
from con_dec_imp_sim.reference import EVALUATION_LENGTH, reference_basis

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
DATA = EXAMPLES / "data"


@pytest.mark.parametrize("object_name", OBJECTS)
def test_recorded_ensemble_gives_the_reference_dimension(object_name):
    data = np.load(DATA / f"{object_name}_ensemble.npz")
    chunks = torch.from_numpy(data["chunks"])
    assert chunks.shape == (128, 24, 10)
    estimate = estimate_subspace(twist_matrix(chunk_to_transforms(chunks), 0.1, chunks[..., 9] > GRIPPER_CLOSED_THRESHOLD))
    U_ref = reference_basis(object_name, torch.from_numpy(data["tcp_pose"]), torch.from_numpy(data["object_pose"]))
    assert int(estimate.dim) == U_ref.shape[-1]
    se = subspace_error(estimate.basis().double().numpy(), U_ref.double().numpy(), EVALUATION_LENGTH[object_name])
    assert se < 5.0


def test_example_notebook_runs():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    pytest.importorskip("matplotlib")
    notebook = nbformat.read(EXAMPLES / "estimate_from_ensemble.ipynb", as_version=4)
    client = nbclient.NotebookClient(
        notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(EXAMPLES)}}
    )
    client.execute()
