"""Isaac Lab simulation of the flying gripper and the four articulated objects (Section 5.1).

Modules that need only torch, numpy or h5py (usable without Isaac Sim):

* ``expert``: scripted experts that generate the demonstrations;
* ``demos``: demonstration files (robomimic HDF5 layout);
* ``evaluation``: execution of the policy chunks and grasp-closure detection in the evaluation;
* ``records``: rollout records written by the evaluation and read by the scorer;
* ``reference``: reference feasible twist subspaces of the four objects.

The subpackages ``assets``, ``mdp``, ``sensors`` and ``tasks`` require a running Isaac Sim
application.
"""

OBJECTS = ("revolute", "cylindrical", "planar", "universal")
"""Names of the four simulated articulated objects, in the paper's order."""
