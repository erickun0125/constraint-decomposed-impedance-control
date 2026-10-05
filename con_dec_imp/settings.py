"""Hyperparameters of the simulation experiments (paper, Appendix H).

The values follow the common configuration and the numerical gain values of the
paper's Appendix H. The estimator, the gains, the policy sampler, the simulation
environments and the evaluation use them as defaults. The trajectory horizon H comes from
the policy checkpoint, and demonstration collection reads its settings from
``configs/collect/<object>.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EstimationSettings:
    """Hyperparameters of the constraint estimation (Algorithm 1, lines 1-9)."""

    num_samples: int = 128
    """Ensemble size N."""

    dt: float = 0.1
    """Waypoint interval used for the finite-difference twists [s]."""

    v_abs: float = 0.005
    """Absolute dimension-selection threshold [m/s] (5 mm/s)."""

    kappa_rel: float = 4.0
    """Relative dimension-selection threshold (ratio to sqrt(lambda_6))."""

    alpha_min: float = 0.03
    """Lower clip of the characteristic length alpha [m]."""

    alpha_max: float = 0.5
    """Upper clip of the characteristic length alpha [m]."""

    denoising_steps: int = 10
    """DDIM denoising steps of the policy sampler (executed reference and ensemble)."""


@dataclass(frozen=True)
class GainSettings:
    """Impedance gains of the flying gripper (Appendices G and H).

    The inertia metric is the block-isotropic surrogate
    ``blkdiag(I_bar * I3, m_bar * I3)`` of the gripper's spatial inertia about the
    control point, used in its normalized form ``blkdiag(l^2 * I3, I3)`` with
    ``l = sqrt(I_bar / m_bar)``: the overall scale ``m_bar`` is absorbed into the
    scalar gains (Appendix A.2, scale normalization). The damping ratio ``zeta`` is
    defined in the same normalized model, ``d = 2 * zeta * sqrt(k)``; the value below
    equals ``sqrt(m_bar)``, i.e. critical damping of the physical gripper mass.
    """

    mass: float = 0.586
    """Translational inertia scale m_bar [kg]."""

    rotational_inertia: float = 0.006220
    """Rotational inertia scale I_bar [kg m^2]."""

    k0: float = 250.0
    """Reference (isotropic) stiffness [1/s^2]."""

    reduction_ratio: float = 0.1
    """Complementary reduction ratio r for Ours and Oracle (Iso uses r = 1)."""

    zeta: float = 0.7655
    """Damping ratio of the normalized design model (reported as 0.766 in the paper)."""


@dataclass(frozen=True)
class ControlSettings:
    """Timing of the policy and the impedance controller in simulation."""

    control_rate: float = 150.0
    """Impedance controller rate [Hz]."""

    policy_rate: float = 10.0
    """Policy waypoint rate [Hz]."""

    executed_waypoints: int = 16
    """Waypoints executed from each predicted chunk before replanning."""

    gain_transition_steps: int = 50
    """Control steps of the linear blend from the isotropic to the decomposed gains."""


ESTIMATION = EstimationSettings()
GAINS = GainSettings()
CONTROL = ControlSettings()
