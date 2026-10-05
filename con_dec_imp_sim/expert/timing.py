"""Timing profiles of the scripted expert (paper, Appendix H, "Simulation demonstration data").

The expert randomizes *when* the operated coordinates move, never *where* they go:

* a single-coordinate object (revolute) follows one S-curve progress profile with zero
  velocity at both ends, :func:`s_curve`;
* each coordinate ``j`` of a multi-coordinate object (cylindrical, planar, universal)
  advances with its own velocity shape ``v_j(tau) = sin^2(pi tau) N(tau; phi_j, sigma_j)``
  with random onset ``phi_j ~ U[0, 1]`` and spread ``sigma_j ~ U[0.1, 0.3]``. The resulting
  curve ``s(tau)`` in the normalized progress space ``[0, 1]^n`` is traversed at constant
  speed, so a curve of arc length ``L`` takes ``T L / sqrt(n)`` seconds (``sqrt(n)`` is the
  length of the straight, fully synchronous curve), :class:`EqualProgressTiming`.

All tensors are float32 except the precomputation of the progress curves (float64).
"""

from __future__ import annotations

import math

import torch


def s_curve(tau: torch.Tensor, blend: float = 0.1) -> torch.Tensor:
    """Progress in ``[0, 1]`` of the cosine-blended trapezoidal profile at normalized time ``tau``.

    The velocity ramps up with a half cosine over the first ``blend`` of the time, cruises,
    and ramps down symmetrically, so the profile starts and ends at rest.
    """
    tau = tau.clamp(0.0, 1.0)
    v_max = 1.0 / (1.0 - blend)
    c = blend / (2.0 * math.pi)
    # The sines are evaluated in float64 and then rounded to the dtype of tau.
    ramp_up = v_max * (tau / 2.0 - (c * torch.sin((math.pi * tau / blend).double())).to(tau.dtype))
    cruise = blend / (2.0 * (1.0 - blend)) + v_max * (tau - blend)
    dt = tau - (1.0 - blend)
    ramp_down = (2.0 - 3.0 * blend) / (2.0 * (1.0 - blend)) + v_max * (
        dt / 2.0 + (c * torch.sin((math.pi * dt / blend).double())).to(tau.dtype)
    )
    s = torch.where(tau < blend, ramp_up, torch.where(tau <= 1.0 - blend, cruise, ramp_down))
    return torch.where(tau >= 1.0, torch.ones_like(s), torch.where(tau <= 0.0, torch.zeros_like(s), s))


def _interpolate(query: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of rows ``ys`` over the monotone rows ``xs``.

    ``query`` is ``(N,)``; ``xs`` is ``(G,)`` or ``(N, G)``; ``ys`` is ``(N, G)`` or ``(N, G, D)``.
    Returns ``(N,)`` or ``(N, D)``.
    """
    num_grid = xs.shape[-1]
    sorted_xs = xs if xs.dim() == 2 else xs.expand(query.shape[0], num_grid).contiguous()
    idx = torch.searchsorted(sorted_xs, query[:, None].contiguous()).clamp(1, num_grid - 1)
    x_lo, x_hi = sorted_xs.gather(1, idx - 1)[:, 0], sorted_xs.gather(1, idx)[:, 0]
    weight = (query - x_lo) / (x_hi - x_lo).clamp(min=1e-12)
    if ys.dim() == 2:
        y_lo, y_hi = ys.gather(1, idx - 1)[:, 0], ys.gather(1, idx)[:, 0]
        return (y_lo + weight * (y_hi - y_lo)).clamp(0.0, 1.0)
    rows = torch.arange(query.shape[0], device=query.device)
    y_lo, y_hi = ys[rows, idx[:, 0] - 1], ys[rows, idx[:, 0]]
    return (y_lo + weight[:, None] * (y_hi - y_lo)).clamp(0.0, 1.0)


class EqualProgressTiming:
    """Per-coordinate progress of a multi-coordinate object, one random draw per environment.

    Args:
        num_envs: Number of parallel environments.
        num_dofs: Number of operated coordinates ``n``.
        duration: Duration ``T`` of the fully synchronous motion [s].
        onset_range: Range of the onset ``phi_j`` (normalized time).
        spread_range: Range of the spread ``sigma_j`` (normalized time).
        grid_size: Resolution of the precomputed progress curves.
        device: Device of the progress tensors.
    """

    def __init__(
        self,
        num_envs: int,
        num_dofs: int,
        duration: float,
        onset_range: tuple[float, float] = (0.0, 1.0),
        spread_range: tuple[float, float] = (0.1, 0.3),
        grid_size: int = 200,
        device: torch.device | str = "cpu",
    ):
        self.num_dofs = num_dofs
        self.duration = duration
        self.onset_range = onset_range
        self.spread_range = spread_range
        self.device = torch.device(device)
        tau = torch.linspace(0.0, 1.0, grid_size, dtype=torch.float64, device=self.device)
        self._tau64 = tau
        self.grid = tau.float()
        self.onset = torch.zeros(num_envs, num_dofs, device=self.device)
        self.spread = torch.ones(num_envs, num_dofs, device=self.device)
        self.curve = self.grid[None, :, None].expand(num_envs, grid_size, num_dofs).clone()
        self.arc = self.grid.expand(num_envs, grid_size).clone()
        self.length = torch.full((num_envs,), math.sqrt(num_dofs), device=self.device)

    def resample(self, generator: torch.Generator | None = None) -> None:
        """Draw new onsets and spreads for every environment (CPU generator, environment order)."""
        for i in range(self.onset.shape[0]):
            onset = torch.empty(self.num_dofs).uniform_(*self.onset_range, generator=generator)
            spread = torch.empty(self.num_dofs).uniform_(*self.spread_range, generator=generator)
            self.onset[i], self.spread[i] = onset.to(self.device), spread.to(self.device)
        self.curve, self.arc, self.length = self._precompute(self.onset, self.spread)

    def _precompute(self, onset: torch.Tensor, spread: torch.Tensor):
        """Progress curves ``s(tau)`` (trapezoidal integral of ``v_j``) and their arc length."""
        tau = self._tau64
        window = torch.sin(math.pi * tau).square()[None, :, None]
        z = (tau[None, :, None] - onset.double()[:, None, :]) / spread.double()[:, None, :]
        velocity = window * torch.exp(-0.5 * z.square())
        step = 1.0 / (tau.shape[0] - 1)
        cumulative = torch.zeros_like(velocity)
        cumulative[:, 1:] = torch.cumsum(0.5 * (velocity[:, :-1] + velocity[:, 1:]) * step, dim=1)
        curve = cumulative / cumulative[:, -1:].clamp(min=1e-12)
        curve[:, 0], curve[:, -1] = 0.0, 1.0
        curve = curve.float()

        segments = torch.linalg.norm(curve.double()[:, 1:] - curve.double()[:, :-1], dim=-1)
        arc = torch.zeros(curve.shape[:2], dtype=torch.float64, device=curve.device)
        arc[:, 1:] = torch.cumsum(segments, dim=1)
        length = arc[:, -1].clamp(min=1e-12)
        arc = arc / length[:, None]
        arc[:, 0], arc[:, -1] = 0.0, 1.0
        return curve, arc.float(), length.float()

    @property
    def total_duration(self) -> torch.Tensor:
        """Duration ``T L / sqrt(n)`` of each environment's motion [s], ``(N,)`` float64."""
        return self.duration * self.length.double() / math.sqrt(self.num_dofs)

    def progress(self, t: torch.Tensor) -> torch.Tensor:
        """Normalized progress ``(N, n)`` of every coordinate at time ``t`` ``(N,)`` [s]."""
        duration = self.total_duration.clamp(min=1e-12).to(t.dtype)
        arc = (t / duration).clamp(0.0, 1.0)
        tau = _interpolate(arc, self.arc, self.grid.expand_as(self.arc))
        return _interpolate(tau, self.grid, self.curve)
