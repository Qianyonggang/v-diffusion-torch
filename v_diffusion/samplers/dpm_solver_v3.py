"""DPM-Solver-v3 sampler specialised for velocity-prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from ..diffusion import GaussianDiffusion, pred_eps_from_v, repeat_along_dim, slice_along_batch

Tensor = torch.Tensor


@dataclass
class DPMSolverState:
    """Cache for multi-step polynomial integration."""

    lambdas: List[Tensor]
    eps: List[Tensor]

    def push(self, lam: Tensor, eps_val: Tensor, max_order: int) -> None:
        self.lambdas.insert(0, lam.detach())
        self.eps.insert(0, eps_val.detach())
        if len(self.lambdas) > max_order:
            self.lambdas.pop()
            self.eps.pop()

    def as_order(self, order: int) -> Tuple[Sequence[Tensor], Sequence[Tensor]]:
        return self.lambdas[:order], self.eps[:order]


class DPMSolverV3Sampler:
    """Minimal DPM-Solver-v3 integrator supporting velocity-parameterised models."""

    def __init__(self, diffusion: GaussianDiffusion, order: int = 3):
        if order not in (1, 2, 3):
            raise ValueError("`order` must be 1, 2 or 3.")
        if diffusion.model_out_type != "v":
            raise ValueError("DPMSolverV3Sampler expects a velocity-prediction model (`model_out_type='v'`).")
        self.diffusion = diffusion
        self.max_order = order

    @staticmethod
    def _integrate_eps(
        lambdas: Sequence[Tensor],
        eps_values: Sequence[Tensor],
        target_lambda: Tensor,
    ) -> Tensor:
        if not lambdas:
            raise ValueError("Integration requires cached evaluations.")

        lambda0 = lambdas[0]
        eps0 = eps_values[0]
        h = target_lambda - lambda0
        integral = eps0 * h

        if len(lambdas) >= 2:
            lambda1 = lambdas[1]
            eps1 = eps_values[1]
            denom01 = lambda0 - lambda1
            slope01 = (eps0 - eps1) / denom01
            integral = integral + 0.5 * slope01 * h * h
        else:
            return integral

        if len(lambdas) >= 3:
            lambda2 = lambdas[2]
            eps2 = eps_values[2]
            denom12 = lambda1 - lambda2
            slope12 = (eps1 - eps2) / denom12
            denom02 = lambda0 - lambda2
            curvature = (slope01 - slope12) / denom02
            integral = integral + curvature * ((h * h * h) / 3.0 + 0.5 * (lambda0 - lambda1) * h * h)
        return integral

    def _eval_model(
        self,
        denoise_fn: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor],
        sample: Tensor,
        t_tensor: Tensor,
        label: Optional[Tensor],
        logsnr_value: float,
    ) -> Tensor:
        diffusion = self.diffusion
        device = sample.device
        dtype = sample.dtype

        if diffusion.w_guide > 0 and label is not None:
            x_in = repeat_along_dim(sample, repeats=2, dim=0)
            t_in = repeat_along_dim(t_tensor, repeats=2, dim=0)
            y_in = repeat_along_dim(label, repeats=2, dim=0)
            y_in[1::2] = 0
            logsnr_in = torch.full((x_in.shape[0],), logsnr_value, device=device, dtype=dtype)
            model_out = denoise_fn(x_in, t_in, y_in)
            eps_all = pred_eps_from_v(x_in, model_out, logsnr_in)
            eps_cond, eps_uncond = slice_along_batch(eps_all, 2)
            eps = eps_uncond + diffusion.w_guide * (eps_cond - eps_uncond)
        else:
            logsnr_tensor = torch.full((sample.shape[0],), logsnr_value, device=device, dtype=dtype)
            model_out = denoise_fn(sample, t_tensor, label)
            eps = pred_eps_from_v(sample, model_out, logsnr_tensor)

        return eps

    def sample(
        self,
        denoise_fn: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor],
        shape: Sequence[int],
        noise: Optional[Tensor] = None,
        label: Optional[Tensor] = None,
        device: str | torch.device = "cpu",
        seed: Optional[int] = None,
    ) -> Tensor:
        device = torch.device(device)
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        if noise is None:
            sample = torch.randn(shape, device=device, generator=generator)
        else:
            sample = noise.to(device)

        if label is not None:
            label = label.to(device)

        diffusion = self.diffusion
        steps = diffusion.sample_timesteps
        t_grid = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float64)
        logsnr_grid = diffusion.logsnr_fn(t_grid).to(torch.float64)
        lambda_grid = 0.5 * (torch.logsigmoid(logsnr_grid) - torch.logsigmoid(-logsnr_grid))

        state = DPMSolverState(lambdas=[], eps=[])
        batch_size = shape[0]
        t_buffer = torch.empty(batch_size, device=device, dtype=torch.float32)

        for step_idx in range(steps, 0, -1):
            cur_lambda = torch.as_tensor(float(lambda_grid[step_idx]), device=device, dtype=sample.dtype)
            next_lambda = torch.as_tensor(float(lambda_grid[step_idx - 1]), device=device, dtype=sample.dtype)
            cur_logsnr = float(logsnr_grid[step_idx])

            t_buffer.fill_(float(t_grid[step_idx]))
            eps = self._eval_model(denoise_fn, sample, t_buffer, label, cur_logsnr)

            state.push(cur_lambda, eps, self.max_order)
            order = min(self.max_order, len(state.lambdas))
            lambdas, eps_vals = state.as_order(order)
            delta = self._integrate_eps(lambdas, eps_vals, next_lambda)
            sample = sample - delta

        return sample
