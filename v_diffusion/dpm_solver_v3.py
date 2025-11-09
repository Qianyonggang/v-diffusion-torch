from typing import Optional

import torch

from .diffusion import repeat_along_dim, pred_eps_from_v


def _expand_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    view_shape = (x.shape[0],) + (1,) * (ref.ndim - 1)
    return x.view(view_shape)


def _phi_1(h: torch.Tensor) -> torch.Tensor:
    eps = torch.full_like(h, 1e-5)
    small = torch.abs(h) < 1e-5
    numerator = torch.expm1(torch.where(small, eps, h))
    denom = torch.where(small, eps, h)
    return torch.where(small, 1 + denom / 2 + denom.pow(2) / 6, numerator / denom)


def _phi_2(h: torch.Tensor) -> torch.Tensor:
    eps = torch.full_like(h, 1e-5)
    small = torch.abs(h) < 1e-5
    denom = torch.where(small, eps, h).pow(2)
    numerator = torch.expm1(torch.where(small, eps, h)) - torch.where(small, eps, h)
    series = 0.5 + torch.where(small, eps, h) / 6
    return torch.where(small, series, numerator / denom)


def _phi_3(h: torch.Tensor) -> torch.Tensor:
    eps = torch.full_like(h, 1e-5)
    small = torch.abs(h) < 1e-5
    denom = torch.where(small, eps, h).pow(3)
    numerator = torch.expm1(torch.where(small, eps, h)) - torch.where(small, eps, h) - 0.5 * torch.where(small, eps, h).pow(2)
    series = 1 / 6 + torch.where(small, eps, h) / 24
    return torch.where(small, series, numerator / denom)


def _stack_condition(cond, repeats: int):
    if cond is None:
        return None
    if isinstance(cond, dict):
        stacked = {}
        for key, value in cond.items():
            if torch.is_tensor(value):
                stacked[key] = repeat_along_dim(value, repeats=repeats)
            else:
                stacked[key] = value
        if repeats == 2 and "cond_mask" in stacked and torch.is_tensor(stacked["cond_mask"]):
            stacked["cond_mask"][1::2] = 0
        return stacked
    if torch.is_tensor(cond):
        stacked = repeat_along_dim(cond, repeats=repeats)
        if repeats == 2:
            stacked[1::2] = 0
        return stacked
    return cond


@torch.inference_mode()
def dpm_solver_v3_sample(
        model,
        diffusion,
        shape,
        device,
        steps: int,
        condition=None,
        generator: Optional[torch.Generator] = None,
        noise: Optional[torch.Tensor] = None,
):
    assert diffusion.model_out_type == "v", "DPM-Solver-v3 implementation expects v-parameterized models."
    if noise is None:
        x = torch.randn(shape, device=device, generator=generator)
    else:
        x = noise.to(device)

    cond = diffusion._prepare_label(condition, device)
    use_cfg = diffusion.w_guide > 0 and cond is not None

    times = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float64)
    prev_eps = []
    prev_lambda = []

    for idx in range(steps):
        s = times[idx]
        t = times[idx + 1]
        s_batch = torch.full((shape[0],), s, device=device, dtype=torch.float64)
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.float64)

        logsnr_s = diffusion.t2logsnr(s_batch, x=x)[0]
        logsnr_t = diffusion.t2logsnr(t_batch, x=x)[0]
        alpha_s = torch.sigmoid(logsnr_s).sqrt()
        sigma_s = torch.sigmoid(-logsnr_s).sqrt()
        alpha_t = torch.sigmoid(logsnr_t).sqrt()
        sigma_t = torch.sigmoid(-logsnr_t).sqrt()

        lam_s = torch.log(alpha_s) - torch.log(sigma_s)
        lam_t = torch.log(alpha_t) - torch.log(sigma_t)

        if use_cfg:
            cond_in = _stack_condition(cond, repeats=2)
            x_in = repeat_along_dim(x, repeats=2)
            t_in = repeat_along_dim(s_batch, repeats=2)
            model_out = model(x_in, t_in, cond_in)
            guided, unguided = model_out.chunk(2)
            model_out = unguided + diffusion.w_guide * (guided - unguided)
        else:
            model_out = model(x, s_batch, cond)

        eps_s = pred_eps_from_v(x, model_out, logsnr_s)

        prev_eps.append(eps_s)
        prev_lambda.append(lam_s)
        h = lam_t - lam_s

        sigma_ratio = _expand_like(sigma_t / sigma_s, x)
        alpha_t_view = _expand_like(alpha_t, x)

        if len(prev_eps) == 1:
            phi1 = _expand_like(_phi_1(h), x)
            x = sigma_ratio * x - alpha_t_view * phi1 * eps_s
        elif len(prev_eps) == 2:
            h0 = prev_lambda[-1] - prev_lambda[-2]
            D1 = (prev_eps[-1] - prev_eps[-2]) / _expand_like(h0, x)
            phi1 = _expand_like(_phi_1(h), x)
            phi2 = _expand_like(_phi_2(h), x)
            x = sigma_ratio * x - alpha_t_view * (phi1 * eps_s - phi2 * D1)
        else:
            h0 = prev_lambda[-1] - prev_lambda[-2]
            h1 = prev_lambda[-2] - prev_lambda[-3]
            D1_0 = (prev_eps[-1] - prev_eps[-2]) / _expand_like(h0, x)
            D1_1 = (prev_eps[-2] - prev_eps[-3]) / _expand_like(h1, x)
            denom = _expand_like(0.5 * (h0 + h1), x)
            D2 = (D1_0 - D1_1) / denom
            phi1 = _expand_like(_phi_1(h), x)
            phi2 = _expand_like(_phi_2(h), x)
            phi3 = _expand_like(_phi_3(h), x)
            x = sigma_ratio * x - alpha_t_view * (phi1 * eps_s - phi2 * D1_0 + phi3 * D2)
            prev_eps.pop(0)
            prev_lambda.pop(0)

    return x

