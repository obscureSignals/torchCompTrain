import torch
from torch.autograd import Function
from torchcomp import compexp_gain, avg, db2amp, amp2db


def arcsigmoid(x: torch.Tensor) -> torch.Tensor:
    return (x / (1 - x)).log()


def comp_gain(x, *args, **kwargs) -> torch.Tensor:
    return compexp_gain(x, *args, exp_ratio=0.9999, exp_thresh=-120, **kwargs)


def avg_rms(audio: torch.Tensor, avg_coef: torch.Tensor) -> torch.Tensor:
    return avg(audio.square().clamp_min(1e-8), avg_coef).sqrt()


def compressor(x, th, ratio, at, rt, make_up, delay: int = 0):

    if x.ndim == 1:
        peak = x.abs().unsqueeze(0) + 1e-8
    elif x.ndim == 2:
        peak = torch.max(x.abs(), dim=0, keepdim=True).values + 1e-8
    else:
        raise ValueError(f"Unexpected input shape: {x.shape}")

    gain = comp_gain(
        peak,
        comp_ratio=ratio,
        comp_thresh=th,
        at=at,
        rt=rt,
    )

    if delay > 0:
        gain = torch.cat([gain[:, delay:], gain.new_ones(gain.shape[0], delay)], dim=1)
    y = (x * gain
         * db2amp(torch.tensor(make_up, device=x.device, dtype=x.dtype)).broadcast_to(
                x.shape[0], 1)
         )

    return y, gain


def esr(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.flatten()
    target = target.flatten()
    diff = pred - target
    return (diff @ diff) / (target @ target)



