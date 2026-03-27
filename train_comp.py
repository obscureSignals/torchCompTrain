import time
from functools import partial
from itertools import accumulate
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import pyloudnorm as pyln
import torch
import wandb
import yaml
from omegaconf import OmegaConf, DictConfig
from torch.nn import ParameterDict, Parameter
from torchaudio import load
from torchaudio.functional import lfilter
from torchcomp import ms2coef, coef2ms
from tqdm import tqdm

from cv_align import (align_cv)
from utils import (
    arcsigmoid,
    compressor,
    esr,
)


def compute_gr_l1_db(
        pred_gain: torch.Tensor,
        target_cv: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-8
    pred_gain_db = 20.0 * torch.log10(torch.clamp(pred_gain, min=eps))
    return torch.mean(torch.abs(pred_gain_db - target_cv))


def save_training_plot(
        loss_in: torch.Tensor,
        loss_tgt: torch.Tensor,
        pred_gain: torch.Tensor,
        target_cv: torch.Tensor,
        sr: int,
        step: int,
        out_dir: str = "temp_plots",
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    eps = 1e-8

    # --- convert predicted gain to dB to match target_cv domain ---
    pred_gr_db = 20.0 * torch.log10(torch.clamp(pred_gain, min=eps))

    # --- convert to numpy ---
    loss_in_np = loss_in[0].detach().cpu().squeeze().numpy()
    loss_tgt_np = loss_tgt[0].detach().cpu().squeeze().numpy()
    pred_gr_db_np = pred_gr_db[0].detach().cpu().squeeze().numpy()
    target_cv_np = target_cv[0].detach().cpu().squeeze().numpy()

    # --- first 5 seconds ---
    N = int(5 * sr)

    loss_in_np = loss_in_np[:N]
    loss_tgt_np = loss_tgt_np[:N]
    pred_gr_db_np = pred_gr_db_np[:N]
    target_cv_np = target_cv_np[:N]

    t = torch.arange(N).cpu().numpy() / sr

    # --- plot ---
    plt.figure(figsize=(12, 6))

    # Top: loss signals
    plt.subplot(2, 1, 1)
    plt.plot(t, loss_tgt_np, label="loss_tgt", linewidth=1.5)
    plt.plot(t, loss_in_np, label="loss_in", linewidth=1.5)
    plt.title("Loss Signals (what the loss sees)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    # Bottom: GR in dB (always show GT metric domain)
    plt.subplot(2, 1, 2)
    plt.plot(t, target_cv_np, label="target_GR_dB", linewidth=1.5)
    plt.plot(t, pred_gr_db_np, label="pred_GR_dB", linewidth=1.5, linestyle="--")
    plt.title("Gain Reduction (dB)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path = Path(out_dir) / f"step_{step:05d}.png"
    plt.savefig(path, dpi=120)
    plt.close()

    return str(path)


def get_loss_tensors(
        pred: torch.Tensor,
        pred_gain: torch.Tensor,
        target_audio: torch.Tensor,
        target_cv: torch.Tensor,
        loss_signal: str,
        loss_fn,
        prefilter,
):
    loss_name = loss_fn.__class__.__name__

    if loss_signal == "audio":
        if loss_name == "MSTELoss":
            return pred, target_audio  # don't prefilter for MSTE
        return prefilter(pred), prefilter(target_audio)

    if loss_signal == "cv":
        if loss_name == "MSTELoss":
            raise ValueError("MSTELoss is not supported for loss_signal='cv'")
        eps = 1e-8
        pred_gain_db = 20.0 * torch.log10(torch.clamp(pred_gain, min=eps))
        return pred_gain_db, target_cv

    raise ValueError(f"Unsupported loss_signal: {loss_signal}")


@hydra.main(config_path="cfg", config_name="config")
def train(cfg: DictConfig):
    tr_cfg = cfg.data.train

    train_input, sr = load(tr_cfg.input)
    train_target_audio, sr2 = load(tr_cfg.target_audio)
    train_target_cv, sr3 = load(tr_cfg.target_cv)

    assert train_input.shape == train_target_audio.shape, "Input and target shapes must match"
    assert train_input.shape[1] == train_target_cv.shape[1], "Input and target shapes must match"

    assert sr == sr2 == sr3

    x_left = train_input[0]
    y_left = train_target_audio[0]

    # If CV is mono stored as [1, T], this is fine.
    # If it somehow has more than one channel, pick the one you want explicitly.
    cv = train_target_cv[0]

    cv_aligned, cv_lag, lags, errors = align_cv(
        x_left,
        y_left,
        cv,
        min_lag=-10,
        max_lag=10,
    )

    print(f"Estimated CV lag relative to audio: {cv_lag} samples")

    # overwrite / keep aligned version
    train_target_cv = cv_aligned.unsqueeze(0)

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(train_input.numpy().T)
    print(f"Train input loudness: {loudness}")

    if "test" in cfg.data:
        test_cfg = cfg.data.test
        test_input, sr3 = load(test_cfg.input)
        assert sr == sr3, "Sample rates must match"
        test_target_audio, sr4 = load(test_cfg.target_audio)
        assert sr == sr4, "Sample rates must match"
        test_target_cv, sr5 = load(test_cfg.target_cv)

        x_left_test = test_input[0]
        y_left_test = test_target_audio[0]

        cv_test = test_target_cv[0]

        cv_aligned_test, cv_lag_test, lags_test, errors_test = align_cv(
            x_left_test,
            y_left_test,
            cv_test,
            min_lag=-10,
            max_lag=10,
        )

        print(f"Estimated test CV lag relative to audio: {cv_lag_test} samples")

        # overwrite / keep aligned version
        test_target_cv = cv_aligned_test.unsqueeze(0)

        assert sr == sr5, "Sample rates must match"
        assert (
                test_input.shape == test_target_audio.shape
        ), "Input and target shapes must match"
        assert test_input.shape[1] == test_target_cv.shape[1], "Input and target shapes must match"

        loudness = meter.integrated_loudness(test_input.numpy().T)
        print(f"Test input loudness: {loudness}")
    else:
        test_input = test_target_audio = None

    m2c = partial(ms2coef, sr=sr)
    c2m = partial(coef2ms, sr=sr)

    config: Any = OmegaConf.to_container(cfg)
    wandb_init = config.pop("wandb_init", {})
    run: Any = wandb.init(config=config, **wandb_init)

    # initialize model
    inits = cfg.compressor.inits
    init_th = torch.tensor(inits.threshold, dtype=torch.float32)
    init_ratio = torch.tensor(inits.ratio, dtype=torch.float32)
    init_at = m2c(torch.tensor(inits.attack_ms, dtype=torch.float32))
    init_make_up_gain = torch.tensor(inits.make_up_gain, dtype=torch.float32)

    param_th = Parameter(init_th)
    param_ratio_logit = Parameter(torch.log(init_ratio - 1))
    param_at_logit = Parameter(arcsigmoid(init_at))
    param_make_up_gain = Parameter(init_make_up_gain)

    param_ratio = lambda: param_ratio_logit.exp() + 1
    param_at = lambda: param_at_logit.sigmoid()

    params = ParameterDict(
        {
            "threshold": param_th,
            "ratio_logit": param_ratio_logit,
            "at_logit": param_at_logit,
            "make_up_gain": param_make_up_gain,
        }
    )

    if cfg.compressor.init_config:
        init_cfg = yaml.safe_load(open(cfg.compressor.init_config))
        init_cfg.pop("formated_params", None)
        init_params = {k: Parameter(torch.tensor(v)) for k, v in init_cfg.items()}
        params.load_state_dict(init_params, strict=False)

    comp_delay = cfg.compressor.delay
    init_rt = m2c(torch.tensor(inits.release_ms, dtype=torch.float32))
    param_rt_logit = Parameter(arcsigmoid(init_rt))
    params["rt_logit"] = param_rt_logit
    param_rt = lambda: param_rt_logit.sigmoid()
    infer = lambda x: compressor(
        x,
        th=param_th,
        ratio=param_ratio(),
        at=param_at(),
        rt=param_rt(),
        make_up=param_make_up_gain,
        delay=comp_delay,
    )

    # initialize optimiser
    optimiser = hydra.utils.instantiate(cfg.optimiser, params.values())

    # initialize scheduler
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimiser)

    # initialize loss function
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)

    prefilter = partial(
        lfilter,
        a_coeffs=torch.tensor(
            [1, -0.995], dtype=torch.float32, device=train_input.device
        ),
        b_coeffs=torch.tensor([1, -1], dtype=torch.float32, device=train_input.device),
        clamp=False,
    )

    def dump_params(loss=None):
        # convert params to dict for yaml
        out = {k: v.item() for k, v in params.items()}

        formated = {
            "attack_ms": c2m(param_at()).item(),
            "ratio": param_ratio().item(),
            "release_ms": c2m(param_rt()).item()
        }

        out["formated_params"] = formated
        if loss is not None:
            out["loss"] = loss
        return out

    final_params = dump_params()

    t0 = time.time()
    history = []

    with tqdm(range(cfg.epochs)) as pbar:

        def step(lowest_loss: torch.Tensor, global_step: int):
            optimiser.zero_grad()
            pred, pred_gain = infer(train_input)

            if torch.isnan(pred).any():
                raise ValueError("NaN in prediction")

            if torch.isinf(pred).any():
                raise ValueError("Inf in prediction")

            if torch.isnan(pred_gain).any():
                raise ValueError("NaN in prediction")

            if torch.isinf(pred_gain).any():
                raise ValueError("Inf in prediction")

            loss_in, loss_tgt = get_loss_tensors(
                pred=pred,
                pred_gain=pred_gain,
                target_audio=train_target_audio,
                target_cv=train_target_cv,
                loss_signal=cfg.loss_signal,
                loss_fn=loss_fn,
                prefilter=prefilter,
            )

            if global_step % 100 == 0:
                save_training_plot(
                    loss_in=loss_in,
                    loss_tgt=loss_tgt,
                    pred_gain=pred_gain,
                    target_cv=train_target_cv,
                    sr=sr,
                    step=global_step,
                    out_dir="temp_plots",
                )

            loss = loss_fn(loss_in, loss_tgt)

            with torch.no_grad():
                gr_l1_db = compute_gr_l1_db(pred_gain, train_target_cv).item()
                esr_val = esr(prefilter(pred), prefilter(train_target_audio)).item()

            if lowest_loss > loss:
                lowest_loss = loss.item()
                final_params.update(dump_params(lowest_loss))
                final_params["esr"] = esr_val
                final_params["gr_l1_db"] = gr_l1_db

            loss.backward()
            optimiser.step()
            scheduler.step(loss.item())

            elapsed_sec = time.time() - t0

            history.append(
                {
                    "step": int(global_step),
                    "elapsed_sec": float(elapsed_sec),
                    "native_loss": float(loss.item()),
                    "gr_l1_db": float(gr_l1_db),
                    "esr": float(esr_val),
                }
            )

            pbar_dict = {
                "loss": loss.item(),
                "lowest_loss": lowest_loss,
                "gr_l1_db": gr_l1_db,
                "elapsed_sec": elapsed_sec,
                "ratio": param_ratio().item(),
                "th": param_th.item(),
                "attack_ms": c2m(param_at()).item(),
                "make_up": param_make_up_gain.item(),
                "lr": optimiser.param_groups[0]["lr"],
                "esr": esr_val,
                "release_ms": c2m(param_rt()).item(),
            }

            pbar.set_postfix(**pbar_dict)

            wandb.log(pbar_dict, step=global_step)

            return lowest_loss

        try:
            losses = list(accumulate(pbar, step, initial=torch.inf))
        except KeyboardInterrupt:
            print("Training interrupted")

    if test_input is not None:
        # load best model
        params.load_state_dict(
            {
                k: torch.tensor(v)
                for k, v in final_params.items()
                if k != "formated_params"
            },
            strict=False,
        )

        pred, pred_gain = infer(test_input)

        loss_in, loss_tgt = get_loss_tensors(
            pred=pred,
            pred_gain=pred_gain,
            target_audio=test_target_audio,
            target_cv=test_target_cv,
            loss_signal=cfg.loss_signal,
            loss_fn=loss_fn,
            prefilter=prefilter,
        )

        test_loss = loss_fn(loss_in, loss_tgt)
        test_gr_l1_db = compute_gr_l1_db(pred_gain, test_target_cv).item()

        esr_val = esr(pred, test_target_audio).item()
        print(f"Test loss: {test_loss}")
        print(f"Test GR L1 (dB): {test_gr_l1_db}")
        print(f"Test ESR: {esr_val}")
        wandb.log(
            {
                "test_loss": test_loss,
                "test_gr_l1_db": test_gr_l1_db,
                "test_esr": esr_val,
            }
        )

    print("Training complete. Saving model...")

    final_params["elapsed_sec"] = time.time() - t0
    final_params["sec_per_step"] = final_params["elapsed_sec"] / max(1, len(history))

    if cfg.ckpt_path:
        ckpt_path = Path(cfg.ckpt_path)
        yaml.dump(final_params, open(ckpt_path, "w"), sort_keys=True)
        wandb.log_artifact(str(ckpt_path), type="parameters")

        hist_path = ckpt_path.with_suffix(".history.csv")
        import csv
        with open(hist_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "elapsed_sec", "native_loss", "gr_l1_db", "esr"]
            )
            writer.writeheader()
            writer.writerows(history)

    summary = {
        "loss": final_params["loss"],
        "gr_l1_db": final_params["gr_l1_db"],
        "ratio": final_params["formated_params"]["ratio"],
        "th": final_params["threshold"],
        "attack_ms": final_params["formated_params"]["attack_ms"],
        "make_up": final_params["make_up_gain"],
        "esr": final_params["esr"],
        "elapsed_sec": final_params["elapsed_sec"],
        "sec_per_step": final_params["sec_per_step"]
    }

    run.summary.update(summary)

    print("Final parameters:")
    print(final_params)

    return


if __name__ == "__main__":
    train()
