import torch
from typing import Tuple


def _shift_1d_zero_pad(x: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Shift a 1D tensor with zero padding.

    Positive shift  -> delay x (shift right)
    Negative shift  -> advance x (shift left)

    Args:
        x: Tensor of shape [T]
        shift: Integer sample shift

    Returns:
        Shifted tensor of shape [T]
    """
    if x.ndim != 1:
        raise ValueError(f"_shift_1d_zero_pad expects 1D input, got shape {tuple(x.shape)}")

    T = x.shape[0]

    if shift == 0:
        return x.clone()

    y = torch.zeros_like(x)

    if shift > 0:
        if shift < T:
            y[shift:] = x[:-shift]
    else:
        s = -shift
        if s < T:
            y[:-s] = x[s:]

    return y


def _mae_for_lag(
    x: torch.Tensor,
    y: torch.Tensor,
    cv_db: torch.Tensor,
    lag: int,
) -> torch.Tensor:
    """
    Compute MAE for a candidate lag using the user's original lag convention:

    - lag < 0: CV is ahead, so delay CV
    - lag > 0: CV is behind, so delay x and y

    Args:
        x: Input audio, shape [T]
        y: Output audio, shape [T]
        cv_db: CV in dB gain, shape [T]
        lag: Integer lag of CV relative to audio

    Returns:
        Scalar tensor MAE
    """
    if lag < 0:
        cv_shift = _shift_1d_zero_pad(cv_db, -lag)  # delay CV by abs(lag)
        x_shift = x
        y_shift = y
    else:
        cv_shift = cv_db
        x_shift = _shift_1d_zero_pad(x, lag)
        y_shift = _shift_1d_zero_pad(y, lag)

    pred = x_shift * torch.pow(10.0, cv_shift / 20.0)
    return torch.mean(torch.abs(y_shift - pred))


@torch.no_grad()
def estimate_cv_lag(
    x: torch.Tensor,
    y: torch.Tensor,
    cv_db: torch.Tensor,
    min_lag: int = -10,
    max_lag: int = 10,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    Estimate integer CV lag relative to audio by minimizing MAE between
    y and x * db2amp(cv).

    Sign convention:
        negative lag -> CV is ahead of audio
        positive lag -> CV is behind audio

    Args:
        x: Input audio, shape [T] or [1, T]
        y: Output audio, shape [T] or [1, T]
        cv_db: Control signal in dB gain, shape [T] or [1, T]
        min_lag: Minimum lag to test
        max_lag: Maximum lag to test

    Returns:
        best_lag: Integer lag of CV relative to audio
        lags: Tensor of tested lags, shape [N]
        errors: Tensor of MAE values, shape [N]
    """
    x = x.squeeze()
    y = y.squeeze()
    cv_db = cv_db.squeeze()

    if x.ndim != 1 or y.ndim != 1 or cv_db.ndim != 1:
        raise ValueError(
            f"estimate_cv_lag expects 1D tensors after squeeze, got "
            f"x={tuple(x.shape)}, y={tuple(y.shape)}, cv_db={tuple(cv_db.shape)}"
        )

    if not (x.shape[0] == y.shape[0] == cv_db.shape[0]):
        raise ValueError(
            f"x, y, and cv_db must have the same length, got "
            f"{x.shape[0]}, {y.shape[0]}, {cv_db.shape[0]}"
        )

    lags = torch.arange(min_lag, max_lag + 1, device=x.device)
    errors = torch.empty_like(lags, dtype=x.dtype)

    for i, lag in enumerate(lags.tolist()):
        errors[i] = _mae_for_lag(x, y, cv_db, lag)

    best_idx = torch.argmin(errors)
    best_lag = int(lags[best_idx].item())

    return best_lag, lags, errors


@torch.no_grad()
def align_cv(
    x: torch.Tensor,
    y: torch.Tensor,
    cv_db: torch.Tensor,
    min_lag: int = -10,
    max_lag: int = 10,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """
    Align CV in place using the lag convention from the user's MATLAB code.

    Returns an aligned CV tensor such that:
        y ≈ x * db2amp(cv_aligned)

    Important:
        Returned lag is the lag of CV relative to audio.
        The shift applied to CV is the opposite sign.

    Sign convention:
        negative lag -> CV is ahead
        positive lag -> CV is behind

    Applied shift to CV:
        shift_cv = -best_lag

    Args:
        x: Input audio, shape [T] or [1, T]
        y: Output audio, shape [T] or [1, T]
        cv_db: CV in dB gain, shape [T] or [1, T]
        min_lag: Minimum lag to test
        max_lag: Maximum lag to test

    Returns:
        cv_aligned: Shifted CV, same shape as input cv_db
        best_lag: Estimated CV lag relative to audio
        lags: Tensor of tested lags
        errors: Tensor of corresponding MAEs
    """
    original_shape = cv_db.shape

    x_1d = x.squeeze()
    y_1d = y.squeeze()
    cv_1d = cv_db.squeeze()

    best_lag, lags, errors = estimate_cv_lag(
        x=x_1d,
        y=y_1d,
        cv_db=cv_1d,
        min_lag=min_lag,
        max_lag=max_lag,
    )

    shift_cv = -best_lag
    cv_aligned = _shift_1d_zero_pad(cv_1d, shift_cv)

    return cv_aligned.reshape(original_shape), best_lag, lags, errors