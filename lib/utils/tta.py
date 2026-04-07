import torch

def flip_left_right(kp3d: torch.Tensor) -> torch.Tensor:
    """
    Flippa le coordinate X dei marker e riordina le coppie left/right.
    Assumo che kp3d sia [B, F, J, 3].
    """
    flipped = kp3d.clone()
    # Inverti l’asse X
    flipped[..., 0] *= -1
    # Qui potresti ri-mappare gli indici dei joints:
    # es. swapped = flipped[..., indice_left], flipped[..., indice_right]
    # per ora presupponiamo simmetria o lasciare così.
    return flipped

def invert_flip_left_right(pred: torch.Tensor) -> torch.Tensor:
    """Ri-inverte la flip (di nuovo X * -1)."""
    inv = pred.clone()
    inv[..., 0] *= -1
    # Se avessi ri-scambiato left/right, qui fai il reverse
    return inv

def temporal_shift(kp3d: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Shifta temporalmente le sequenze di +shift o -shift frame, padding con zeri.
    """
    B, F, J, C = kp3d.shape
    out = torch.zeros_like(kp3d)
    if shift > 0:
        out[:, shift:] = kp3d[:, :-shift]
    elif shift < 0:
        out[:, :shift] = kp3d[:, -shift:]
    else:
        out = kp3d
    return out

def invert_temporal_shift(pred: torch.Tensor, shift: int) -> torch.Tensor:
    """Riporta la predizione alla sua timeline originale."""
    return temporal_shift(pred, -shift)