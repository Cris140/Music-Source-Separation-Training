# utils.py
# coding: utf-8
__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"

import numpy as np
import torch
import torch.nn as nn
import yaml
import librosa
import torch.nn.functional as F
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any, Union


# ----------------------------------------------------------------------------- #
#                        CONFIG & MODEL LOADING                                 #
# ----------------------------------------------------------------------------- #
def load_config(model_type: str, config_path: str) -> Union[ConfigDict, OmegaConf]:
    with open(config_path) as f:
        if model_type == "htdemucs":
            return OmegaConf.load(config_path)
        return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))


def get_model_from_config(model_type: str, config_path: str):
    cfg = load_config(model_type, config_path)
    if model_type == "mdx23c":
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net

        return TFC_TDF_net(cfg), cfg
    if model_type == "htdemucs":
        from models.demucs4ht import get_model

        return get_model(cfg), cfg
    if model_type == "segm_models":
        from models.segm_models import Segm_Models_Net

        return Segm_Models_Net(cfg), cfg
    if model_type == "torchseg":
        from models.torchseg_models import Torchseg_Net

        return Torchseg_Net(cfg), cfg
    if model_type == "mel_band_roformer":
        from models.bs_roformer import MelBandRoformer

        return MelBandRoformer(**dict(cfg.model)), cfg
    if model_type == "bs_roformer":
        from models.bs_roformer import BSRoformer

        return BSRoformer(**dict(cfg.model)), cfg
    if model_type == "swin_upernet":
        from models.upernet_swin_transformers import Swin_UperNet_Model

        return Swin_UperNet_Model(cfg), cfg
    if model_type == "bandit":
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple

        return MultiMaskMultiSourceBandSplitRNNSimple(**cfg.model), cfg
    if model_type == "bandit_v2":
        from models.bandit_v2.bandit import Bandit

        return Bandit(**cfg.kwargs), cfg
    if model_type == "scnet_unofficial":
        from models.scnet_unofficial import SCNet

        return SCNet(**cfg.model), cfg
    if model_type == "scnet":
        from models.scnet import SCNet

        return SCNet(**cfg.model), cfg
    if model_type == "apollo":
        from models.look2hear.models import BaseModel

        return BaseModel.apollo(**cfg.model), cfg
    if model_type == "bs_mamba2":
        from models.ts_bs_mamba2 import Separator

        return Separator(**cfg.model), cfg
    raise ValueError(f"Unknown model type: {model_type}")


# ----------------------------------------------------------------------------- #
#                               HELPERS                                         #
# ----------------------------------------------------------------------------- #
def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    win = torch.ones(window_size)
    win[:fade_size] = torch.linspace(0.0, 1.0, fade_size)
    win[-fade_size:] = torch.linspace(1.0, 0.0, fade_size)
    return win


def prefer_target_instrument(cfg: ConfigDict) -> List[str]:
    return [cfg.training.target_instrument] if cfg.training.get("target_instrument") else cfg.training.instruments


# ----------------------------------------------------------------------------- #
#                               DEMIX                                           #
# ----------------------------------------------------------------------------- #
def demix(
    cfg: ConfigDict,
    model: torch.nn.Module,
    mix: torch.Tensor,  # tensor FP16 na GPU
    device: torch.device,
    model_type: str,
    pbar: bool = False,
) -> Dict[str, np.ndarray]:
    """Separa fontes, evitando qualquer mismatch de dimensão."""

    is_demucs = model_type == "htdemucs"
    if is_demucs:
        chunk_size = cfg.training.samplerate * cfg.training.segment
        n_inst = len(cfg.training.instruments)
        step = chunk_size // cfg.inference.num_overlap
    else:
        chunk_size = cfg.audio.chunk_size
        n_inst = len(prefer_target_instrument(cfg))
        step = chunk_size // cfg.inference.num_overlap
        fade_size = chunk_size // 10
        border = chunk_size - step
        win_base = _getWindowingArray(chunk_size, fade_size).to(device)
        if mix.shape[-1] > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = cfg.inference.batch_size
    use_amp = getattr(cfg.training, "use_amp", True)

    out_buf = torch.zeros((n_inst,) + mix.shape[-2:], dtype=torch.float16, device=device)
    cnt_buf = torch.zeros_like(out_buf)

    i = 0
    chunks, meta = [], []
    bar = tqdm(total=mix.shape[1], desc="chunks", leave=False) if pbar else None

    while i < mix.shape[1]:
        part = mix[:, i : i + chunk_size]
        raw_len = part.shape[-1]
        part = nn.functional.pad(part, (0, chunk_size - raw_len), mode="reflect")
        chunks.append(part)
        meta.append((i, raw_len))
        i += step

        if len(chunks) >= batch_size or i >= mix.shape[1]:
            with torch.cuda.amp.autocast(device.type == "cuda" and use_amp), torch.inference_mode():
                preds = model(torch.stack(chunks, 0))

            for j, (start, raw_len) in enumerate(meta):
                pred_len = preds[j].shape[-1]
                usable = min(raw_len, pred_len)

                if is_demucs:
                    out_buf[..., start : start + usable] += preds[j, ..., :usable]
                    cnt_buf[..., start : start + usable] += 1
                else:
                    win = win_base
                    if start == 0:
                        win = win.clone()
                        win[:fade_size] = 1
                    elif i >= mix.shape[1]:
                        win = win.clone()
                        win[-fade_size:] = 1

                    out_buf[..., start : start + usable] += preds[j, ..., :usable] * win[..., :usable]
                    cnt_buf[..., start : start + usable] += win[..., :usable]

            chunks.clear()
            meta.clear()
        if bar:
            bar.update(step)
    if bar:
        bar.close()

    result = (out_buf / cnt_buf).float().cpu().numpy()
    if not is_demucs and mix.shape[-1] > 2 * border and border > 0:
        result = result[..., border:-border]

    instruments = cfg.training.instruments if is_demucs else prefer_target_instrument(cfg)
    return {k: v for k, v in zip(instruments, result)}


# ----------------------------------------------------------------------------- #
#                               MÉTRICAS (inalteradas)                          #
# ----------------------------------------------------------------------------- #
def sdr(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    eps = 1e-8
    num = np.sum(ref**2, axis=(1, 2)) + eps
    den = np.sum((ref - est) ** 2, axis=(1, 2)) + eps
    return 10 * np.log10(num / den)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    eps = 1e-8
    scale = np.sum(estimate * reference + eps, axis=(0, 1)) / np.sum(reference**2 + eps, axis=(0, 1))
    reference = reference * np.expand_dims(scale, (0, 1))
    return float(
        np.mean(
            10
            * np.log10(
                np.sum(reference**2, axis=(0, 1))
                / (np.sum((reference - estimate) ** 2, axis=(0, 1)) + eps)
                + eps
            )
        )
    )


def L1Freq_metric(reference: np.ndarray, estimate: np.ndarray, fft_size=2048, hop_size=1024, device="cpu") -> float:
    ref = torch.from_numpy(reference).to(device)
    est = torch.from_numpy(estimate).to(device)
    loss = 10 * F.l1_loss(torch.abs(torch.stft(est, fft_size, hop_size, return_complex=True)),
                          torch.abs(torch.stft(ref, fft_size, hop_size, return_complex=True)))
    return 100 / (1 + float(loss.cpu()))


def NegLogWMSE_metric(reference: np.ndarray, estimate: np.ndarray, mix: np.ndarray, device="cpu") -> float:
    from torch_log_wmse import LogWMSE

    metric = LogWMSE(audio_length=reference.shape[-1] / 44100, sample_rate=44100, return_as_loss=False)
    r = torch.from_numpy(reference)[None, None].to(device)
    e = torch.from_numpy(estimate)[None, None].to(device)
    m = torch.from_numpy(mix)[None].to(device)
    return -float(metric(m, r, e).cpu())


def AuraSTFT_metric(reference: np.ndarray, estimate: np.ndarray, device="cpu") -> float:
    from auraloss.freq import STFTLoss

    loss = STFTLoss(device=device)
    r = torch.from_numpy(reference)[None].to(device)
    e = torch.from_numpy(estimate)[None].to(device)
    return float(100 / (1 + 10 * loss(r, e)))


def AuraMRSTFT_metric(reference: np.ndarray, estimate: np.ndarray, device="cpu") -> float:
    from auraloss.freq import MultiResolutionSTFTLoss

    loss = MultiResolutionSTFTLoss(device=device)
    r = torch.from_numpy(reference)[None].float().to(device)
    e = torch.from_numpy(estimate)[None].float().to(device)
    return float(100 / (1 + 10 * loss(r, e)))


def bleed_full(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int = 44100,
    n_fft: int = 4096,
    hop_length: int = 1024,
    n_mels: int = 512,
    device="cpu",
) -> Tuple[float, float]:
    from torchaudio.transforms import AmplitudeToDB

    ref = torch.from_numpy(reference).float().to(device)
    est = torch.from_numpy(estimate).float().to(device)
    win = torch.hann_window(n_fft).to(device)

    D1 = torch.abs(torch.stft(ref, n_fft, hop_length, window=win, return_complex=True))
    D2 = torch.abs(torch.stft(est, n_fft, hop_length, window=win, return_complex=True))

    mel = torch.from_numpy(librosa.filters.mel(sr, n_fft, n_mels)).to(device)
    S1, S2 = torch.matmul(mel, D1), torch.matmul(mel, D2)
    S1_db = AmplitudeToDB()(S1)
    S2_db = AmplitudeToDB()(S2)

    diff = S2_db - S1_db
    pos, neg = diff[diff > 0], diff[diff < 0]
    bleedless = 100 / (torch.mean(pos) + 1) if pos.numel() else 100
    fullness = 100 / (-torch.mean(neg) + 1) if neg.numel() else 100
    return float(bleedless.cpu()), float(fullness.cpu())


def get_metrics(
    metrics: List[str],
    reference: np.ndarray,
    estimate: np.ndarray,
    mix: np.ndarray,
    device="cpu",
) -> Dict[str, float]:
    ret: Dict[str, float] = {}
    L = min(reference.shape[1], estimate.shape[1])
    reference, estimate, mix = reference[..., :L], estimate[..., :L], mix[..., :L]

    if "sdr" in metrics:
        ret["sdr"] = sdr(reference[None], estimate[None])[0]
    if "si_sdr" in metrics:
        ret["si_sdr"] = si_sdr(reference, estimate)
    if "l1_freq" in metrics:
        ret["l1_freq"] = L1Freq_metric(reference, estimate, device=device)
    if "neg_log_wmse" in metrics:
        ret["neg_log_wmse"] = NegLogWMSE_metric(reference, estimate, mix, device=device)
    if "aura_stft" in metrics:
        ret["aura_stft"] = AuraSTFT_metric(reference, estimate, device=device)
    if "aura_mrstft" in metrics:
        ret["aura_mrstft"] = AuraMRSTFT_metric(reference, estimate, device=device)
    if "bleedless" in metrics or "fullness" in metrics:
        b, f = bleed_full(reference, estimate, device=device)
        if "bleedless" in metrics:
            ret["bleedless"] = b
        if "fullness" in metrics:
            ret["fullness"] = f
    return ret
