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
from typing import Dict, List, Tuple, Union


# ------------------------------------------------------------------ #
# CONFIG & MODEL                                                     #
# ------------------------------------------------------------------ #
def load_config(model_type: str, cfg_path: str) -> Union[ConfigDict, OmegaConf]:
    with open(cfg_path) as f:
        return OmegaConf.load(cfg_path) if model_type == "htdemucs" else ConfigDict(yaml.load(f, Loader=yaml.FullLoader))


def get_model_from_config(model_type: str, cfg_path: str):
    cfg = load_config(model_type, cfg_path)
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


# ------------------------------------------------------------------ #
# HELPERS                                                            #
# ------------------------------------------------------------------ #
def _window_array(N: int, fade: int) -> torch.Tensor:
    win = torch.ones(N)
    win[:fade] = torch.linspace(0.0, 1.0, fade)
    win[-fade:] = torch.linspace(1.0, 0.0, fade)
    return win


def prefer_target_instrument(cfg: ConfigDict) -> List[str]:
    return [cfg.training.target_instrument] if cfg.training.get("target_instrument") else cfg.training.instruments


# ------------------------------------------------------------------ #
# DEMIX                                                              #
# ------------------------------------------------------------------ #
def demix(
    cfg: ConfigDict,
    model: torch.nn.Module,
    mix: torch.Tensor,        # tensor FP16 na GPU
    device: torch.device,
    model_type: str,
    pbar: bool = False,
) -> Dict[str, np.ndarray]:

    is_demucs = model_type == "htdemucs"
    if is_demucs:
        chunk = cfg.training.samplerate * cfg.training.segment
        overlap = cfg.inference.num_overlap
        n_inst = len(cfg.training.instruments)
    else:
        chunk = cfg.audio.chunk_size
        overlap = cfg.inference.num_overlap
        n_inst = len(prefer_target_instrument(cfg))
        fade = chunk // 10
        border = chunk - chunk // overlap
        base_window = _window_array(chunk, fade).to(device)
        if mix.shape[-1] > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    step = chunk // overlap
    batch = cfg.inference.batch_size
    use_amp = getattr(cfg.training, "use_amp", True)

    out = torch.zeros((n_inst,) + mix.shape[-2:], dtype=torch.float16, device=device)
    cnt = torch.zeros_like(out)

    buf, meta, i = [], [], 0
    bar = tqdm(total=mix.shape[1], desc="chunks", leave=False) if pbar else None

    while i < mix.shape[1]:
        seg = mix[:, i : i + chunk]
        seg_len = seg.shape[-1]
        pad_back = chunk - seg_len

        # -------- escolha segura do modo de padding --------
        if pad_back > 0:
            # 'reflect' só se o acolchoamento < comprimento do tensor
            pad_mode = "reflect" if (not is_demucs and pad_back < seg_len) else "constant"
            seg = nn.functional.pad(seg, (0, pad_back), mode=pad_mode)
        buf.append(seg)
        meta.append((i, seg_len))
        i += step

        if len(buf) >= batch or i >= mix.shape[1]:
            with torch.cuda.amp.autocast(device.type == "cuda" and use_amp), torch.inference_mode():
                pred = model(torch.stack(buf, 0))

            for j, (st, raw_len) in enumerate(meta):
                pred_len = pred[j].shape[-1]
                usable = min(raw_len, pred_len)

                if is_demucs:
                    out[..., st : st + usable] += pred[j, ..., :usable]
                    cnt[..., st : st + usable] += 1
                else:
                    win = base_window
                    if st == 0:
                        win = win.clone()
                        win[:fade] = 1
                    elif i >= mix.shape[1]:
                        win = win.clone()
                        win[-fade:] = 1
                    out[..., st : st + usable] += pred[j, ..., :usable] * win[..., :usable]
                    cnt[..., st : st + usable] += win[..., :usable]

            buf.clear(), meta.clear()
        if bar:
            bar.update(step)
    if bar:
        bar.close()

    res = (out / cnt).float().cpu().numpy()
    if not is_demucs and mix.shape[-1] > 2 * border and border > 0:
        res = res[..., border:-border]

    insts = cfg.training.instruments if is_demucs else prefer_target_instrument(cfg)
    return {k: v for k, v in zip(insts, res)}


# ------------------------------------------------------------------ #
# MÉTRICAS (sem alterações de lógica)                                #
# ------------------------------------------------------------------ #
def sdr(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    eps = 1e-8
    num = np.sum(ref**2, axis=(1, 2)) + eps
    den = np.sum((ref - est) ** 2, axis=(1, 2)) + eps
    return 10 * np.log10(num / den)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    eps = 1e-8
    scale = np.sum(estimate * reference + eps, axis=(0, 1)) / np.sum(reference**2 + eps, axis=(0, 1))
    reference *= np.expand_dims(scale, (0, 1))
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


def L1Freq_metric(ref: np.ndarray, est: np.ndarray, fft_size=2048, hop=1024, device="cpu") -> float:
    r, e = torch.from_numpy(ref).to(device), torch.from_numpy(est).to(device)
    loss = 10 * F.l1_loss(torch.abs(torch.stft(e, fft_size, hop, return_complex=True)),
                          torch.abs(torch.stft(r, fft_size, hop, return_complex=True)))
    return 100 / (1 + float(loss.cpu()))


def NegLogWMSE_metric(ref: np.ndarray, est: np.ndarray, mix: np.ndarray, device="cpu") -> float:
    from torch_log_wmse import LogWMSE

    wmse = LogWMSE(audio_length=ref.shape[-1] / 44100, sample_rate=44100, return_as_loss=False)
    return -float(wmse(torch.from_numpy(mix)[None].to(device),
                       torch.from_numpy(ref)[None, None].to(device),
                       torch.from_numpy(est)[None, None].to(device)).cpu())


def AuraSTFT_metric(r: np.ndarray, e: np.ndarray, device="cpu") -> float:
    from auraloss.freq import STFTLoss

    loss = STFTLoss(device=device)
    return float(100 / (1 + 10 * loss(torch.from_numpy(r)[None].to(device),
                                      torch.from_numpy(e)[None].to(device))))


def AuraMRSTFT_metric(r: np.ndarray, e: np.ndarray, device="cpu") -> float:
    from auraloss.freq import MultiResolutionSTFTLoss

    loss = MultiResolutionSTFTLoss(device=device)
    return float(100 / (1 + 10 * loss(torch.from_numpy(r)[None].float().to(device),
                                      torch.from_numpy(e)[None].float().to(device))))


def bleed_full(
    r: np.ndarray,
    e: np.ndarray,
    sr: int = 44100,
    n_fft: int = 4096,
    hop: int = 1024,
    n_mels: int = 512,
    device="cpu",
) -> Tuple[float, float]:
    from torchaudio.transforms import AmplitudeToDB

    R = torch.from_numpy(r).float().to(device)
    E = torch.from_numpy(e).float().to(device)
    win = torch.hann_window(n_fft).to(device)

    D1 = torch.abs(torch.stft(R, n_fft, hop, window=win, return_complex=True))
    D2 = torch.abs(torch.stft(E, n_fft, hop, window=win, return_complex=True))
    mel = torch.from_numpy(librosa.filters.mel(sr, n_fft, n_mels)).to(device)
    S1, S2 = torch.matmul(mel, D1), torch.matmul(mel, D2)
    S1_db, S2_db = AmplitudeToDB()(S1), AmplitudeToDB()(S2)

    diff = S2_db - S1_db
    pos, neg = diff[diff > 0], diff[diff < 0]
    bleedless = 100 / (torch.mean(pos) + 1) if pos.numel() else 100
    fullness = 100 / (-torch.mean(neg) + 1) if neg.numel() else 100
    return float(bleedless.cpu()), float(fullness.cpu())


def get_metrics(
    metrics: List[str],
    ref: np.ndarray,
    est: np.ndarray,
    mix: np.ndarray,
    device="cpu",
) -> Dict[str, float]:
    res: Dict[str, float] = {}
    L = min(ref.shape[1], est.shape[1])
    ref, est, mix = ref[..., :L], est[..., :L], mix[..., :L]

    if "sdr" in metrics:
        res["sdr"] = sdr(ref[None], est[None])[0]
    if "si_sdr" in metrics:
        res["si_sdr"] = si_sdr(ref, est)
    if "l1_freq" in metrics:
        res["l1_freq"] = L1Freq_metric(ref, est, device=device)
    if "neg_log_wmse" in metrics:
        res["neg_log_wmse"] = NegLogWMSE_metric(ref, est, mix, device=device)
    if "aura_stft" in metrics:
        res["aura_stft"] = AuraSTFT_metric(ref, est, device=device)
    if "aura_mrstft" in metrics:
        res["aura_mrstft"] = AuraMRSTFT_metric(ref, est, device=device)
    if "bleedless" in metrics or "fullness" in metrics:
        b, f = bleed_full(ref, est, device=device)
        if "bleedless" in metrics:
            res["bleedless"] = b
        if "fullness" in metrics:
            res["fullness"] = f
    return res
