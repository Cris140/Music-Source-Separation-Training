# utils.py
# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

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
#                       CARREGAMENTO DE CONFIGURAÇÕES                           #
# ----------------------------------------------------------------------------- #
def load_config(model_type: str, config_path: str) -> Union[ConfigDict, OmegaConf]:
    try:
        with open(config_path, "r") as f:
            if model_type == "htdemucs":
                return OmegaConf.load(config_path)
            return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def get_model_from_config(model_type: str, config_path: str) -> Tuple:
    config = load_config(model_type, config_path)

    if model_type == "mdx23c":
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == "htdemucs":
        from models.demucs4ht import get_model
        model = get_model(config)
    elif model_type == "segm_models":
        from models.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type == "torchseg":
        from models.torchseg_models import Torchseg_Net
        model = Torchseg_Net(config)
    elif model_type == "mel_band_roformer":
        from models.bs_roformer import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == "bs_roformer":
        from models.bs_roformer import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == "swin_upernet":
        from models.upernet_swin_transformers import Swin_UperNet_Model
        model = Swin_UperNet_Model(config)
    elif model_type == "bandit":
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(**config.model)
    elif model_type == "bandit_v2":
        from models.bandit_v2.bandit import Bandit
        model = Bandit(**config.kwargs)
    elif model_type == "scnet_unofficial":
        from models.scnet_unofficial import SCNet
        model = SCNet(**config.model)
    elif model_type == "scnet":
        from models.scnet import SCNet
        model = SCNet(**config.model)
    elif model_type == "apollo":
        from models.look2hear.models import BaseModel
        model = BaseModel.apollo(**config.model)
    elif model_type == "bs_mamba2":
        from models.ts_bs_mamba2 import Separator
        model = Separator(**config.model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, config


# ----------------------------------------------------------------------------- #
#                               UTILITÁRIOS                                     #
# ----------------------------------------------------------------------------- #
def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    fadein, fadeout = torch.linspace(0, 1, fade_size), torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:], window[:fade_size] = fadeout, fadein
    return window


# ----------------------------------------------------------------------------- #
#                                DEMIX                                          #
# ----------------------------------------------------------------------------- #
def demix(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: torch.Tensor,       # tensor FP16 já na GPU
    device: torch.device,
    model_type: str,
    pbar: bool = False,
) -> Dict[str, np.ndarray]:

    use_demucs = model_type == "htdemucs"
    if use_demucs:
        chunk_size = config.training.samplerate * config.training.segment
        num_instruments = len(config.training.instruments)
        num_overlap = config.inference.num_overlap
        step = chunk_size // num_overlap
    else:
        chunk_size = config.audio.chunk_size
        num_instruments = len(prefer_target_instrument(config))
        num_overlap = config.inference.num_overlap
        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]
        window = _getWindowingArray(chunk_size, fade_size).to(device)
        if length_init > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = config.inference.batch_size
    use_amp = getattr(config.training, "use_amp", True)

    with torch.cuda.amp.autocast(enabled=device.type == "cuda" and use_amp), torch.inference_mode():
        res = torch.zeros((num_instruments,) + mix.shape[-2:], dtype=torch.float16, device=device)
        cnt = torch.zeros_like(res)

        i = 0
        buf, loc = [], []
        bar = tqdm(total=mix.shape[1], desc="chunks", leave=False) if pbar else None

        while i < mix.shape[1]:
            part = mix[:, i : i + chunk_size]
            clen = part.shape[-1]
            pad_mode = "reflect" if not use_demucs and clen > chunk_size // 2 else "constant"
            part = nn.functional.pad(part, (0, chunk_size - clen), mode=pad_mode)
            buf.append(part)
            loc.append((i, clen))
            i += step

            if len(buf) >= batch_size or i >= mix.shape[1]:
                x = model(torch.stack(buf, 0))

                for j, (st, ln) in enumerate(loc):
                    out_len = x[j].shape[-1]            # ← comprimento real
                    if out_len != ln:                   #   (pode ser menor ~319)
                        ln = out_len

                    if use_demucs:
                        res[..., st : st + ln] += x[j, ..., :ln]
                        cnt[..., st : st + ln] += 1
                    else:
                        win = window
                        if st == 0:
                            win = win.clone()
                            win[:fade_size] = 1
                        elif i >= mix.shape[1]:
                            win = win.clone()
                            win[-fade_size:] = 1

                        res[..., st : st + ln] += x[j, ..., :ln] * win[..., :ln]
                        cnt[..., st : st + ln] += win[..., :ln]

                buf.clear(), loc.clear()
            if bar:
                bar.update(step)
        if bar:
            bar.close()

        est = (res / cnt).float().cpu().numpy()
        if not use_demucs and length_init > 2 * border and border > 0:
            est = est[..., border:-border]

    instruments = config.training.instruments if use_demucs else prefer_target_instrument(config)
    return {k: v for k, v in zip(instruments, est)}


# ----------------------------------------------------------------------------- #
#                                MÉTRICAS                                       #
# ----------------------------------------------------------------------------- #
def sdr(references: np.ndarray, estimates: np.ndarray) -> np.ndarray:
    eps = 1e-8
    num = np.sum(np.square(references), axis=(1, 2)) + eps
    den = np.sum(np.square(references - estimates), axis=(1, 2)) + eps
    return 10 * np.log10(num / den)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    eps = 1e-8
    scale = np.sum(estimate * reference + eps, axis=(0, 1)) / np.sum(reference**2 + eps, axis=(0, 1))
    scale = np.expand_dims(scale, axis=(0, 1))
    reference = reference * scale
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


def L1Freq_metric(
    reference: np.ndarray,
    estimate: np.ndarray,
    fft_size: int = 2048,
    hop_size: int = 1024,
    device: str = "cpu",
) -> float:
    reference = torch.from_numpy(reference).to(device)
    estimate = torch.from_numpy(estimate).to(device)

    reference_stft = torch.stft(reference, fft_size, hop_size, return_complex=True)
    estimated_stft = torch.stft(estimate, fft_size, hop_size, return_complex=True)

    loss = 10 * F.l1_loss(torch.abs(estimated_stft), torch.abs(reference_stft))
    return 100 / (1.0 + float(loss.cpu().numpy()))


def NegLogWMSE_metric(
    reference: np.ndarray,
    estimate: np.ndarray,
    mixture: np.ndarray,
    device: str = "cpu",
) -> float:
    from torch_log_wmse import LogWMSE

    log_wmse = LogWMSE(
        audio_length=reference.shape[-1] / 44100,
        sample_rate=44100,
        return_as_loss=False,
        bypass_filter=False,
    )

    reference = torch.from_numpy(reference).unsqueeze(0).unsqueeze(0).to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).unsqueeze(0).to(device)
    mixture = torch.from_numpy(mixture).unsqueeze(0).to(device)

    return -float(log_wmse(mixture, reference, estimate).cpu().numpy())


def AuraSTFT_metric(reference: np.ndarray, estimate: np.ndarray, device: str = "cpu") -> float:
    from auraloss.freq import STFTLoss

    stft_loss = STFTLoss(w_log_mag=1.0, w_lin_mag=0.0, w_sc=1.0, device=device)
    reference = torch.from_numpy(reference).unsqueeze(0).to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).to(device)
    return float(100 / (1.0 + 10 * stft_loss(reference, estimate)))


def AuraMRSTFT_metric(reference: np.ndarray, estimate: np.ndarray, device: str = "cpu") -> float:
    from auraloss.freq import MultiResolutionSTFTLoss

    mrstft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 4096],
        hop_sizes=[256, 512, 1024],
        win_lengths=[1024, 2048, 4096],
        scale="mel",
        n_bins=128,
        sample_rate=44100,
        perceptual_weighting=True,
        device=device,
    )
    reference = torch.from_numpy(reference).unsqueeze(0).float().to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).float().to(device)
    return float(100 / (1.0 + 10 * mrstft_loss(reference, estimate)))


def bleed_full(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int = 44100,
    n_fft: int = 4096,
    hop_length: int = 1024,
    n_mels: int = 512,
    device: str = "cpu",
) -> Tuple[float, float]:
    from torchaudio.transforms import AmplitudeToDB

    reference = torch.from_numpy(reference).float().to(device)
    estimate = torch.from_numpy(estimate).float().to(device)
    window = torch.hann_window(n_fft).to(device)

    D1 = torch.abs(torch.stft(reference, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True))
    D2 = torch.abs(torch.stft(estimate, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True))

    mel_basis = torch.from_numpy(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)).to(device)
    S1_db = AmplitudeToDB()(torch.matmul(mel_basis, D1))
    S2_db = AmplitudeToDB()(torch.matmul(mel_basis, D2))

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
    device: str = "cpu",
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    min_len = min(reference.shape[1], estimate.shape[1])
    reference, estimate, mix = reference[..., :min_len], estimate[..., :min_len], mix[..., :min_len]

    if "sdr" in metrics:
        result["sdr"] = sdr(np.expand_dims(reference, 0), np.expand_dims(estimate, 0))[0]
    if "si_sdr" in metrics:
        result["si_sdr"] = si_sdr(reference, estimate)
    if "l1_freq" in metrics:
        result["l1_freq"] = L1Freq_metric(reference, estimate, device=device)
    if "neg_log_wmse" in metrics:
        result["neg_log_wmse"] = NegLogWMSE_metric(reference, estimate, mix, device=device)
    if "aura_stft" in metrics:
        result["aura_stft"] = AuraSTFT_metric(reference, estimate, device=device)
    if "aura_mrstft" in metrics:
        result["aura_mrstft"] = AuraMRSTFT_metric(reference, estimate, device=device)
    if "bleedless" in metrics or "fullness" in metrics:
        b, f = bleed_full(reference, estimate, device=device)
        if "bleedless" in metrics:
            result["bleedless"] = b
        if "fullness" in metrics:
            result["fullness"] = f
    return result


def prefer_target_instrument(config: ConfigDict) -> List[str]:
    return [config.training.target_instrument] if config.training.get("target_instrument") else config.training.instruments
