#phase_fix_module.py
"""
Refatoração de torch_colab.py
Mantém API pública process_files para correção de fase/magnitude,
mas remove lógica de linha de comando para uso como módulo importável.
"""

from __future__ import annotations
import torch
import torchaudio
import os
import glob

# -------------------- Funções internas -----------------------------
def _frequency_blend_phases(
    phase1, phase2, freq_bins, *,
    low_cutoff: int, high_cutoff: int,
    base_factor: float = 0.25, scale_factor: float = 1.85
):
    if phase1.shape != phase2.shape:
        raise ValueError("phase1 e phase2 devem ter o mesmo formato.")
    if len(freq_bins) != phase1.shape[0]:
        raise ValueError("freq_bins deve ter o mesmo tamanho da dimensão de frequência.")
    if low_cutoff >= high_cutoff:
        raise ValueError("low_cutoff deve ser menor que high_cutoff.")

    blended_phase = torch.zeros_like(phase1)
    blend_factors = torch.zeros_like(freq_bins)

    blend_factors[freq_bins < low_cutoff] = base_factor
    blend_factors[freq_bins > high_cutoff] = base_factor + scale_factor

    in_range = (freq_bins >= low_cutoff) & (freq_bins <= high_cutoff)
    blend_factors[in_range] = base_factor + scale_factor * (
        (freq_bins[in_range] - low_cutoff) / (high_cutoff - low_cutoff)
    )

    for i in range(phase1.shape[0]):
        blended_phase[i, :] = (
            (1 - blend_factors[i]) * phase1[i, :] + blend_factors[i] * phase2[i, :]
        )

    return torch.remainder(blended_phase + torch.pi, 2 * torch.pi) - torch.pi


def _transfer_magnitude_phase(
    source_file: str,
    target_file: str,
    *,
    transfer_magnitude: bool,
    transfer_phase: bool,
    low_cutoff: int,
    high_cutoff: int,
    scale_factor: float,
    output_32bit: bool,
    output_folder: str | None
):
    tgt_name, tgt_ext = os.path.splitext(os.path.basename(target_file))
    tgt_name = (
        tgt_name.replace("_other", "")
        .replace("_vocals", "")
        .replace("_instrumental", "")
        .replace("_Other", "")
        .replace("_Vocals", "")
        .replace("_Instrumental", "")
        .strip()
    )

    out_file = (
        os.path.join(output_folder, f"{tgt_name} (Fixed Instrumental){tgt_ext}")
        if output_folder
        else os.path.join(os.path.dirname(target_file), f"{tgt_name} (Corrected){tgt_ext}")
    )

    print(f"Phase Fixing {tgt_name}{tgt_ext}…")
    src_wav, src_sr = torchaudio.load(source_file)
    tgt_wav, tgt_sr = torchaudio.load(target_file)
    if src_sr != tgt_sr:
        raise ValueError("Sample rates incompatíveis.")

    n_fft      = 2048
    hop_length = 512
    window     = torch.hann_window(n_fft, device=src_wav.device)

    # 1) com argumentos nomeados  (mais claro)
    src_stft = torch.stft(
        src_wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, center=True, return_complex=True
    )
    tgt_stft = torch.stft(
        tgt_wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, center=True, return_complex=True
    )
    freqs = torch.linspace(0, src_sr // 2, steps=n_fft // 2 + 1)

    modified = []
    for s, t in zip(src_stft, tgt_stft):
        s_mag, s_phs = torch.abs(s), torch.angle(s)
        t_mag, t_phs = torch.abs(t), torch.angle(t)

        stft_mod = t.clone()
        if transfer_magnitude:
            stft_mod = s_mag * torch.exp(1j * torch.angle(stft_mod))
        if transfer_phase:
            phs_blend = _frequency_blend_phases(
                t_phs, s_phs, freqs,
                low_cutoff=low_cutoff, high_cutoff=high_cutoff,
                scale_factor=scale_factor
            )
            stft_mod = torch.abs(stft_mod) * torch.exp(1j * phs_blend)

        modified.append(stft_mod)

    wav_out = torch.istft(
        torch.stack(modified),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=src_wav.size(1)
    )

    if tgt_ext.lower() == ".flac":
        torchaudio.save(out_file, wav_out, tgt_sr)
    else:
        torchaudio.save(
            out_file, wav_out, tgt_sr,
            encoding="PCM_S", bits_per_sample=32 if output_32bit else 16
        )
    print("Salvo:", out_file)


# -------------------- API pública ------------------------------------------------
def process_files(
    *,
    base_folder: str,
    unwa_folder: str,
    output_folder: str,
    low_cutoff: int = 500,
    high_cutoff: int = 5000,
    scale_factor: float = 1.85,
    output_32bit: bool = False
) -> None:
    """
    Itera sobre pares instrumental/unwa e aplica correção de fase.
    """
    os.makedirs(output_folder, exist_ok=True)

    unwa_files = sorted(glob.glob(os.path.join(unwa_folder, "*")))
    if not unwa_files:
        print("Nenhum arquivo encontrado em", unwa_folder)
        return

    for unwa in unwa_files:
        base = (
            os.path.splitext(os.path.basename(unwa))[0]
            .replace("_other", "")
            .replace("_vocals", "")
            .replace("_instrumental", "")
            .replace("_Other", "")
            .replace("_Vocals", "")
            .replace("_Instrumental", "")
            .strip()
        )
        inst_file = os.path.join(base_folder, f"{base}_instrumental{os.path.splitext(unwa)[1]}")
        if not os.path.exists(inst_file):
            print("Aviso: instrumental não encontrado para", unwa)
            continue

        _transfer_magnitude_phase(
            source_file=inst_file,
            target_file=unwa,
            transfer_magnitude=False,
            transfer_phase=True,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            scale_factor=scale_factor,
            output_32bit=output_32bit,
            output_folder=output_folder,
        )
