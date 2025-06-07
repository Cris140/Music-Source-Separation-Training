import argparse
import torch
import torchaudio
import os
import gc
import glob

def frequency_blend_phases(phase1, phase2, freq_bins, low_cutoff=500, high_cutoff=5000, base_factor=0.25, scale_factor=1.85):
    if phase1.shape != phase2.shape:
        raise ValueError("phase1 and phase2 must have the same shape.")
    if len(freq_bins) != phase1.shape[0]:
        raise ValueError("freq_bins must have the same length as the number of frequency bins in phase1 and phase2.")
    if low_cutoff >= high_cutoff:
        raise ValueError("low_cutoff must be less than high_cutoff.")

    blended_phase = torch.zeros_like(phase1)
    blend_factors = torch.zeros_like(freq_bins)

    blend_factors[freq_bins < low_cutoff] = base_factor
    blend_factors[freq_bins > high_cutoff] = base_factor + scale_factor
    in_range_mask = (freq_bins >= low_cutoff) & (freq_bins <= high_cutoff)
    blend_factors[in_range_mask] = base_factor + scale_factor * (
        (freq_bins[in_range_mask] - low_cutoff) / (high_cutoff - low_cutoff)
    )

    for i in range(phase1.shape[0]):
        blended_phase[i, :] = (1 - blend_factors[i]) * phase1[i, :] + blend_factors[i] * phase2[i, :]

    blended_phase = torch.remainder(blended_phase + torch.pi, 2 * torch.pi) - torch.pi

    return blended_phase

def transfer_magnitude_phase(source_file, target_file, transfer_magnitude=True, transfer_phase=True, low_cutoff=500, high_cutoff=5000, scale_factor=1.85, output_32bit=False, output_folder=None):
    target_name, target_ext = os.path.splitext(os.path.basename(target_file))
    target_name = target_name.replace("_other", "").replace("_vocals", "").replace("_instrumental", "").replace("_Other", "").replace("_Vocals", "").replace("_Instrumental", "").strip()

    output_file = os.path.join(output_folder, f"{target_name} (Fixed Instrumental){target_ext}") if output_folder else os.path.join(os.path.dirname(target_file), f"{target_name} (Corrected){target_ext}")

    print(f"Phase Fixing {target_name}{target_ext}...")
    source_waveform, source_sr = torchaudio.load(source_file)
    target_waveform, target_sr = torchaudio.load(target_file)

    if source_sr != target_sr:
        raise ValueError("Sample rates of source and target audio files must match.")

    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft)

    source_stfts = torch.stft(source_waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, pad_mode="reflect")
    target_stfts = torch.stft(target_waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, pad_mode="reflect")

    freqs = torch.linspace(0, source_sr // 2, steps=n_fft // 2 + 1)

    modified_stfts = []
    for source_stft, target_stft in zip(source_stfts, target_stfts):
        source_mag, source_phs = torch.abs(source_stft), torch.angle(source_stft)
        target_mag, target_phs = torch.abs(target_stft), torch.angle(target_stft)

        modified_stft = target_stft.clone()
        if transfer_magnitude:
            modified_stft = source_mag * torch.exp(1j * torch.angle(modified_stft))

        if transfer_phase:
            blended_phase = frequency_blend_phases(target_phs, source_phs, freqs, low_cutoff, high_cutoff, scale_factor)
            modified_stft = torch.abs(modified_stft) * torch.exp(1j * blended_phase)

        modified_stfts.append(modified_stft)

    modified_waveform = torch.istft(
        torch.stack(modified_stfts),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=source_waveform.size(1)
    )

    ext = target_ext.lower()
    if ext == '.flac':
        # Para FLAC, não passar encoding nem bits_per_sample
        torchaudio.save(output_file, modified_waveform, target_sr)
    else:
        # Para WAV ou outros, aplicar encoding e bits_per_sample
        torchaudio.save(
            output_file,
            modified_waveform,
            target_sr,
            encoding="PCM_S",
            bits_per_sample=32 if output_32bit else 16
        )

    print(f"Corrected file saved as {output_file}")

def process_files(base_folder, unwa_folder, output_folder, low_cutoff, high_cutoff, scale_factor, output_32bit):
    unwa_files = glob.glob(os.path.join(unwa_folder, "*"))
    unwa_files.sort()

    for unwa_file in unwa_files:
        base_name_with_suffix = os.path.splitext(os.path.basename(unwa_file))[0]
        base_name = base_name_with_suffix.strip().replace("_other", "").replace("_vocals", "").replace("_instrumental", "").replace("_Other", "").replace("_Vocals", "").replace("_Instrumental", "")

        instrumental_file = os.path.join(base_folder, f"{base_name}_instrumental{os.path.splitext(unwa_file)[1]}")

        if os.path.exists(instrumental_file):
            transfer_magnitude_phase(
                source_file=instrumental_file,
                target_file=unwa_file,
                transfer_magnitude=False,
                transfer_phase=True,
                low_cutoff=low_cutoff,
                high_cutoff=high_cutoff,
                scale_factor=scale_factor,
                output_32bit=output_32bit,
                output_folder=output_folder
            )
        else:
            print(f"Warning: No matching instrumental file found for {unwa_file}, skipping.")

        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer magnitude and/or phase between audio files.")
    parser.add_argument("--base_folder", required=True, help="Path to the base folder containing instrumental files (kim).")
    parser.add_argument("--unwa_folder", required=True, help="Path to the folder containing corresponding unwa files.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for corrected files.")
    parser.add_argument("--low_cutoff", type=int, default=500, help="Low cutoff frequency for phase blending.")
    parser.add_argument("--high_cutoff", type=int, default=5000, help="High cutoff frequency for phase blending.")
    parser.add_argument("--scale_factor", type=float, default=1.85, help="Scale factor for phase blending.")
    parser.add_argument("--output_32bit", action="store_true", help="Save the output as a 32-bit file.")

    args = parser.parse_args()

    process_files(
        base_folder=args.base_folder,
        unwa_folder=args.unwa_folder,
        output_folder=args.output_folder,
        low_cutoff=args.low_cutoff,
        high_cutoff=args.high_cutoff,
        scale_factor=args.scale_factor,
        output_32bit=args.output_32bit
    )
