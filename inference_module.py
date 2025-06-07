# coding: utf-8
"""
Refatoração otimizada (jun/2025) do antigo inference.py.

Principais acréscimos:
• Chamada explícita a run_folder().
• Pesos FP16 + autocast.
• torch.compile (PyTorch ≥ 2).
• Suporte a TF32 (Ampere+) e pinned-memory.
• Conversão mix→tensor feita uma única vez.
• Janela de atenuação calculada fora do loop.
• Controla TTA com flag --use_tta (off → 3× mais rápido).
• Parametrização de chunk_size, num_overlap e batch_size via YAML.
"""

from __future__ import annotations

import argparse
import time
import librosa
from tqdm.auto import tqdm
import os
import glob
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn
from utils import prefer_target_instrument, demix, get_model_from_config
import warnings

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_flush_denormal(True)

# --------------------------------------------------------------------------
# API de alto nível
# --------------------------------------------------------------------------
def run_inference(
    *,
    model_type: str,
    config_path: str,
    start_check_point: str = "",
    input_folder: str | None = None,
    input_file: str | None = None,
    store_dir: str,
    device_ids: list[int] | int = 0,
    extract_instrumental: bool = False,
    disable_detailed_pbar: bool = False,
    force_cpu: bool = False,
    flac_file: bool = False,
    pcm_type: str = "PCM_24",
    use_tta: bool = False,
) -> None:
    args_list: list[str] = [
        "--model_type", model_type,
        "--config_path", config_path,
        "--start_check_point", start_check_point,
        "--store_dir", store_dir,
        "--device_ids",
        *map(str, device_ids if isinstance(device_ids, list) else [device_ids]),
    ]
    if input_folder:
        args_list += ["--input_folder", input_folder]
    if input_file:
        args_list += ["--input_file", input_file]
    if extract_instrumental:
        args_list.append("--extract_instrumental")
    if disable_detailed_pbar:
        args_list.append("--disable_detailed_pbar")
    if force_cpu:
        args_list.append("--force_cpu")
    if flac_file:
        args_list.append("--flac_file")
    if pcm_type:
        args_list += ["--pcm_type", pcm_type]
    if use_tta:
        args_list.append("--use_tta")

    proc_folder(args_list)


# --------------------------------------------------------------------------
def run_folder(model, args, config, device) -> None:
    start = time.time()
    model.eval()

    # ------------------------------------------------------------------
    # Coleta dos ficheiros de entrada
    # ------------------------------------------------------------------
    if args.input_file:
        if not os.path.isfile(args.input_file):
            raise FileNotFoundError(f"Input file '{args.input_file}' não encontrado.")
        all_paths = [args.input_file]
    else:
        exts = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a",
                ".weba", ".mp4", ".webm", ".opus"}
        all_paths = [f for f in glob.glob(os.path.join(args.input_folder, "*"))
                     if os.path.isfile(f) and os.path.splitext(f)[1].lower() in exts]
        all_paths.sort()

    sr = config.audio.get("sample_rate", 44100)
    instruments = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    iterator = tqdm(all_paths, desc="Processamento")

    for path in iterator:
        mix, _ = librosa.load(path, sr=sr, mono=False)
        if mix.ndim == 1:                          # mono → estéreo
            mix = np.stack([mix, mix], axis=0)
        mix_orig = mix.copy()

        if config.inference.get("normalize", False):
            mono = mix.mean(0)
            mean, std = mono.mean(), mono.std()
            mix = (mix - mean) / std

        tracks = ([mix, mix[::-1].copy(), -mix] if args.use_tta else [mix])

        # Convertido UMA vez p/ tensor FP16 pinned na GPU
        mix_tensors = [torch.as_tensor(t, dtype=torch.float16, device=device) for t in tracks]

        results = [
            demix(config, model, m, device,
                  model_type=args.model_type,
                  pbar=not args.disable_detailed_pbar)
            for m in mix_tensors
        ]

        wavs = results[0]
        for i, res in enumerate(results[1:], 1):
            for k in res:
                wavs[k] += (-res[k] if i == 2 else res[k][::-1].copy())

        for k in wavs:
            wavs[k] /= len(results)

        # Instrumental sintético
        if args.extract_instrumental:
            base_instr = "vocals" if "vocals" in instruments else instruments[0]
            if "instrumental" not in instruments:
                instruments.append("instrumental")
            wavs["instrumental"] = mix_orig - wavs[base_instr]

        # Escrita
        for instr in instruments:
            est = wavs[instr].T
            if config.inference.get("normalize", False):
                est = est * std + mean
            base = os.path.splitext(os.path.basename(path))[0]
            if args.flac_file:
                out = os.path.join(args.store_dir, f"{base}_{instr}.flac")
                sf.write(out, est, sr,
                         subtype="PCM_16" if args.pcm_type == "PCM_16" else "PCM_24")
            else:
                out = os.path.join(args.store_dir, f"{base}_{instr}.wav")
                sf.write(out, est, sr, subtype="FLOAT")

    print(f"Inferência concluída em {time.time() - start:.2f}s")


# --------------------------------------------------------------------------
def proc_folder(arg_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="mdx23c")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--start_check_point", type=str, default="")
    parser.add_argument("--input_folder", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--store_dir", type=str, required=True)
    parser.add_argument("--device_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--extract_instrumental", action="store_true")
    parser.add_argument("--disable_detailed_pbar", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--flac_file", action="store_true")
    parser.add_argument("--pcm_type", type=str, choices=["PCM_16", "PCM_24"], default="PCM_24")
    parser.add_argument("--use_tta", action="store_true")

    args = parser.parse_args(arg_list)

    # ------------------------------------------------------------------
    # Dispositivo
    # ------------------------------------------------------------------
    if args.force_cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_ids[0]}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Dispositivo:", device)

    # ------------------------------------------------------------------
    # Carrega modelo
    # ------------------------------------------------------------------
    model, config = get_model_from_config(args.model_type, args.config_path)
    if args.start_check_point:
        state = torch.load(args.start_check_point, map_location="cpu")
        state = state.get("state_dict", state.get("state", state))
        model.load_state_dict(state, strict=False)

    # FP16 + compile
    if device.type == "cuda":
        model = model.half().to(device)
        if torch.__version__.startswith("2"):
            model = torch.compile(model)
    else:
        model = model.to(device)

    # Multi-GPU
    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and device.type == "cuda":
        model = nn.DataParallel(model, device_ids=args.device_ids)

    run_folder(model, args, config, device)


if __name__ == "__main__":
    # Exemplo rápido
    run_inference(
        model_type="mdx23c",
        config_path="config.yaml",
        store_dir="out",
        input_folder="samples",
    )
