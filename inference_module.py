# coding: utf-8
"""
Inference otimizado · Jun/2025
---------------------------------
Novidades:
• Converte para PCM_int (16/24 bits) antes do sf.write em FLAC,
  eliminando o AssertionError.
• Clipping em [-1, 1] e uso de numpy contíguo.
"""

from __future__ import annotations
import argparse, time, librosa, os, glob, torch, warnings, numpy as np, soundfile as sf
from tqdm.auto import tqdm
import torch.nn as nn
from utils import prefer_target_instrument, demix, get_model_from_config

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_flush_denormal(True)


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
    compile_mode: str = "default",
) -> None:

    args = [
        "--model_type", model_type,
        "--config_path", config_path,
        "--start_check_point", start_check_point,
        "--store_dir", store_dir,
        "--device_ids", *map(str, device_ids if isinstance(device_ids, list) else [device_ids]),
        "--compile_mode", compile_mode,
    ]
    if input_folder:          args += ["--input_folder", input_folder]
    if input_file:            args += ["--input_file", input_file]
    if extract_instrumental:  args.append("--extract_instrumental")
    if disable_detailed_pbar: args.append("--disable_detailed_pbar")
    if force_cpu:             args.append("--force_cpu")
    if flac_file:             args.append("--flac_file")
    if pcm_type:              args += ["--pcm_type", pcm_type]
    if use_tta:               args.append("--use_tta")

    proc_folder(args)


# --------------------------------------------------------------------------
def run_folder(model, args, config, device) -> None:
    start = time.time()
    model.eval()

    # ---------------- lista de arquivos -----------------
    if args.input_file:
        files = [args.input_file]
    else:
        exts = {".mp3",".wav",".flac",".aac",".ogg",".m4a",".weba",".mp4",".webm",".opus"}
        files = [f for f in glob.glob(os.path.join(args.input_folder, "*"))
                 if os.path.isfile(f) and os.path.splitext(f)[1].lower() in exts]
        files.sort()

    sr = config.audio.get("sample_rate", 44100)
    instruments = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    for path in tqdm(files, desc="Processamento"):
        mix, _ = librosa.load(path, sr=sr, mono=False)
        if mix.ndim == 1:
            mix = np.stack([mix, mix], axis=0)
        mix_orig = mix.copy()

        if config.inference.get("normalize", False):
            mono = mix.mean(0)
            mean, std = mono.mean(), mono.std()
            mix = (mix - mean) / std

        mixes = ([mix, mix[::-1], -mix] if args.use_tta else [mix])
        tensors = [torch.as_tensor(m, dtype=torch.float16, device=device) for m in mixes]

        results = [demix(config, model, t, device, model_type=args.model_type,
                         pbar=not args.disable_detailed_pbar) for t in tensors]

        wavs = results[0]
        for i, r in enumerate(results[1:], 1):
            for k in r:
                wavs[k] += -r[k] if i == 2 else r[k][::-1]

        for k in wavs:
            wavs[k] /= len(results)

        if args.extract_instrumental:
            ref = "vocals" if "vocals" in instruments else instruments[0]
            if "instrumental" not in instruments: instruments.append("instrumental")
            wavs["instrumental"] = mix_orig - wavs[ref]

        # ------------- salvar stems --------------
        base = os.path.splitext(os.path.basename(path))[0]
        for inst in instruments:
            est = wavs[inst].T
            if config.inference.get("normalize", False):
                est = est * std + mean

            if args.flac_file:
                outfile = os.path.join(args.store_dir, f"{base}_{inst}.flac")
                est = np.clip(est, -1.0, 1.0)                 # evita overflow
                if args.pcm_type == "PCM_16":
                    est_int = (est * 32767.0).round().astype(np.int16)
                else:  # PCM_24 → usa int32 com 24 bits significativos
                    est_int = (est * 8388607.0).round().astype(np.int32)
                est_int = np.ascontiguousarray(est_int)
                sf.write(outfile, est_int, sr, subtype=args.pcm_type)
            else:
                outfile = os.path.join(args.store_dir, f"{base}_{inst}.wav")
                sf.write(outfile, np.ascontiguousarray(est.astype(np.float32)),
                         sr, subtype="FLOAT")

    print(f"Inferência concluída em {time.time()-start:.1f}s")


# --------------------------------------------------------------------------
def proc_folder(arg_list: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_type")
    p.add_argument("--config_path")
    p.add_argument("--start_check_point", default="")
    p.add_argument("--input_folder"); p.add_argument("--input_file")
    p.add_argument("--store_dir", required=True)
    p.add_argument("--device_ids", nargs="+", type=int, default=[0])
    p.add_argument("--extract_instrumental", action="store_true")
    p.add_argument("--disable_detailed_pbar", action="store_true")
    p.add_argument("--force_cpu", action="store_true")
    p.add_argument("--flac_file", action="store_true")
    p.add_argument("--pcm_type", choices=["PCM_16","PCM_24"], default="PCM_24")
    p.add_argument("--use_tta", action="store_true")
    p.add_argument("--compile_mode", choices=["default","reduce","autotune","off"],
                   default="default")
    args = p.parse_args(arg_list)

    # --------------- dispositivo ---------------
    if args.force_cpu:          device = torch.device("cpu")
    elif torch.cuda.is_available(): device = torch.device(f"cuda:{args.device_ids[0]}")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print("Dispositivo:", device)

    # --------------- modelo --------------------
    model, cfg = get_model_from_config(args.model_type, args.config_path)
    if args.start_check_point:
        state = torch.load(args.start_check_point, map_location="cpu")
        state = state.get("state_dict", state.get("state", state))
        model.load_state_dict(state, strict=False)

    if device.type == "cuda":
        model = model.half().to(device)
        if args.compile_mode != "off" and torch.__version__.startswith("2"):
            mode = {"default": None, "reduce": "reduce-overhead", "autotune": "max-autotune"}[args.compile_mode]
            model = torch.compile(model, mode=mode)
    else:
        model = model.to(device)

    if len(args.device_ids) > 1 and device.type == "cuda":
        model = nn.DataParallel(model, device_ids=args.device_ids)

    run_folder(model, args, cfg, device)


if __name__ == "__main__":
    run_inference(model_type="mdx23c", config_path="config.yaml",
                  store_dir="out", input_folder="samples")
