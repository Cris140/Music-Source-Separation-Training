# coding: utf-8
"""
Inference otimizada para Mel-Roformer · VRAM friendly
─────────────────────────────────────────────────────
Novidades
• Flag --compile_mode  [default|reduce|autotune|off]
• torch.cuda.empty_cache() após cada música
• chunk_size/batch_size continuam configuráveis
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
    compile_mode: str = "default",      # ← NOVO
) -> None:

    args_list: list[str] = [
        "--model_type", model_type,
        "--config_path", config_path,
        "--start_check_point", start_check_point,
        "--store_dir", store_dir,
        "--device_ids", *map(str, device_ids if isinstance(device_ids, list) else [device_ids]),
        "--compile_mode", compile_mode,
    ]
    if input_folder:          args_list += ["--input_folder", input_folder]
    if input_file:            args_list += ["--input_file", input_file]
    if extract_instrumental:  args_list.append("--extract_instrumental")
    if disable_detailed_pbar: args_list.append("--disable_detailed_pbar")
    if force_cpu:             args_list.append("--force_cpu")
    if flac_file:             args_list.append("--flac_file")
    if pcm_type:              args_list += ["--pcm_type", pcm_type]
    if use_tta:               args_list.append("--use_tta")

    proc_folder(args_list)

# --------------------------------------------------------------------------
def run_folder(model, args, config, device) -> None:
    start = time.time()
    model.eval()

    if args.input_file:
        if not os.path.isfile(args.input_file):
            raise FileNotFoundError(f"Input file '{args.input_file}' não encontrado.")
        all_paths = [args.input_file]
    else:
        exts = {".mp3",".wav",".flac",".aac",".ogg",".m4a",".weba",".mp4",".webm",".opus"}
        all_paths = [f for f in glob.glob(os.path.join(args.input_folder, "*"))
                     if os.path.isfile(f) and os.path.splitext(f)[1].lower() in exts]
        all_paths.sort()

    sr = config.audio.get("sample_rate", 44100)
    instruments = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    for path in tqdm(all_paths, desc="Processamento"):
        mix, _ = librosa.load(path, sr=sr, mono=False)
        if mix.ndim == 1:
            mix = np.stack([mix, mix], axis=0)
        mix_orig = mix.copy()

        if config.inference.get("normalize", False):
            mono = mix.mean(0)
            mean, std = mono.mean(), mono.std()
            mix = (mix - mean) / std

        mixes = ([mix, mix[::-1].copy(), -mix] if args.use_tta else [mix])
        tensors = [torch.as_tensor(m, dtype=torch.float16, device=device) for m in mixes]

        outs = [demix(config, model, t, device,
                      model_type=args.model_type,
                      pbar=not args.disable_detailed_pbar) for t in tensors]

        wavs = outs[0]
        for i, o in enumerate(outs[1:], 1):
            for k in o:
                wavs[k] += (-o[k] if i == 2 else o[k][::-1].copy())
        for k in wavs: wavs[k] /= len(outs)

        if args.extract_instrumental:
            ref = "vocals" if "vocals" in instruments else instruments[0]
            if "instrumental" not in instruments: instruments.append("instrumental")
            wavs["instrumental"] = mix_orig - wavs[ref]

        base = os.path.splitext(os.path.basename(path))[0]
        for inst in instruments:
            est = wavs[inst].T
            if config.inference.get("normalize", False): est = est * std + mean
            if args.flac_file:
                sf.write(os.path.join(args.store_dir,f"{base}_{inst}.flac"),
                         est, sr, subtype=("PCM_16" if args.pcm_type=="PCM_16" else "PCM_24"))
            else:
                sf.write(os.path.join(args.store_dir,f"{base}_{inst}.wav"),
                         est, sr, subtype="FLOAT")

        if device.type == "cuda":   # libera cache entre faixas
            torch.cuda.empty_cache()

    print(f"Inferência: {time.time()-start:.1f}s")

# --------------------------------------------------------------------------
def proc_folder(arg_list: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", type=str, default="mdx23c")
    p.add_argument("--config_path", required=True)
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
                   default="default")              # ← NOVO
    args = p.parse_args(arg_list)

    # dispositivo ------------------------------------------------------
    if args.force_cpu:          device = torch.device("cpu")
    elif torch.cuda.is_available(): device = torch.device(f"cuda:{args.device_ids[0]}")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else:                       device = torch.device("cpu")
    print("Dispositivo:", device)

    # modelo -----------------------------------------------------------
    model, cfg = get_model_from_config(args.model_type, args.config_path)
    if args.start_check_point:
        s = torch.load(args.start_check_point, map_location="cpu")
        s = s.get("state_dict", s.get("state", s))
        model.load_state_dict(s, strict=False)

    if device.type=="cuda":
        model = model.half().to(device)

        # torch.compile opcional
        if args.compile_mode!="off" and torch.__version__.startswith("2"):
            mode = {"default":None,"reduce":"reduce-overhead",
                    "autotune":"max-autotune"}[args.compile_mode]
            model = torch.compile(model, mode=mode)

    else:
        model = model.to(device)

    if len(args.device_ids)>1 and device.type=="cuda":
        model = nn.DataParallel(model, device_ids=args.device_ids)

    run_folder(model, args, cfg, device)

if __name__ == "__main__":
    run_inference(model_type="mdx23c", config_path="config.yaml",
                  store_dir="out", input_folder="samples")
