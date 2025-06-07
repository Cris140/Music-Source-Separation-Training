"""
Master-script (jun/2025) otimizado para Mel-Roformer
---------------------------------------------------
• Permite definir chunk_size, num_overlap, batch_size e use_tta por linha de comando.
• Corrige o dicionário mapping do _prepare (erro de parêntese).
• Mantém todas as demais acelerações (FP16, torch.compile, etc.).
"""

from __future__ import annotations
import os, glob, yaml, torch, argparse
from urllib.parse import quote
import inference_module as inference
import phase_fix_module as phasefix

# ------------------------- CLI ------------------------------------
def _parse_args():
    ap = argparse.ArgumentParser(prog="separation_refactored.py",
                                 description="Pipeline rápido Mel-Roformer")
    ap.add_argument("--chunk_size",  type=int, default=1_048_576,
                    help="tamanho do bloco em amostras (default: 1 Mi)")
    ap.add_argument("--num_overlap", type=int, default=1,
                    help="número de sobreposições (default: 1)")
    ap.add_argument("--batch_size",  type=int, default=4,
                    help="batch size de inferência (default: 4)")
    ap.add_argument("--use_tta",     action="store_true",
                    help="ativa Test-Time Augmentation (3× mais lento)")
    return ap.parse_args()

cli = _parse_args()

# -------------------- Configuração fixa ---------------------------
input_path    = "C:/Users/Cris/Music-Source-Separation-Training/songs/"
output_folder = "C:/Users/Cris/Music-Source-Separation-Training/separated/"
source_model  = "VOCALS-Mel-Roformer FT3 Preview (by unwa)"
target_model  = "INST-Mel-Roformer INSTV7 (by Gabox)"
export_fmt    = "flac PCM_16"

flac_file = export_fmt.startswith("flac")
pcm_type  = export_fmt.split(" ")[1] if flac_file else None

tmp_src = os.path.join(output_folder, "_tmp_src")
tmp_tgt = os.path.join(output_folder, "_tmp_tgt")

extensions = (".wav",".mp3",".m4a",".weba",".flac",".ogg",
              ".mp4",".webv",".opus",".m4v",".avi",".mpg",".mkv")

# -------------------- utilidades internas -------------------------
class _IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)

def _tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))
yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _tuple_constructor)

def _edit_cfg(path: str):
    """Altera chunk_size / num_overlap / batch_size do YAML."""
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    data["audio"]["chunk_size"]        = cli.chunk_size
    data["inference"]["num_overlap"]   = cli.num_overlap
    data["inference"]["batch_size"]    = cli.batch_size
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False,
                  sort_keys=False, Dumper=_IndentDumper, allow_unicode=True)

def _dl(url: str, dst="ckpts"):
    os.makedirs(dst, exist_ok=True)
    fname = os.path.join(dst, os.path.basename(quote(url, safe=':/')))
    if not os.path.exists(fname):
        print("Baixando:", fname)
        torch.hub.download_url_to_file(url, fname)
    return fname

def _clean(path: str):
    if os.path.isdir(path):
        for f in glob.glob(os.path.join(path, "*.*")):
            os.remove(f)

def _prepare(defn: str, is_source: bool):
    """Retorna (model_type, cfg_path, ckpt_path)."""
    if is_source:
        mapping = {
            "VOCALS-Mel-Roformer FT3 Preview (by unwa)": (
                "mel_band_roformer",
                (
                    _dl("https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml"),
                    _dl("https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft3_prev.ckpt"),
                ),
            ),
        }
    else:  # target
        mapping = {
            "INST-Mel-Roformer INSTV7 (by Gabox)": (
                "mel_band_roformer",
                (
                    _dl("https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml"),
                    _dl("https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxV7.ckpt"),
                ),
            ),
        }

    if defn not in mapping:
        raise ValueError(f"Modelo '{defn}' não mapeado.")
    mtype, (cfg, ckpt) = mapping[defn]
    _edit_cfg(cfg)
    return mtype, cfg, ckpt

# --------------------------- Execução ------------------------------
if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)

    # ---------- SOURCE ----------
    _clean(tmp_src)
    mtype, cfg, ckpt = _prepare(source_model, True)
    inference.run_inference(
        model_type=mtype,
        config_path=cfg,
        start_check_point=ckpt,
        input_folder=None if input_path.lower().endswith(extensions) else input_path,
        input_file=input_path if input_path.lower().endswith(extensions) else None,
        store_dir=tmp_src,
        extract_instrumental=True,
        flac_file=flac_file,
        pcm_type=pcm_type,
        use_tta=cli.use_tta,
    )

    # ---------- TARGET ----------
    _clean(tmp_tgt)
    mtype, cfg, ckpt = _prepare(target_model, False)
    inference.run_inference(
        model_type=mtype,
        config_path=cfg,
        start_check_point=ckpt,
        input_folder=None if input_path.lower().endswith(extensions) else input_path,
        input_file=input_path if input_path.lower().endswith(extensions) else None,
        store_dir=tmp_tgt,
        flac_file=flac_file,
        pcm_type=pcm_type,
        use_tta=cli.use_tta,
    )

    # ---------- Phase-Fix ----------
    phasefix.process_files(
        base_folder=os.path.normpath(tmp_src),
        unwa_folder=os.path.normpath(tmp_tgt),
        output_folder=output_folder,
        low_cutoff=1900,
        high_cutoff=2000,
        scale_factor=0.05,
        output_32bit=False,
    )

    print("Pipeline concluído.")
