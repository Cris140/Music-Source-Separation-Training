"""
Pipeline Mel-Roformer com controle de VRAM
──────────────────────────────────────────
Uso:
  python separation_refactored.py --chunk_size 524288 --num_overlap 2 \
       --batch_size 3 --compile_mode off
"""
from __future__ import annotations
import os, glob, yaml, torch, argparse
from urllib.parse import quote
import inference_module as inference
import phase_fix_module as phasefix

# --------------------------- CLI -----------------------------------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_size",  type=int, default=1_048_576)
    ap.add_argument("--num_overlap", type=int, default=1)
    ap.add_argument("--batch_size",  type=int, default=4)
    ap.add_argument("--use_tta",     action="store_true")
    ap.add_argument("--compile_mode", choices=["default","reduce","autotune","off"],
                    default="default")                        # ← NOVO
    return ap.parse_args()
ARGS = _cli()

# --------------------------- Constantes ----------------------------
input_path    = "./songs/"
output_folder = "./separated/"
source_model  = "VOCALS-Mel-Roformer FT3 Preview (by unwa)"
target_model  = "INST-Mel-Roformer INSTV7 (by Gabox)"
export_fmt    = "flac PCM_16"

flac_file = export_fmt.startswith("flac")
pcm_type  = export_fmt.split(" ")[1] if flac_file else None
tmp_src, tmp_tgt = (os.path.join(output_folder, p) for p in ("_tmp_src","_tmp_tgt"))
extensions = (".wav",".mp3",".m4a",".weba",".flac",".ogg",
              ".mp4",".webv",".opus",".m4v",".avi",".mpg",".mkv")

# ------------------ utilidades internas ----------------------------
class _Indent(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)
def _tuple(loader,node): return tuple(loader.construct_sequence(node))
yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _tuple)

def _edit_cfg(path: str):
    with open(path) as f: data = yaml.load(f, Loader=yaml.SafeLoader)
    data["audio"]["chunk_size"]      = ARGS.chunk_size
    data["inference"]["num_overlap"] = ARGS.num_overlap
    data["inference"]["batch_size"]  = ARGS.batch_size
    with open(path,"w") as f:
        yaml.dump(data,f,default_flow_style=False,sort_keys=False,
                  Dumper=_Indent,allow_unicode=True)

def _dl(url:str,dst="ckpts"):
    os.makedirs(dst,exist_ok=True)
    fn=os.path.join(dst,os.path.basename(quote(url,safe=':/')))
    if not os.path.exists(fn):
        print("Baixando:",fn)
        torch.hub.download_url_to_file(url,fn)
    return fn

def _clean(p:str):
    if os.path.isdir(p):
        for f in glob.glob(os.path.join(p,"*.*")): os.remove(f)

def _prepare(defn:str,src:bool):
    if src:
        mapping={
            "VOCALS-Mel-Roformer FT3 Preview (by unwa)":(
                "mel_band_roformer",
                (_dl("https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml"),
                 _dl("https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft3_prev.ckpt")),)}
    else:
        mapping={
            "INST-Mel-Roformer INSTV7 (by Gabox)":(
                "mel_band_roformer",
                (_dl("https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml"),
                 _dl("https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxV7.ckpt")),)}
    mtype,(cfg,ckpt)=mapping[defn]; _edit_cfg(cfg); return mtype,cfg,ckpt

# ------------------------- Execução --------------------------------
if __name__=="__main__":
    os.makedirs(output_folder,exist_ok=True)

    _clean(tmp_src)
    mt,cfg,ck=_prepare(source_model,True)
    inference.run_inference(
        model_type=mt, config_path=cfg, start_check_point=ck,
        input_folder=None if input_path.lower().endswith(extensions) else input_path,
        input_file=input_path if input_path.lower().endswith(extensions) else None,
        store_dir=tmp_src, extract_instrumental=True,
        flac_file=flac_file, pcm_type=pcm_type,
        use_tta=ARGS.use_tta, compile_mode=ARGS.compile_mode)

    _clean(tmp_tgt)
    mt,cfg,ck=_prepare(target_model,False)
    inference.run_inference(
        model_type=mt, config_path=cfg, start_check_point=ck,
        input_folder=None if input_path.lower().endswith(extensions) else input_path,
        input_file=input_path if input_path.lower().endswith(extensions) else None,
        store_dir=tmp_tgt,
        flac_file=flac_file, pcm_type=pcm_type,
        use_tta=ARGS.use_tta, compile_mode=ARGS.compile_mode)

    phasefix.process_files(
        base_folder=os.path.normpath(tmp_src),
        unwa_folder=os.path.normpath(tmp_tgt),
        output_folder=output_folder,
        low_cutoff=1900, high_cutoff=2000, scale_factor=0.05)

    print("Pipeline concluído.")
