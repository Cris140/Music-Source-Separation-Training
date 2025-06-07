import os
import torch
import yaml
from urllib.parse import quote
import subprocess
import time
import sys
import glob
from IPython.display import clear_output
extensions = (".wav", ".mp3", ".m4a", ".weba", ".flac", ".ogg", ".mp4", ".webv", ".opus", ".m4v", ".avi", ".mpg", ".mkv")

class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)

def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)

# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',
tuple_constructor)

def conf_edit(config_path, chunk_size, overlap):
    with open(config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # handle cases where 'use_amp' is missing from config:
    if 'use_amp' not in data.keys():
      data['training']['use_amp'] = True

    data['audio']['chunk_size'] = chunk_size
    data['inference']['num_overlap'] = overlap

    if data['inference']['batch_size'] == 1:
      data['inference']['batch_size'] = 2

    print("Using custom overlap and chunk_size values:")
    print(f"overlap = {data['inference']['num_overlap']}")
    print(f"chunk_size = {data['audio']['chunk_size']}")
    print(f"batch_size = {data['inference']['batch_size']}")

    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, Dumper=IndentDumper, allow_unicode=True)

def download_file(url):
    # Encode the URL to handle spaces and special characters
    encoded_url = quote(url, safe=':/')

    path = 'ckpts'
    os.makedirs(path, exist_ok=True)
    filename = os.path.basename(encoded_url)
    file_path = os.path.join(path, filename)

    if os.path.exists(file_path):
        return

    try:
        response = torch.hub.download_url_to_file(encoded_url, file_path)
        print(f"File '{filename}' downloaded successfully")
    except Exception as e:
        print(f"Error downloading file '{filename}' from '{url}': {e}")

#@markdown # Separation
#@markdown *Separation config:*
input = "./songs/Tim Maia - O Descobridor Dos Sete Mares.flac" # @param ["/content/drive/MyDrive/songs/"] {"allow-input":true}
output_folder = "./separated/" # @param ["/content/drive/MyDrive/separated/"] {"allow-input":true}
#@markdown
source_model = 'VOCALS-Mel-Roformer FT3 Preview (by unwa)' #@param ['VOCALS-MelBand-Roformer (by Becruily)', 'VOCALS-Mel-Roformer big beta 4 (by unwa)', 'VOCALS-Melband-Roformer BigBeta5e (by unwa)','VOCALS-Melband-Roformer BigBeta6 (by unwa)', 'VOCALS-Melband-Roformer BigBeta6X (by unwa)', 'VOCALS-MelBand-Roformer (by KimberleyJSN)', 'VOCALS-MelBand-Roformer Kim FT (by Unwa)', 'VOCALS-MelBand-Roformer Kim FT 2 (by Unwa)', 'VOCALS-MelBand-Roformer Kim FT 2 Bleedless (by Unwa)', 'VOCALS-Mel-Roformer FT3 Preview (by unwa)', 'VOCALS-BS-Roformer_1296 (by viperx)', 'VOCALS-BS-Roformer_1297 (by viperx)', 'VOCALS-BS-RoformerLargev1 (by unwa)', 'VOCALS-BS-Roformer Revive (by unwa)']
target_model = 'INST-Mel-Roformer INSTV7 (by Gabox)' #@param ['INST-MelBand-Roformer (by Becruily)', 'INST-Mel-Roformer v1 (by unwa)','INST-Mel-Roformer v2 (by unwa)', 'INST-Mel-Roformer v1e (by unwa)', 'INST-Mel-Roformer v1e+ (by unwa)', 'INST-Mel-Roformer INSTV7 (by Gabox)', 'INST-VOC-Mel-Roformer a.k.a. duality (by unwa)', 'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa)', 'INST-MelBand-Roformer inst_gabox3 (by Gabox)']
export_format = 'flac PCM_16' #@param ['wav FLOAT', 'flac PCM_16', 'flac PCM_24']
keep_only_corrected_files = True # @param {"type":"boolean"}
#@markdown ---
#@markdown *Roformers custom config:*
overlap = 2 #@param {type:"slider", min:2, max:40, step:1}
chunk_size = 485100 #@param [132300, 352800, 485100] {type:"raw"}
source_inference = True # @param {"type":"boolean"}
target_inference = True # @param {"type":"boolean"}
#phasefix_testmode = True # @param {"type":"boolean"}
#@markdown ---
#@markdown *Phase Fixer custom config:*
scale_factor = 0.05 # @param {"type":"slider","min":0.5,"max":3,"step":0.05}
low_cutoff = 1900 # @param {"type":"slider","min":100,"max":2000,"step":100}
high_cutoff = 2000 # @param {"type":"slider","min":2000,"max":10000,"step":100}


"""if not os.path.exists(input):
  print(f"Invalid Input! Make sure to input a valid Google Drive Path that points to a folder or audio file.")
  sys.exit()"""

'''if output_folder.startswith(r"/content/drive/MyDrive/"):
  if not os.path.exists(output_folder):
    print(f"Trying to create output directory \"{output_folder}\"")
    os.mkdir(output_folder)
    time.sleep(3)
    if os.path.exists(output_folder):
      print(f"Output directory created successfully")
    else:
      print(f"Couldn't create output directory. Make sure to input a valid directory that exists in your Drive.")
      sys.exit()
else:
  print(f"Invalid Output Folder! Make sure to input a valid Google Drive Path, like \"/content/drive/MyDrive/separated)\"")
  sys.exit()'''

if export_format.startswith('flac'):
    flac_file = True
    pcm_type = export_format.split(' ')[1]
else:
    flac_file = False
    pcm_type = None

if keep_only_corrected_files == True:
  output_base = f"temp/source"
  output_unwa = f"temp/target"
else:
  output_base = f"{output_folder}/source"
  output_unwa = f"{output_folder}/target"

if input.endswith(extensions):
    input_type_arg = ['--input_file', input]
else:
    input_type_arg = ['--input_folder', input]

if source_inference == True:
  if source_model == 'VOCALS-BS-Roformer_1297 (by viperx)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/model_bs_roformer_ep_317_sdr_12.9755.yaml'
    start_check_point = 'ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt'
    download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml')
    download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    conf_edit(config_path, chunk_size, overlap)

  elif source_model == 'VOCALS-BS-Roformer_1296 (by viperx)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/model_bs_roformer_ep_368_sdr_12.9628.yaml'
    start_check_point = 'ckpts/model_bs_roformer_ep_368_sdr_12.9628.ckpt'
    download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt')
    download_file('https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml')
    conf_edit(config_path, chunk_size, overlap)

  elif source_model == 'VOCALS-MelBand-Roformer (by KimberleyJSN)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_vocals_mel_band_roformer_kj.yaml'
    start_check_point = 'ckpts/MelBandRoformer.ckpt'
    download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml')
    download_file('https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt')
    conf_edit(config_path, chunk_size, overlap)

  elif source_model == 'VOCALS-MelBand-Roformer Kim FT (by Unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
    start_check_point = 'ckpts/kimmel_unwa_ft.ckpt'
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt')
    conf_edit(config_path, chunk_size, overlap)

  elif source_model == 'VOCALS-MelBand-Roformer Kim FT 2 Bleedless (by Unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
    start_check_point = 'ckpts/kimmel_unwa_ft2_bleedless.ckpt'
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2_bleedless.ckpt')
    conf_edit(config_path, chunk_size, overlap)

  elif source_model == 'VOCALS-MelBand-Roformer Kim FT 2 (by Unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
    start_check_point = 'ckpts/kimmel_unwa_ft2.ckpt'
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2.ckpt')
    conf_edit(config_path, chunk_size, overlap)

  elif source_model == 'VOCALS-BS-Roformer Revive (by unwa)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/config.yaml'
    start_check_point = 'ckpts/bs_roformer_revive.ckpt'
    download_file('https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive.ckpt')
    download_file('https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/config.yaml')
    conf_edit(config_path, chunk_size, overlap)

  elif source_model == 'VOCALS-BS-RoformerLargev1 (by unwa)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/config_bsrofoL.yaml'
    start_check_point = 'ckpts/BS-Roformer_LargeV1.ckpt'
    download_file('https://huggingface.co/jarredou/unwa_bs_roformer/resolve/main/BS-Roformer_LargeV1.ckpt')
    download_file('https://huggingface.co/jarredou/unwa_bs_roformer/raw/main/config_bsrofoL.yaml')
    conf_edit(config_path, chunk_size, overlap)

  elif source_model == 'VOCALS-Melband-Roformer BigBeta6X (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/big_beta6x.yaml'
    start_check_point = 'ckpts/big_beta6x.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6x.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6x.yaml')
    conf_edit(config_path, chunk_size, overlap)

  elif source_model == 'VOCALS-Melband-Roformer BigBeta6 (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/big_beta6.yaml'
    start_check_point = 'ckpts/big_beta6.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.yaml')
    conf_edit(config_path, chunk_size, overlap)

  elif source_model == 'VOCALS-Melband-Roformer BigBeta5e (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/big_beta5e.yaml'
    start_check_point = 'ckpts/big_beta5e.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.yaml')
    conf_edit(config_path, chunk_size, overlap)

  elif source_model == 'VOCALS-Mel-Roformer big beta 4 (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_big_beta4.yaml'
    start_check_point = 'ckpts/melband_roformer_big_beta4.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/raw/main/config_melbandroformer_big_beta4.yaml')
    conf_edit(config_path, chunk_size, overlap)

  elif source_model == 'VOCALS-MelBand-Roformer (by Becruily)':
      model_type = 'mel_band_roformer'
      config_path = 'ckpts/config_vocals_becruily.yaml'
      start_check_point = 'ckpts/mel_band_roformer_vocals_becruily.ckpt'
      download_file('https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/config_vocals_becruily.yaml')
      download_file('https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt')
      conf_edit(config_path, chunk_size, overlap)

  elif source_model == 'VOCALS-Mel-Roformer FT3 Preview (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
    start_check_point = 'ckpts/kimmel_unwa_ft3_prev.ckpt'
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft3_prev.ckpt')
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
    conf_edit(config_path, chunk_size, overlap)

  if source_inference:
    if os.path.exists('temp/source/'):
        # Remove todos os arquivos dentro de temp/source/
        files = glob.glob('temp/source/*.*')
        for file in files:
            os.remove(file)

    print("STARTING SOURCE MODEL INFERENCE")

    # Monta o comando base
    command = [
        'python', 'inference.py',
        '--model_type', model_type,
        '--config_path', config_path,
        '--start_check_point', start_check_point,
        *input_type_arg,
        '--store_dir', output_base,
        '--extract_instrumental'
    ]

    # Adiciona opções extras
    if flac_file:
        command.append('--flac_file')
    if pcm_type:
        command.extend(['--pcm_type', pcm_type])
    command.append('--use_tta')
    # Executa o comando
    result = subprocess.run(command, capture_output=True, text=True)

    # Exibe a saída do processo
    print(result.stdout)
    if result.stderr:
        print("Erros durante a execução:", result.stderr)

if target_inference == True:
  if target_model == 'INST-Mel-Roformer v1 (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_inst.yaml'
    start_check_point = 'ckpts/melband_roformer_inst_v1.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml')
    conf_edit(config_path, chunk_size, overlap)

  elif target_model == 'INST-Mel-Roformer v2 (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_inst_v2.yaml'
    start_check_point = 'ckpts/melband_roformer_inst_v2.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v2.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst_v2.yaml')
    conf_edit(config_path, chunk_size, overlap)

  elif target_model == 'INST-Mel-Roformer v1e (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_inst.yaml'
    start_check_point = 'ckpts/inst_v1e.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml')
    conf_edit(config_path, chunk_size, overlap)

  elif target_model == 'INST-VOC-Mel-Roformer a.k.a. duality (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_instvoc_duality.yaml'
    start_check_point = 'ckpts/melband_roformer_instvoc_duality_v1.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml')
    conf_edit(config_path, chunk_size, overlap)

  elif target_model == 'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_instvoc_duality.yaml'
    start_check_point = 'ckpts/melband_roformer_instvox_duality_v2.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml')
    conf_edit(config_path, chunk_size, overlap)

  elif target_model == 'INST-MelBand-Roformer inst_gabox3 (by Gabox)':
      model_type = 'mel_band_roformer'
      config_path = 'ckpts/inst_gabox.yaml'
      start_check_point = 'ckpts/inst_gabox3.ckpt'
      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox3.ckpt')
      conf_edit(config_path, chunk_size, overlap)

  elif target_model == 'INST-MelBand-Roformer (by Becruily)':
      model_type = 'mel_band_roformer'
      config_path = 'ckpts/config_instrumental_becruily.yaml'
      start_check_point = 'ckpts/mel_band_roformer_instrumental_becruily.ckpt'
      download_file('https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/config_instrumental_becruily.yaml')
      download_file('https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt')
      conf_edit(config_path, chunk_size, overlap)

  elif target_model == 'INST-Mel-Roformer INSTV7 (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/inst_gabox.yaml'
    start_check_point = 'ckpts/Inst_GaboxV7.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxV7.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

  elif target_model == 'INST-Mel-Roformer v1e+ (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_inst.yaml'
    start_check_point = 'ckpts/inst_v1e_plus.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e_plus.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml')
    conf_edit(config_path, chunk_size, overlap)


  if target_inference:
    if os.path.exists('temp/target/'):
        # Remove todos os arquivos dentro de temp/target/
        files = glob.glob('temp/target/*.*')
        for file in files:
            os.remove(file)

    print("STARTING TARGET MODEL INFERENCE")

    # Monta o comando base
    command = [
        'python', 'inference.py',
        '--model_type', model_type,
        '--config_path', config_path,
        '--start_check_point', start_check_point,
        *input_type_arg,
        '--store_dir', output_unwa
    ]

    # Adiciona opções extras
    if flac_file:
        command.append('--flac_file')
    if pcm_type:
        command.extend(['--pcm_type', pcm_type])
    command.append('--use_tta')

    # Executa o comando
    result = subprocess.run(command, capture_output=True, text=True)

    # Exibe a saída do processo
    print(result.stdout)
    if result.stderr:
        print("Erros durante a execução:", result.stderr)

#if phasefix_testmode == True:
if source_inference == True and target_inference == True:
  clear_output(wait=True)
  print(f"Starting Phase Fixing process...")

output_base = os.path.normpath(f"./{output_base}")
output_unwa = os.path.normpath(f"./{output_unwa}")  
command = [
    'python', 'torch_colab.py',
    '--base_folder', output_base,
    '--unwa_folder', output_unwa,
    '--low_cutoff', str(low_cutoff),
    '--high_cutoff', str(high_cutoff),
    '--scale_factor', str(scale_factor),
    '--output_folder', output_folder
]

# Executa o comando
result = subprocess.run(command, capture_output=True, text=True)

# Exibe a saída do processo
print(result.stdout)
if result.stderr:
    print("Erros durante a execução:", result.stderr)
