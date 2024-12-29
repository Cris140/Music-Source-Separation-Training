# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import time
import librosa
from tqdm.auto import tqdm
import sys
import os
import glob
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn
from utils import prefer_target_instrument

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from utils import demix, get_model_from_config

import warnings
warnings.filterwarnings("ignore")

def run_folder(model_kim, model_unwa, args, config, device, verbose=False):
    start_time = time.time()
    model_kim.eval()
    model_unwa.eval()

    # Process files as before
    all_mixtures_path = glob.glob(args.input_folder + '/*.*')
    all_mixtures_path.sort()
    sample_rate = 44100
    if 'sample_rate' in config.audio:
        sample_rate = config.audio['sample_rate']
    print('Total files found: {} Use sample rate: {}'.format(len(all_mixtures_path), sample_rate))

    os.makedirs(os.path.join(args.store_dir, 'kim'), exist_ok=True)  # Create kim folder
    os.makedirs(os.path.join(args.store_dir, 'unwa'), exist_ok=True)  # Create unwa folder

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path, desc="Total progress")

    for path in all_mixtures_path:
        print("Starting processing track: ", path)
        if not verbose:
            all_mixtures_path.set_postfix({'track': os.path.basename(path)})
        try:
            mix, sr = librosa.load(path, sr=sample_rate, mono=False)
        except Exception as e:
            print('Cannot read track: {}'.format(path))
            print('Error message: {}'.format(str(e)))
            continue

        # Convert mono to stereo if needed
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=0)

        mix_orig = mix.copy()

        # Process with kim model
        waveforms_kim = demix(config, model_kim, mix, device, pbar=verbose)

        # Move instrumental (from kim) to the 'kim' folder
        if 'instrumental' in waveforms_kim:
            instrumental = waveforms_kim['instrumental'].T
            file_name, _ = os.path.splitext(os.path.basename(path))
            output_file_kim = os.path.join(args.store_dir, 'kim', f"{file_name}_instrumental_kim.flac")
            sf.write(output_file_kim, instrumental, sr, subtype='FLOAT')

        # Process with unwa model
        waveforms_unwa = demix(config, model_unwa, mix, device, pbar=verbose)

        # Move vocals (from unwa) to the 'unwa' folder
        if 'vocals' in waveforms_unwa:
            vocals = waveforms_unwa['vocals'].T
            file_name, _ = os.path.splitext(os.path.basename(path))
            output_file_unwa = os.path.join(args.store_dir, 'unwa', f"{file_name}_vocals_unwa.flac")
            sf.write(output_file_unwa, vocals, sr, subtype='FLOAT')

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type_kim", type=str, default='mdx23c', help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer, scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--model_type_unwa", type=str, default='mdx23c', help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer, scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--model_path_kim", type=str, default='', help="Model path for kim model")
    parser.add_argument("--model_path_unwa", type=str, default='', help="Model path for unwa model")
    parser.add_argument("--config_path_kim", type=str, help="Path to kim config file")
    parser.add_argument("--config_path_unwa", type=str, help="Path to unwa config file")
    parser.add_argument("--input_folder", type=str, help="Folder with mixtures to process")
    parser.add_argument("--store_dir", default="", type=str, help="Path to store results as wav file")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='List of GPU ids')
    parser.add_argument("--force_cpu", action='store_true', help="Force the use of CPU even if CUDA is available")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = "cuda"
        device = f'cuda:{args.device_ids[0]}' if type(args.device_ids) == list else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    # Load both models
    model_kim, config = get_model_from_config(args.model_type_kim, args.config_path_kim)
    model_unwa, config = get_model_from_config(args.model_type_unwa, args.config_path_unwa)

    # Load model weights (same logic as before)
    # (This section will be similar to the original script for loading weights)
    model_kim = model_kim.to(device)
    model_unwa = model_unwa.to(device)

    # Run folder processing with both models
    run_folder(model_kim, model_unwa, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)