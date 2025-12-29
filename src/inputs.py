import torch
import torchaudio
import numpy as np
import pandas as pd
import csv
import os.path
from pathlib import Path

#hyperarameters for log-mel:
sample_rate1 = 16000
n_fft = 1024 # determines the resolution of frequency axis
hop_length = 256 # determines the resolution of time axis | normally 1/4 of n_fft
n_mels = 80 # number of mel filter banks | 80 is a sweet spot for emotion discernment | sums th energy in frequency bands of n_mels size to avoid spiking sound

def get_inputs(csv_path: str):
    ds = pd.read_csv(csv_path)
    inputs: list[tuple[torch.Tensor, int]] = []
    base = Path(".").resolve()
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate1,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    for idx, uri in enumerate(ds['Audio']):

        path = (base / uri).resolve()
        waveform, sample_rate = torchaudio.load(path)
        if (waveform.shape[0] > 1):
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        mel_spec = mel_spectogram(waveform)
        log_mel = torch.log1p(mel_spec)
        inputs.append((log_mel, idx))

    max_len: int = max([w[0].shape[-1] for w in inputs])

    padded_inputs = []
    for w, idx in inputs:
        if w.shape[-1] < max_len:
            w = torch.nn.functional.pad(w, (0, max_len - w.shape[-1]))
        else:
            w = w[:, :max_len]
        padded_inputs.append((w, idx))

    return padded_inputs

def create_audio_CSV(csv_path: str, audio_csv_name: str):
    new_file = Path(".").resolve() / audio_csv_name
    if new_file.exists():
        return 
    inputs = get_inputs(csv_path)
    output_dir = Path("preprocessed_audio")
    output_dir.mkdir(exist_ok=True)

    rows = []
    for waveform, idx in inputs:
        tensor_path = output_dir / f"{idx}.pt"
        torch.save(waveform, tensor_path)
        rows.append([str(tensor_path), idx])  # CSV contains path and idx

    with open(audio_csv_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["tensor_path", "idx"])  # header
        writer.writerows(rows)

