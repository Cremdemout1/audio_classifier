import sounddevice as sd
import numpy as np
import torch

def user_listen():
    fs: int = 16000
    duration = 2.0 # seconds

    device_info = sd.query_devices(kind='input')
    channels = device_info['max_input_channels']

    audioRecording: np.array = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    if (channels > 1): # convert to mono sound
        audioRecording = np.mean(audioRecording, axis=1)
    audio_tensor = torch.from_numpy(audioRecording).float().unsqueeze(0)  # [1, T]
    return audio_tensor