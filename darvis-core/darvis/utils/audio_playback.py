import io
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

_playback_lock = threading.Lock()
_interrupt_flag = threading.Event()


def play_wav_bytes(
    audio_bytes: bytes,
    blocking: bool = True,
    target_sample_rate: Optional[int] = None,
    priority: bool = False
) -> None:
    import wave

    with io.BytesIO(audio_bytes) as buffer:
        with wave.open(buffer, 'rb') as wav:
            sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            n_frames = wav.getnframes()
            audio_data = wav.readframes(n_frames)

    if sample_width == 2:
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(audio_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
        audio = audio.mean(axis=1)

    if target_sample_rate and target_sample_rate != sample_rate:
        audio = _resample(audio, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    if priority:
        _interrupt_flag.set()
        sd.stop()

    with _playback_lock:
        if priority:
            _interrupt_flag.clear()
        sd.play(audio, samplerate=sample_rate)
        if blocking:
            sd.wait()


def is_interrupted() -> bool:
    return _interrupt_flag.is_set()


def stop_playback() -> None:
    sd.stop()


def _resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    if orig_rate == target_rate:
        return audio

    duration = len(audio) / orig_rate
    target_length = int(duration * target_rate)

    x_orig = np.linspace(0, duration, len(audio))
    x_target = np.linspace(0, duration, target_length)

    return np.interp(x_target, x_orig, audio).astype(np.float32)
