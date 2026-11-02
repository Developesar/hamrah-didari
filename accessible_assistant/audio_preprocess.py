"""Speech preparation with NumPy: DC removal, voice activity, spectral subtraction and MFCC analysis."""
from __future__ import annotations

import numpy as np

FRAME_MS, HOP_MS = 25, 10


def _frames(x: np.ndarray, rate: int):
    size, hop = int(rate * FRAME_MS / 1000), int(rate * HOP_MS / 1000)
    if len(x) < size:
        x = np.pad(x, (0, size - len(x)))
    count = 1 + (len(x) - size) // hop
    index = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    return x[index], size, hop


def remove_dc(x: np.ndarray) -> np.ndarray:
    return x - float(np.mean(x))


def pre_emphasis(x: np.ndarray, coef: float = 0.97) -> np.ndarray:
    return np.append(x[:1], x[1:] - coef * x[:-1]).astype(np.float32)


def frame_energy_db(x: np.ndarray, rate: int) -> np.ndarray:
    frames, _, _ = _frames(x, rate)
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)


def voice_activity(x: np.ndarray, rate: int, margin_db: float = 10.0) -> dict:
    """Energy VAD relative to the quietest 10% of frames (noise floor)."""
    energy = frame_energy_db(x, rate)
    floor = float(np.percentile(energy, 10))
    speech = energy > floor + margin_db
    peak = float(np.percentile(energy, 95))
    return {"speech_ratio": float(speech.mean()), "noise_floor_db": round(floor, 1),
            "snr_estimate_db": round(peak - floor, 1), "speech_frames": speech}


def trim_silence(x: np.ndarray, rate: int, pad_ms: int = 200) -> np.ndarray:
    speech = voice_activity(x, rate)["speech_frames"]
    if not speech.any():
        return x
    hop = int(rate * HOP_MS / 1000)
    first, last = int(np.argmax(speech)), len(speech) - 1 - int(np.argmax(speech[::-1]))
    pad = int(rate * pad_ms / 1000)
    start = max(0, first * hop - pad)
    end = min(len(x), (last + 1) * hop + int(rate * FRAME_MS / 1000) + pad)
    return x[start:end]


def spectral_subtraction(x: np.ndarray, rate: int, strength: float = 1.0, floor: float = 0.05) -> np.ndarray:
    """Subtract the average magnitude spectrum of the quietest frames; overlap-add resynthesis."""
    size, hop = int(rate * FRAME_MS / 1000), int(rate * HOP_MS / 1000)
    padded = np.pad(x, (size, size + hop))  # full overlap at both ends, nothing cut off
    frames, size, hop = _frames(padded, rate)
    window = np.hanning(size).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=1)
    magnitude, phase = np.abs(spectrum), np.angle(spectrum)
    energy = np.mean(frames ** 2, axis=1)
    quiet = magnitude[energy <= np.percentile(energy, 10)]
    noise = quiet.mean(axis=0) if len(quiet) else np.zeros(magnitude.shape[1])
    cleaned = np.maximum(magnitude - strength * noise, floor * magnitude)
    output = np.fft.irfft(cleaned * np.exp(1j * phase), n=size, axis=1) * window
    result = np.zeros(hop * (len(frames) - 1) + size, dtype=np.float32)
    weight = np.zeros_like(result)
    for i, frame in enumerate(output):
        result[i * hop:i * hop + size] += frame
        weight[i * hop:i * hop + size] += window ** 2
    result = result / np.maximum(weight, 1e-6)
    return result[size:size + len(x)].astype(np.float32)


def _mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def _hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def mfcc(x: np.ndarray, rate: int, n_mfcc: int = 13, n_mels: int = 26) -> np.ndarray:
    """Classic MFCC: pre-emphasis, Hamming frames, power spectrum, mel filterbank, log, DCT-II."""
    frames, size, _ = _frames(pre_emphasis(x), rate)
    n_fft = 1 << (size - 1).bit_length()
    power = np.abs(np.fft.rfft(frames * np.hamming(size), n=n_fft, axis=1)) ** 2 / n_fft
    mel_points = _hz(np.linspace(_mel(0), _mel(rate / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * mel_points / rate).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            bank[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            bank[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    log_mel = np.log(power @ bank.T + 1e-10)
    n = np.arange(n_mels)
    dct = np.cos(np.pi / n_mels * (n[None, :] + 0.5) * np.arange(n_mfcc)[:, None])
    return (log_mel @ dct.T).astype(np.float32)


def prepare_speech(x: np.ndarray, rate: int, denoise_below_snr_db: float = 20.0) -> tuple[np.ndarray, dict]:
    """DC removal, silence trim, denoise only for noisy recordings; returns signal and analysis."""
    x = remove_dc(np.asarray(x, dtype=np.float32))
    vad = voice_activity(x, rate)
    steps = ["remove_dc"]
    if vad["speech_ratio"] > 0:
        x = trim_silence(x, rate)
        steps.append("trim_silence")
    if 0 < vad["snr_estimate_db"] < denoise_below_snr_db:
        x = spectral_subtraction(x, rate)
        steps.append("spectral_subtraction")
    features = mfcc(x, rate) if len(x) else np.zeros((0, 13), np.float32)
    analysis = {"steps": steps, "speech_ratio": round(vad["speech_ratio"], 3),
                "snr_estimate_db": vad["snr_estimate_db"], "noise_floor_db": vad["noise_floor_db"],
                "mfcc_frames": int(features.shape[0]),
                "mfcc_mean": [round(float(v), 2) for v in features.mean(axis=0)] if len(features) else []}
    return x, analysis
