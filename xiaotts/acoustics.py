"""WORLD-assisted acoustic annotation; WORLD is not required for neural inference."""
import math
import numpy as np
import pyworld
from scipy.signal import resample_poly
import soundfile as sf
from xiaotts.dsp import SAMPLE_RATE, HOP, N_FFT


def read_audio(path, max_seconds=None):
    wave, sr = sf.read(path, dtype="float64", always_2d=True)
    if wave.shape[1] != 1:
        raise ValueError("Expected mono recording")
    wave = wave[:, 0]
    if sr != SAMPLE_RATE:
        divisor = math.gcd(sr, SAMPLE_RATE)
        wave = resample_poly(wave, SAMPLE_RATE//divisor, sr//divisor)
    if max_seconds:
        wave = wave[:int(max_seconds*SAMPLE_RATE)]
    if not np.isfinite(wave).all() or np.max(np.abs(wave)) < 1e-5:
        raise ValueError("Invalid or silent audio")
    # Do not destructively normalize or trim raw data.
    return np.ascontiguousarray(wave, dtype=np.float64)


def extract_controls(wave):
    frame_period = HOP / SAMPLE_RATE * 1000
    f0, times = pyworld.dio(wave, SAMPLE_RATE, frame_period=frame_period,
                           f0_floor=55, f0_ceil=750)
    f0 = pyworld.stonemask(wave, f0, times, SAMPLE_RATE)
    spectrum = pyworld.cheaptrick(wave, f0, times, SAMPLE_RATE, fft_size=2048)
    aperiodicity = pyworld.d4c(wave, f0, times, SAMPLE_RATE, fft_size=2048)
    voiced = f0 > 0
    if voiced.any():
        continuous_f0 = np.interp(np.arange(len(f0)), np.flatnonzero(voiced), f0[voiced])
    else:
        continuous_f0 = np.full_like(f0, 180)
    controls = np.zeros((len(f0), 67), dtype=np.float32)
    controls[:, 0] = np.log(continuous_f0.clip(55, 750))
    controls[:, 1] = np.where(voiced, 8, -8)
    padded = np.pad(wave, (N_FFT//2, N_FFT//2), mode="reflect")
    rms = np.sqrt(np.array([np.mean(padded[i*HOP:i*HOP+N_FFT]**2)
                           for i in range(len(f0))]).clip(1e-10))
    controls[:, 2] = np.log(rms.clip(1e-5, 1))
    frequencies = np.linspace(0, SAMPLE_RATE/2, spectrum.shape[1])
    mel_max = 2595*np.log10(1+SAMPLE_RATE/2/700)
    harmonic = np.sqrt((spectrum*(1-aperiodicity**2)).clip(1e-12))
    noise = np.sqrt((spectrum*aperiodicity**2).clip(1e-12))
    for value, bands, offset in ((harmonic, 40, 3), (noise, 24, 43)):
        band_hz = 700*(10**(np.linspace(0, mel_max, bands)/2595)-1)
        controls[:, offset:offset+bands] = np.array([
            np.interp(band_hz, frequencies, np.log(frame.clip(1e-6))) for frame in value])
    # Common centering retains the relative harmonic/noise level.
    controls[:, 3:] -= np.max(controls[:, 3:], axis=1, keepdims=True)
    samples = (len(f0)-1)*HOP
    waveform = wave[:samples].astype(np.float32)
    return controls, waveform, (f0, spectrum, aperiodicity)
