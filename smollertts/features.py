"""Native HiFi-GAN mel targets and recorded pitch/voicing/energy annotations.

Mel conventions follow Jungil Kong's HiFi-GAN (MIT; licenses/HIFIGAN-LICENSE.txt).
WORLD and eSpeak NG are preprocessing tools, never deployed neural components.
"""
from functools import lru_cache
import math
import re
import subprocess
import unicodedata

import numpy as np
import torch
from torch.nn import functional as F

SAMPLE_RATE, HOP, FFT = 22050, 256, 1024
VOWELS = set("aeiouyɑɐɒæɛɜɞəɚɝɘɤɨɪɔœøɵʉʊʌɯɶɷ")


@lru_cache(maxsize=1)
def mel_basis():
    from librosa.filters import mel
    return torch.from_numpy(mel(sr=SAMPLE_RATE, n_fft=FFT, n_mels=80, fmin=0, fmax=8000)).float()


def native_mel(wave):
    """B×samples -> B×80×frames; magnitude mel, not power mel; no normalization."""
    if wave.ndim != 2 or wave.shape[-1] < FFT or not torch.isfinite(wave).all():
        raise ValueError("Expected finite audio of at least 1024 samples")
    padded = F.pad(wave.unsqueeze(1), (384, 384), mode="reflect").squeeze(1)
    spectrum = torch.stft(padded, FFT, hop_length=HOP, win_length=FFT,
                          window=torch.hann_window(FFT, device=wave.device, dtype=wave.dtype),
                          center=False, normalized=False, onesided=True, return_complex=True)
    magnitude = (spectrum.real.square() + spectrum.imag.square() + 1e-9).sqrt()
    return (mel_basis().to(wave) @ magnitude).clamp_min(1e-5).log()


def recorded_prosody(wave):
    import pyworld
    # The original annotator measured WORLD at t*hop and then re-gridded to
    # native HiFi-GAN centres (t+.5)*hop. Keep that convention exactly.
    wave = np.ascontiguousarray(wave, dtype=np.float64)
    f0, times = pyworld.dio(wave, SAMPLE_RATE, frame_period=HOP / SAMPLE_RATE * 1000,
                           f0_floor=55, f0_ceil=750)
    f0 = pyworld.stonemask(wave, f0, times, SAMPLE_RATE)
    voiced = f0 > 0
    continuous = np.interp(np.arange(len(f0)), np.flatnonzero(voiced), f0[voiced]) if voiced.any() else np.full_like(f0, 180)
    log_pitch = np.log(continuous.clip(55, 750)).astype(np.float32)
    frames = len(wave) // HOP
    centres = np.arange(frames) + .5
    pitch = np.interp(centres, np.arange(len(f0)), log_pitch)
    voice = (np.interp(centres, np.arange(len(f0)), np.where(voiced, 8, -8)) > 0).astype(np.float32)
    native = wave[:frames * HOP].astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(native, (384, 384), mode="reflect"), FFT)[::HOP][:frames]
    energy = np.log(np.sqrt(np.maximum((windows.astype(np.float64) ** 2).mean(-1), 1e-10)))
    return np.stack(((pitch - math.log(180)) / .5, voice, (energy + 3) / 2), -1).astype(np.float32)


def split_phone(phone):
    phone = unicodedata.normalize("NFC", phone)
    if phone in ("", "sil", "sp", "<eps>"):
        return "<sil>", 0
    if phone in ("spn", "<unk>", "unk"):
        raise ValueError("Unknown-word/noise phone in alignment")
    stress = 1 if "ˈ" in phone else 2 if "ˌ" in phone else 0
    return re.sub(r"[ˈˌ˥˦˧˨˩]", "", phone), stress


def is_vowel(phone):
    return any(char in VOWELS for char in phone) or "\u0329" in phone


@lru_cache(maxsize=100000)
def word_stress(word, executable="espeak-ng"):
    result = subprocess.run([executable, "-q", "--ipa=1", "-v", "en-us", word],
                            capture_output=True, check=True, text=True, encoding="utf-8", timeout=30)
    stresses, pending = [], 0
    for phone in re.split(r"[_\s]+", result.stdout.strip()):
        if "ˈ" in phone:
            pending = 1
        elif "ˌ" in phone:
            pending = 2
        if is_vowel(phone):
            stresses.append(pending)
            pending = 0
    return tuple(stresses)


def aligned_phones(alignment, samples, vocabulary, espeak="espeak-ng"):
    entries = alignment["tiers"]["phones"]["entries"]
    words = alignment["tiers"]["words"]["entries"]
    if not entries:
        raise ValueError("Empty phone alignment")
    labels = [split_phone(phone) for _, _, phone in entries]
    matched = 0
    for start, end, word in words:
        indices = [i for i, (a, b, phone) in enumerate(entries)
                   if start - 1e-4 <= (a + b) / 2 < end and phone not in ("", "sil", "sp")]
        if not indices or not word:
            continue
        vowels = [i for i in indices if is_vowel(labels[i][0])]
        stresses = word_stress(word, espeak)
        if len(vowels) == len(stresses):
            matched += 1
            for i, value in zip(vowels, stresses):
                labels[i] = (labels[i][0], value or labels[i][1])
    ends, full, cursor = [], [], 0.
    duration = samples / SAMPLE_RATE
    for (start, end, _), label in zip(entries, labels):
        if not np.isfinite([start, end]).all() or end <= max(start, cursor) or start < cursor - .021 or end > duration + .1:
            raise ValueError("Invalid or overlapping alignment intervals")
        if start > cursor + .005:
            ends.append(start)
            full.append(("<sil>", 0))
        ends.append(end)
        full.append(label)
        cursor = end
    if cursor < duration:
        ends.append(duration + HOP / SAMPLE_RATE)
        full.append(("<sil>", 0))
    tokens = np.array([vocabulary[phone] for phone, _ in full], dtype=np.int64)
    stress = np.array([value for _, value in full], dtype=np.int64)
    frames = samples // HOP
    indices = np.searchsorted(ends, (np.arange(frames) + .5) * HOP / SAMPLE_RATE, side="right").clip(0, len(tokens) - 1)
    durations = np.bincount(indices, minlength=len(tokens)).astype(np.int64)
    return tokens, stress, durations, {"words": len(words), "stress_words_matched": matched}
