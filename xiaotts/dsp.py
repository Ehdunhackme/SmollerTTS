"""Zero-parameter differentiable harmonic/noise source-filter renderer.

Minimum-phase filters are reconstructed from log-amplitude bands. Source and
filter operations are explicit so they can be ported to browser DSP. This
implementation must pass the oracle reconstruction gate before broad training.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

SAMPLE_RATE = 22050
HOP = 256
N_FFT = 1024
HARMONIC_BANDS = 40
NOISE_BANDS = 24


def mel(hz):
    return 2595 * torch.log10(1 + hz / 700)


def band_matrix(bands, n_fft=N_FFT, sample_rate=SAMPLE_RATE):
    bins = torch.linspace(0, sample_rate/2, n_fft//2+1)
    positions = mel(bins) / mel(torch.tensor(sample_rate/2)) * (bands-1)
    low = positions.floor().long().clamp(0, bands-2)
    weight = (positions-low).clamp(0, 1)
    matrix = torch.zeros(len(bins), bands)
    matrix[torch.arange(len(bins)), low] = 1-weight
    matrix[torch.arange(len(bins)), low+1] = weight
    return matrix


def minimum_phase(log_amplitude, n_fft=N_FFT):
    cepstrum = torch.fft.irfft(log_amplitude, n=n_fft, dim=-1)
    lifter = torch.zeros(n_fft, device=cepstrum.device, dtype=cepstrum.dtype)
    lifter[0], lifter[n_fft//2] = 1, 1
    lifter[1:n_fft//2] = 2
    return torch.exp(torch.fft.rfft(cepstrum*lifter, n=n_fft, dim=-1))


class Synthesizer(nn.Module):
    def __init__(self, harmonics=128):
        super().__init__()
        self.harmonics = harmonics
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.register_buffer("harmonic_matrix", band_matrix(HARMONIC_BANDS), persistent=False)
        self.register_buffer("noise_matrix", band_matrix(NOISE_BANDS), persistent=False)

    def forward(self, controls, seed=None, excitation=None, return_excitation=False):
        # FFT and phase accumulation remain float32 even during mixed precision training.
        controls = controls.float()
        batch, frames, _ = controls.shape
        samples = (frames-1)*HOP
        if samples < N_FFT:
            raise ValueError("Synthesis needs at least five frames")
        def upsample(values):
            return F.interpolate(values[:, None], size=samples, mode="linear", align_corners=True)[:, 0]
        f0 = upsample(controls[:, :, 0].clamp(math.log(55), math.log(750)).exp())
        voicing = upsample(controls[:, :, 1].sigmoid())
        energy = upsample(controls[:, :, 2].clamp(-12, 0).exp())
        phase = torch.cumsum(f0 / SAMPLE_RATE, dim=1) * (2*math.pi)
        # Chunk harmonics to bound intermediate memory for full-length utterances.
        pulse = torch.zeros_like(phase)
        harmonic_count = (SAMPLE_RATE/2/f0).floor().clamp(1, self.harmonics)
        for start in range(1, self.harmonics+1, 16):
            harmonic = torch.arange(start, min(start+16, self.harmonics+1), device=f0.device)
            angles = phase[:, :, None] * harmonic
            allowed = f0[:, :, None]*harmonic < SAMPLE_RATE/2
            pulse = pulse + (torch.cos(angles)*allowed).sum(-1)
        pulse = pulse * torch.sqrt(2/harmonic_count) * voicing
        generator = None
        if seed is not None:
            generator = torch.Generator(device=f0.device).manual_seed(seed)
        noise = torch.randn(batch, samples, device=f0.device, generator=generator)
        # Supplying both sources isolates the filter/OLA path in browser parity
        # tests. Normal inference/training always uses the sources above.
        if excitation is not None:
            pulse, noise = excitation
            if pulse.shape != (batch, samples) or noise.shape != pulse.shape:
                raise ValueError("Excitation must match the synthesized sample count")
        harmonic_log = controls[:, :, 3:43] @ self.harmonic_matrix.T
        noise_log = controls[:, :, 43:67] @ self.noise_matrix.T
        # Relative filter levels preserve harmonic/noise balance; energy sets total RMS.
        center = torch.maximum(harmonic_log.amax(-1, keepdim=True), noise_log.amax(-1, keepdim=True))
        harmonic_filter = minimum_phase((harmonic_log-center).clamp(-12, 0)).transpose(1, 2)
        noise_filter = minimum_phase((noise_log-center).clamp(-12, 0)).transpose(1, 2)
        def stft(wave):
            return torch.stft(wave, N_FFT, HOP, window=self.window, return_complex=True)
        spectrum = stft(pulse)*harmonic_filter + stft(noise)*noise_filter
        wave = torch.istft(spectrum, N_FFT, HOP, window=self.window, length=samples)
        rms = F.avg_pool1d(F.pad(wave[:, None].square(), (N_FFT//2, N_FFT//2), mode="reflect"),
                           N_FFT, HOP)[:, 0].clamp_min(1e-8).sqrt()
        # Smooth gain avoids dependence on arbitrary input amplitude calibration.
        output = wave * (energy / upsample(rms).clamp_min(1e-4)).clamp(max=100)
        return (output, pulse, noise) if return_excitation else output


def spectral_loss(prediction, target):
    losses = []
    for size in (256, 512, 1024, 2048):
        window = torch.hann_window(size, device=prediction.device)
        def magnitude(wave):
            return torch.stft(wave.float(), size, size//4, window=window,
                              return_complex=True).abs().clamp_min(1e-5)
        pred, ref = magnitude(prediction), magnitude(target)
        convergence = (pred-ref).norm(dim=(-2,-1)) / ref.norm(dim=(-2,-1)).clamp_min(1e-5)
        losses.append(convergence.mean() + (pred.log()-ref.log()).abs().mean())
    return torch.stack(losses).mean()
