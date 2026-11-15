import torch
from xiaotts.model import XiaoTTS, ModelConfig, expand_durations
from xiaotts.dsp import Synthesizer, spectral_loss


def test_five_languages_and_tones_train_through_duration_expansion():
    torch.set_num_threads(2)
    torch.manual_seed(4)
    model = XiaoTTS().eval()
    assert model.parameter_count == 757828
    tokens = torch.tensor([[2, 3, 4, 5, 0]]).repeat(5, 1)
    tones = torch.tensor([[0, 1, 3, 4, 0]]).repeat(5, 1)
    stress = torch.tensor([[0, 1, 0, 2, 0]]).repeat(5, 1)
    indices, positions = expand_durations(torch.tensor([2, 0, 3, 4, 0]))
    duration, controls = model(tokens, torch.arange(5), tones, stress,
                               indices[None].repeat(5, 1), positions[None].repeat(5, 1, 1),
                               torch.ones(5, len(indices)))
    assert controls.shape == (5, 9, 67)
    assert torch.isfinite(controls).all()
    (controls.square().mean()+duration.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not torch.equal(controls[0], controls[1])
    changed, _ = model.encode(tokens, torch.arange(5), torch.zeros_like(tones), stress)
    original, _ = model.encode(tokens, torch.arange(5), tones, stress)
    assert not torch.equal(original, changed)
    short, _ = model.encode(tokens[:, :4], torch.arange(5), tones[:, :4], stress[:, :4])
    torch.testing.assert_close(original[:, :4], short, atol=1e-5, rtol=1e-5)


def test_renderer_has_no_learned_parameters_and_finite_gradients():
    torch.set_num_threads(2)
    controls = torch.zeros(1, 17, 67)
    controls[:, :, 0] = 5.2
    controls[:, :, 1] = 4
    controls[:, :, 2] = -3
    controls[:, :, 43:] = -4
    controls.requires_grad_()
    renderer = Synthesizer(harmonics=32)
    assert sum(p.numel() for p in renderer.parameters()) == 0
    audio = renderer(controls, seed=7)
    assert audio.shape == (1, 4096)
    assert torch.isfinite(audio).all() and 0.01 < audio.square().mean().sqrt() < 0.2
    target = 0.05*torch.sin(torch.arange(4096)[None]*0.05)
    spectral_loss(audio, target).backward()
    assert torch.isfinite(controls.grad).all()
    assert controls.grad[:, :, 0].abs().sum() > 0
    assert controls.grad[:, :, 3:].abs().sum() > 0
