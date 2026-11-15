import torch
from xiaotts.train import acoustic_loss


def test_padding_cannot_improve_or_worsen_acoustic_scores():
    duration = torch.tensor([[1.0, 2.0, 50.0]], requires_grad=True)
    controls = torch.zeros(1, 8, 67, requires_grad=True)
    target = torch.zeros_like(controls)
    target[:, :5, 0] = 5.2
    target[:, :5, 1] = 8
    batch = dict(tokens=torch.tensor([[3, 4, 0]]), durations=torch.tensor([[2, 3, 0]]),
                 controls=target, mask=torch.tensor([[1,1,1,1,1,0,0,0]], dtype=torch.float32))
    initial, _ = acoustic_loss(duration, controls, batch)
    changed = controls.detach().clone()
    changed[:, 5:] = 1000
    other, _ = acoustic_loss(duration, changed, batch)
    torch.testing.assert_close(initial, other)
    initial.backward()
    assert duration.grad[0, -1] == 0
    assert controls.grad[:, 5:].abs().sum() == 0
