import torch
from torch.nn import functional as F
from xiaotts.dsp import reflect_pad, linear_upsample


def test_deterministic_replacements_match_forward_and_cpu_gradient():
    torch.manual_seed(22)
    for length, padding in ((7, 3), (31, 12)):
        values = torch.randn(2, length, requires_grad=True)
        a = reflect_pad(values, padding)
        b = F.pad(values[:, None], (padding,padding), mode='reflect')[:, 0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        weights = torch.randn_like(a)
        ga = torch.autograd.grad((a*weights).sum(), values)[0]
        gb = torch.autograd.grad((b*weights).sum(), values)[0]
        torch.testing.assert_close(ga,gb)
        a = linear_upsample(values, 113)
        b = F.interpolate(values[:, None], size=113, mode='linear', align_corners=True)[:,0]
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        weights = torch.randn_like(a)
        ga = torch.autograd.grad((a*weights).sum(), values)[0]
        gb = torch.autograd.grad((b*weights).sum(), values)[0]
        torch.testing.assert_close(ga,gb,atol=2e-5,rtol=2e-5)
