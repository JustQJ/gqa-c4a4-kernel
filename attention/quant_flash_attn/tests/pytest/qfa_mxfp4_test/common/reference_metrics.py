"""Diagnostic FP64 differences; these do not replace quantized-kernel acceptance."""
import torch


def tensor_gap(actual, reference):
    a, b = actual.double(), reference.double()
    if a.shape != b.shape or not torch.equal(torch.isposinf(a), torch.isposinf(b)):
        raise AssertionError('FP64 output shape or empty-row positions differ')
    finite = torch.isfinite(b)
    if not torch.isfinite(a[finite]).all():
        raise AssertionError('Nonfinite result at a finite FP64 reference position')
    a, b = a[finite].reshape(-1), b[finite].reshape(-1)
    if not a.numel():
        return dict(elements=0, max_abs=0., mean_abs=0., rmse=0., relative_l2=0., cosine=1.)
    diff = a-b
    norm_a, norm_b = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    cosine = float(torch.dot(a, b)/(norm_a*norm_b)) if norm_a > 0 and norm_b > 0 else float(norm_a == norm_b)
    return dict(elements=a.numel(), max_abs=diff.abs().max().item(), mean_abs=diff.abs().mean().item(),
                rmse=diff.square().mean().sqrt().item(), relative_l2=(torch.linalg.vector_norm(diff)/norm_b.clamp_min(1e-30)).item(),
                cosine=cosine)


def reference_gap(actual, reference):
    return dict(attention=tensor_gap(actual[0], reference[0]), lse=tensor_gap(actual[1], reference[1]))
