# %% Run test
import torch
from ckatorch import cka_base


# def cka_base(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#     """
#     Example linear CKA implementation.
#     Assumes:
#         a: (n_samples, n_features_a)
#         b: (n_samples, n_features_b)
#     Returns:
#         scalar tensor
#     """
#     if a.ndim != 2 or b.ndim != 2:
#         raise ValueError("a and b must both be 2D tensors")
#     if a.shape[0] != b.shape[0]:
#         raise ValueError("a and b must have the same number of samples")

#     a = a - a.mean(dim=0, keepdim=True)
#     b = b - b.mean(dim=0, keepdim=True)

#     hsic = torch.norm(a.T @ b, p="fro") ** 2
#     var1 = torch.norm(a.T @ a, p="fro")
#     var2 = torch.norm(b.T @ b, p="fro")

#     denom = var1 * var2
#     if denom <= eps:
#         return torch.tensor(float("nan"), device=a.device, dtype=a.dtype)

#     return hsic / denom


def run_cka_biased_unbiased_behavior_checks(
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> bool:
    """Behavior of biased vs unbiased linear CKA (same API, two flags)—no alternate implementation.

    Uses standard facts: (1) finite-sample Gram corrections differ, so small-n gaps are large;
    (2) both estimators track the same population quantity, so gaps shrink as n grows;
    (3) identity CKA(A, A) is 1 for either centering.
    """

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype)

    def check(name: str, value, passed: bool):
        status = "PASS" if passed else "FAIL"
        if verbose:
            print(f"[{status}] {name}: {value}")
        return passed

    all_passed = True

    def gap(a: torch.Tensor, b: torch.Tensor):
        ret1 = cka_base(a, b, unbiased=False)
        ret2 = cka_base(a, b, unbiased=True)
        print(f"{ret1=}\n{ret2=}")
        return (ret1 - ret2).abs()

    # 1. Small n, i.i.d. Gaussian: biased vs unbiased differ substantially (finite-sample correction)
    torch.manual_seed(123)
    n_small, d = 30, 24
    A = randn(n_small, d)
    B = randn(n_small, d)
    g = gap(A, B)
    all_passed &= check(
        "small n: |biased - unbiased| is large (i.i.d. Gaussian)",
        float(g),
        (g > 0.15).item(),
    )

    # 2. Large n, same kind of draw: the two modes nearly agree (asymptotic / large-sample behavior)
    torch.manual_seed(456)
    n_large = 5000
    A = randn(n_large, 32)
    B = randn(n_large, 32)
    g = gap(A, B)
    all_passed &= check(
        "large n: |biased - unbiased| is small (same DGP as case 1 spirit)",
        float(g),
        (g < 0.02).item(),
    )

    # 3. Identity: both modes return ~1 (centering mode should not break self-similarity)
    torch.manual_seed(0)
    A = randn(80, 40)
    b_self = cka_base(A, A, unbiased=False)
    u_self = cka_base(A, A, unbiased=True)
    one = torch.tensor(1.0, device=device, dtype=dtype)
    ok_id = torch.isclose(b_self, one, atol=1e-8, rtol=1e-5).item() and torch.isclose(
        u_self, one, atol=1e-8, rtol=1e-5
    ).item()
    all_passed &= check(
        "identity: biased and unbiased both ~= 1 for CKA(A, A)",
        (float(b_self), float(u_self)),
        ok_id,
    )

    # 4. Strong alignment vs independence: both modes rank signal over noise (same qualitative story)
    torch.manual_seed(7)
    n, d = 120, 32
    A = randn(n, d)
    B_signal = A + 0.02 * randn(n, d)
    B_indep = randn(n, d)
    for label, unbiased in [("biased", False), ("unbiased", True)]:
        s_align = cka_base(A, B_signal, unbiased=unbiased)
        s_indep = cka_base(A, B_indep, unbiased=unbiased)
        all_passed &= check(
            f"signal > independent noise ({label})",
            (float(s_align), float(s_indep)),
            (s_align > s_indep + 0.1).item(),
        )

    if verbose:
        print()
        print("BIASED vs UNBIASED BEHAVIOR:", "PASS" if all_passed else "FAIL")

    return all_passed



run_cka_biased_unbiased_behavior_checks()