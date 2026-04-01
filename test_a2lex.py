"""Correctness tests for the A2Lex (A2 hexagonal lattice FSQ) tokenizer."""

import math
import sys
import torch

sys.path.insert(0, "/workspace/oat")

from oat.tokenizer.oat.quantizer.fsq import FSQ
from oat.tokenizer.a2lex.quantizer.a2_lex_fsq import A2LexFSQ, SQRT3_OVER_2


def test_codebook_size_preservation():
    """Test 1: Codebook size is identical between FSQ and A2LexFSQ."""
    print("Test 1: Codebook size preservation ... ", end="")
    for levels in [[8, 5, 5, 5], [3, 3, 3], [8, 8], [5, 5, 5, 5, 5]]:
        fsq = FSQ(levels=levels)
        a2 = A2LexFSQ(levels=levels)
        assert fsq.codebook_size == a2.codebook_size, (
            f"Codebook size mismatch for levels={levels}: "
            f"FSQ={fsq.codebook_size}, A2={a2.codebook_size}"
        )
    print("PASSED")


def test_hex_rounding_integer():
    """Test 2: Hex rounding returns valid integer lattice coords."""
    print("Test 2: Hex rounding integer ... ", end="")
    a2 = A2LexFSQ(levels=[8, 5, 5, 5])
    u = torch.randn(100)
    v = torch.randn(100)
    u_r, v_r = a2._hex_round_pair(u, v)
    assert torch.all(u_r == u_r.round()), "u_r must be integers"
    assert torch.all(v_r == v_r.round()), "v_r must be integers"
    print("PASSED")


def test_hex_rounding_optimality():
    """Test 3: Hex rounding is closer than independent rounding in A2 metric."""
    print("Test 3: Hex rounding optimality ... ", end="")
    a2 = A2LexFSQ(levels=[8, 5, 5, 5])

    u_cont = torch.randn(10000) * 3
    v_cont = torch.randn(10000) * 3
    u_hex, v_hex = a2._hex_round_pair(u_cont, v_cont)
    u_ind, v_ind = u_cont.round(), v_cont.round()

    def a2_dist(du, dv):
        return du ** 2 + du * dv + dv ** 2

    d_hex = a2_dist(u_cont - u_hex, v_cont - v_hex)
    d_ind = a2_dist(u_cont - u_ind, v_cont - v_ind)

    assert torch.all(d_hex <= d_ind + 1e-6), "Hex rounding must be optimal in A2 metric"
    differs = ~torch.isclose(u_hex, u_ind) | ~torch.isclose(v_hex, v_ind)
    if differs.any():
        assert (d_hex[differs] < d_ind[differs] - 1e-6).any(), \
            "Hex should beat independent somewhere"
    print("PASSED")


def test_index_roundtrip():
    """Test 4: Index roundtrip (forward_z -> indices_to_embedding)."""
    print("Test 4: Index roundtrip ... ", end="")
    for levels in [[8, 5, 5, 5], [3, 3, 3], [8, 8], [5, 5, 5, 5, 5]]:
        a2 = A2LexFSQ(levels=levels)
        a2.eval()
        z = torch.randn(4, 8, len(levels))
        quant, tokens = a2.forward_z(z)
        reconstructed = a2.indices_to_embedding(tokens)
        assert torch.allclose(quant, reconstructed, atol=1e-5), (
            f"indices_to_embedding must recover exact quantized Cartesian values "
            f"for levels={levels}"
        )
    print("PASSED")


def test_cartesian_lattice_roundtrip():
    """Test 5: Cartesian <-> lattice roundtrip."""
    print("Test 5: Cartesian-lattice roundtrip ... ", end="")
    a2 = A2LexFSQ(levels=[8, 5, 5, 5])
    nonneg = torch.tensor([[0, 0, 2, 3], [7, 4, 4, 4]], dtype=torch.float)
    cart = a2._to_cartesian(nonneg)
    nonneg_back = a2._cartesian_to_lattice_nonneg(cart)
    assert torch.allclose(nonneg, nonneg_back, atol=1e-5), \
        f"Roundtrip failed: {nonneg} vs {nonneg_back}"
    print("PASSED")


def test_hexagonal_pattern():
    """Test 6: Cartesian values form hexagonal pattern."""
    print("Test 6: Hexagonal pattern ... ", end="")
    a2_small = A2LexFSQ(levels=[5, 5])
    all_indices = torch.arange(25)
    embeddings = a2_small.indices_to_embedding(all_indices)  # (25, 2)
    cart = a2_small._denormalize_cartesian(embeddings)
    # Index 5 -> lattice (0,1) -> Cartesian (0.5, sqrt3/2)
    # lex: pair_idx = a + b * L0 = 0 + 1 * 5 = 5
    assert torch.isclose(cart[5, 0], torch.tensor(0.5), atol=1e-5), \
        f"Expected x=0.5, got {cart[5, 0].item()}"
    assert torch.isclose(cart[5, 1], torch.tensor(SQRT3_OVER_2), atol=1e-5), \
        f"Expected y={SQRT3_OVER_2}, got {cart[5, 1].item()}"
    # Index 0 -> lattice (0,0) -> Cartesian (0, 0)
    assert torch.isclose(cart[0, 0], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(cart[0, 1], torch.tensor(0.0), atol=1e-5)
    # Index 1 -> lattice (1,0) -> Cartesian (1, 0)
    assert torch.isclose(cart[1, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(cart[1, 1], torch.tensor(0.0), atol=1e-5)
    print("PASSED")


def test_embedding_differs_from_fsq():
    """Test 7: A2 codebook differs from FSQ codebook."""
    print("Test 7: Embedding differs from FSQ ... ", end="")
    fsq = FSQ(levels=[5, 5])
    a2 = A2LexFSQ(levels=[5, 5])
    fsq_cb = fsq.implicit_codebook
    a2_cb = a2.implicit_codebook
    assert not torch.allclose(fsq_cb, a2_cb), "A2 codebook must differ from FSQ codebook"
    print("PASSED")


def test_all_indices_covered():
    """Test 8: Every index in [0, codebook_size) is reachable."""
    print("Test 8: All indices covered ... ", end="")
    for levels in [[8, 5, 5, 5], [3, 3, 3]]:
        a2 = A2LexFSQ(levels=levels)
        all_indices = torch.arange(a2.codebook_size)
        embeddings = a2.indices_to_embedding(all_indices)
        recovered = a2.codes_to_indices(embeddings)
        assert torch.equal(recovered.long(), all_indices.long()), \
            f"Not all indices round-trip correctly for levels={levels}"
    print("PASSED")


def test_gradient_flow():
    """Test 9: Gradient flows through encoder output."""
    print("Test 9: Gradient flow ... ", end="")
    a2 = A2LexFSQ(levels=[8, 5, 5, 5])
    a2.eval()
    z = torch.randn(2, 4, 4, requires_grad=True)
    quant, tokens = a2.forward_z(z)
    loss = quant.sum()
    loss.backward()
    assert z.grad is not None, "Gradient must flow to encoder output"
    assert not torch.all(z.grad == 0), "Gradient must be non-zero"
    print("PASSED")


def test_odd_dimensions():
    """Test 10: Odd number of dimensions (remainder dim)."""
    print("Test 10: Odd dimensions (remainder) ... ", end="")
    levels = [3, 3, 3]
    a2 = A2LexFSQ(levels=levels)
    a2.eval()
    assert a2.has_remainder
    assert a2.num_pairs == 1
    assert a2.codebook_size == 27

    z = torch.randn(2, 4, 3)
    quant, tokens = a2.forward_z(z)
    reconstructed = a2.indices_to_embedding(tokens)
    assert torch.allclose(quant, reconstructed, atol=1e-5), \
        "Roundtrip failed for odd dimensions"
    print("PASSED")


def test_implicit_codebook_shape():
    """Test 11: Implicit codebook shape."""
    print("Test 11: Implicit codebook shape ... ", end="")
    a2 = A2LexFSQ(levels=[8, 5, 5, 5])
    assert a2.implicit_codebook.shape == (1000, 4), \
        f"Expected (1000, 4), got {a2.implicit_codebook.shape}"
    print("PASSED")


if __name__ == "__main__":
    test_codebook_size_preservation()
    test_hex_rounding_integer()
    test_hex_rounding_optimality()
    test_index_roundtrip()
    test_cartesian_lattice_roundtrip()
    test_hexagonal_pattern()
    test_embedding_differs_from_fsq()
    test_all_indices_covered()
    test_gradient_flow()
    test_odd_dimensions()
    test_implicit_codebook_shape()
    print("\nAll tests passed!")
