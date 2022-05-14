"""Contains pytests for the Transformer modules."""

import pytest

import torch
from transformer import Transformer


@pytest.mark.parametrize('N_encoder_layer', [1, 4])
@pytest.mark.parametrize('N_decoder_layer', [1, 6])
@pytest.mark.parametrize('dim_mdl', [8, 16])
@pytest.mark.parametrize('N_heads', [1, 3])
@pytest.mark.parametrize('dim_ff', [128, 256])
def test_basic_setup(
    N_encoder_layer: int,
    N_decoder_layer: int,
    dim_mdl: int,
    N_heads: int,
    dim_ff: int,
):
    _ = Transformer(
        N_encoder_layer=N_encoder_layer,
        N_decoder_layer=N_decoder_layer,
        dim_mdl=dim_mdl,
        N_heads=N_heads,
        dim_ff=dim_ff,
    )


@pytest.mark.parametrize('N_encoder_layer', [1, 4])
@pytest.mark.parametrize('N_decoder_layer', [1, 6])
@pytest.mark.parametrize('dim_mdl', [8, 16])
@pytest.mark.parametrize('N_heads', [1, 3])
@pytest.mark.parametrize('dim_ff', [128, 256])
@pytest.mark.parametrize('batch_size', [1, 11])
@pytest.mark.parametrize('x_len', [2, 12])
@pytest.mark.parametrize('y_len', [2, 13])
def test_input_output_dimensions(
    N_encoder_layer: int,
    N_decoder_layer: int,
    dim_mdl: int,
    N_heads: int,
    dim_ff: int,
    batch_size: int,
    x_len: int,
    y_len: int,
):
    transformer = Transformer(
        N_encoder_layer=N_encoder_layer,
        N_decoder_layer=N_decoder_layer,
        dim_mdl=dim_mdl,
        N_heads=N_heads,
        dim_ff=dim_ff,
    )

    x = torch.randn(batch_size, x_len, dim_mdl)
    y = torch.randn(batch_size, y_len, dim_mdl)

    out = transformer(x, y)

    assert batch_size == out.shape[0]
    assert y_len == out.shape[1]
    assert dim_mdl == out.shape[2]
