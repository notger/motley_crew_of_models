"""Defines the positional encoding as per the original paper.

We do it in a separate file, as that encoding is used in the encoder
and the decoder and everyone loves DNRY.
"""

import torch


def positional_encoding(
    sequence_length: int,
    dim_mdl: int,
    device: torch.device = torch.device('cpu'),
):
    # Generate the batch-wise row-like representation of the position
    # and the column-like representation of the dimension,
    # then calculate the angle between both:
    pos = torch.arange(sequence_length, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_mdl, dtype=torch.float, device=device).reshape(1, 1, -1)
    angle = pos / (1e4 ** torch.div(dim, dim_mdl, rounding_mode="floor"))

    # Now return the sine of the angle for even indices and the cosine for odd indices.
    # Note that torch.where gives you an if/else-like behaviour out of the box.
    # A slightly more conventional way would have been a list comprehension, but
    # torch.where gives you a Tensor back, so we save us a conversion in the code.
    return torch.where(
        dim.long() % 2 == 0, 
        torch.sin(angle), 
        torch.cos(angle)
    )
