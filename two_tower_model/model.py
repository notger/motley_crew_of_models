"""Defines a simple Two-Tower-model.

The first tower takes the user features, containing the movies watched and the weighted genre rating
and projects this onto the embedding space.

The second tower takes the one-hot encoded movies as well as the one-hot encoded genres of the movies,
projects both and then concatenates it to eventually the same embedding space.
We are doing it like this to reduce the calculation complexity and RAM usage. This basically enforces
that the off diagonal embedding matrices for anything movie-related are zero. Of course, we could have
just done it differently.
"""

import torch


class TwoTowerModel(torch.nn.Module):
    def __init__(
        self,
        user_features: torch.Tensor, 
        movie_features: torch.Tensor, 
        genre_features: torch.Tensor,
        user_features_embedding_size: int,
        movie_embedding_size: int,
        genre_embedding_size: int,
    ):
        super(TwoTowerModel, self).__init__()
        self.Wu = torch.nn.Linear(user_features.shape[1], user_features_embedding_size)
        self.Wm = torch.nn.Linear(movie_features.shape[1], movie_embedding_size)
        self.Wg = torch.nn.Linear(genre_features.shape[1], genre_embedding_size)
        self.double()

    def forward(
        self,
        user_features: torch.Tensor, 
        movie_features: torch.Tensor, 
        genre_features: torch.Tensor,
    ):
        """Returns the result of a forward step.

        For training purposes, this should receive a mini-batch, for evaluation purposes,
        this can receive the complete dataset.

        Please note that this should receive torch Tensors, not pd.DataFrames!
        """
        z_first_tower = self.Wu(user_features)
        z_second_tower = torch.cat(
            (self.Wm(movie_features), self.Wg(genre_features)), dim=1,
        )

        return torch.einsum('ij, ij -> i', z_first_tower, z_second_tower)
        
    def train_step():
        pass