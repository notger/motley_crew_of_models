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


class TwoTowerModel:
    def __init__(
        user_features: torch.Tensor, 
        movie_features: torch.Tensor, 
        genre_features: torch.Tensor,
    ):
        self.W_user = torch.randn(user_features.shape[1], user_features_embedding_size)
        self.b_user = torch.randn(user_features_embedding_size)

        self.W_movies = torch.randn(movie_features.shape[1], movie_embedding_size)
        self.b_movies = torch.randn(movie_embedding_size)

        self.W_genres = torch.randn(genre_features.shape[1], genre_embedding_size)
        self.b_genres = torch.randn(genre_embedding_size)

        self.loss = torch.nn.MSELoss()

    def forward(
        user_features: torch.Tensor, 
        movie_features: torch.Tensor, 
        genre_features: torch.Tensor,
    ):
        """Returns the result of a forward step.

        For training purposes, this should receive a mini-batch, for evaluation purposes,
        this can receive the complete dataset.

        Please note that this should receive torch Tensors, not pd.DataFrames!
        """
        z_first_tower = torch.tanh(user_features @ self.W_user + self.b_user)

        z_second_tower = torch.cat(
            (
                torch.tanh(movie_features @ self.W_movies + self.b_movies),
                torch.tanh(genre_features @ self.W_genres + self.b_genres),
            ),
            dim=1,
        )

        return torch.einsum('ij, ij -> i', z_first_tower, z_second_tower)

    def calculate_loss(
        user_features: torch.Tensor,
        movie_features: torch.Tensor,
        genre_features: torch.Tensor,
        user_ratings: torch.Tensor,
    )
        """Wrapper to calculate the loss.

        Please note that in order to do that, it needs to receive the user ratings for a given movie.
        """
        z = self.forward(user_features, movie_features, genre_features)
        return self.loss(z, user_ratings)

    def train_step():
        pass
