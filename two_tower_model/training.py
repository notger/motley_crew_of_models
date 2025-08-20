import torch
import numpy as np


def generate_batch(
    user_features: torch.Tensor, 
    movie_features: torch.Tensor, 
    genre_features: torch.Tensor,
    user_movie_ratings: torch.Tensor, 
    batch_size: int
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    # Generate a random batch of users:
    user_indices = torch.randint(0, user_features.shape[0], (batch_size,))
    X_user = user_features[user_indices, :]
    
    # Generate a random batch of movies and genres:
    movie_indices = torch.randint(0, movie_features.shape[0], (batch_size,))
    X_movies = movie_features[movie_indices, :]
    X_genres = genre_features[movie_indices, :]

    # Extract user ratings for the batch of users and the batch of movies generated:
    X_ratings = np.zeros((batch_size,))
    for ui, mi in enumerate(movie_indices):
        X_ratings[ui] = user_movie_ratings[ui, mi]
    X_ratings = torch.from_numpy(X_ratings)

    return X_user, X_movies, X_genres, X_ratings
