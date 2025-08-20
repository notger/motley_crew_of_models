"""Main file to govern the process flow of training and evaluating the Two-Tower model.

The separation of code here could be better, but this is a quick and dirty proof of concept,
not production-ready code, so please excuse occasional lack of abstraction.

Notes on nomenclature:
- The term embedding exists twice in our model: We use it when we convert an id to a scalar
  number to keep one-hot encoded matrices from exploding and we use it for the result of the
  coordinate transformation of said prior encodings (and other stuff) into the embedding space.
- We will try to stick to the following conventions for variable naming for anything which 
  directly touches the model (features, weights, outputs) ...
    (1) anything with "user" in it contains the movies watched and genres rated by the user
        or the average ratings of the user for given movie, homologically sorted to the 
        "movies" and "genre" feature vectors
    (2) anything with "movie" in it contains the one-hot encoded movie id
    (3) anything with "genre" in it contains the genres the movie belongs to
"""

import tqdm
import torch
import numpy as np

from two_tower_model.loader import *
from two_tower_model.features import *
from two_tower_model.model import *


# ================================== Parameters ==============================================
movie_embedding_size = 20
genre_embedding_size = 20
user_features_embedding_size = movie_embedding_size + genre_embedding_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_epochs = 10_000  # We will be running random batches from our data in each epoch.
batch_size = 100
log_every = 1000
loss_filter_window_length = 10  # We will log the last x losses to calculate a running mean.

# ================================== Data loading and prep ===================================
movies, ratings = load_raw_data(user_limit=1000, movie_limit=500)
genre_lookup, genres = generate_genre_lookup(movies)
lookup_movie_id_to_emb, lookup_emb_to_movie_id, lookup_genre_to_emb, lookup_emb_to_genre = generate_embeddings(movies, genres)
aggregated_user_scores = generate_aggregated_user_scores(ratings)

user_features, user_movie_ratings = generate_user_features(aggregated_user_scores, genre_lookup, lookup_genre_to_emb, lookup_movie_id_to_emb)
movie_features, genre_features = generate_movie_features(genre_lookup, lookup_genre_to_emb, lookup_movie_id_to_emb)

# Convert all input vectors into torch tensors:
user_features = torch.from_numpy(user_features).float()
user_movie_ratings = torch.from_numpy(user_movie_ratings).float()
movie_features = torch.from_numpy(movie_features).float()
genre_features = torch.from_numpy(genre_features).float()

# ================================== Model definition ========================================
mdl = TwoTowerModel(
    user_features, movie_features, genre_features,
    user_features_embedding_size, movie_embedding_size, genre_embedding_size
)

# ================================== Model training ==========================================
optimiser = torch.optim.SGD(mdl.parameters(), lr=0.001, momentum=0.9)
loss_fn = torch.nn.MSELoss()

def generate_batch(user_features, movie_features, genre_features, user_movie_ratings, batch_size):
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

filtered_loss = [1e6] * loss_filter_window_length
for epoch in tqdm.tqdm(range(1, N_epochs + 1)):
    # Generate the random batch:
    user_batch, move_batch, genre_batch, ratings_batch = generate_batch(
        user_features, movie_features, genre_features, user_movie_ratings, batch_size
    )

    output = mdl.forward(user_batch, move_batch, genre_batch)
    loss = loss_fn(output, ratings_batch)
    loss.backward()

    optimiser.step()

    filtered_loss = filtered_loss[1:] + [loss.item()]  # This could be more elegant, but we are doing so many heavy stuff here, this does not matter anymore.

    if epoch % log_every == 0:
        print(f"Epoch {epoch} loss: {np.mean(filtered_loss)}")

# ================================== Model evaluation ========================================