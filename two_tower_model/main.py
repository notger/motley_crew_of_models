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

import torch
import pandas as pd

from two_tower_model.loader import *
from two_tower_model.features import *
from two_tower_model.model import *


# ================================== Parameters ==============================================
movie_embedding_size = 20
genre_embedding_size = 20
user_features_embedding_size = movie_embedding_size + genre_embedding_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================================== Data loading and prep ===================================
movies, ratings = load_raw_data(user_limit=1000, movie_limit=500)
genre_lookup, genres = generate_genre_lookup(movies)
lookup_movie_id_to_emb, lookup_emb_to_movie_id, lookup_genre_to_emb, lookup_emb_to_genre = generate_embeddings(movies, genres)
aggregated_user_scores = generate_aggregated_user_scores(ratings)

user_features, user_movie_ratings = generate_user_features(aggregated_user_scores, genre_lookup, lookup_genre_to_emb)
movie_features, genre_features = generate_movie_features(genre_lookup, lookup_genre_to_emb, lookup_movie_id_to_emb)

# Convert all input vectors into torch tensors:
raise NotImplementedError()

# ================================== Model definition ========================================
mdl = TwoTowerModel(user_features, movie_features, genre_features)

# ================================== Model training ==========================================




# ================================== Model evaluation ========================================