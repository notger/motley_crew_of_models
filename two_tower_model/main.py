"""Main file to govern the process flow of training and evaluating the Two-Tower model.

The separation of code here could be better, but this is a quick and dirty proof of concept,
not production-ready code, so please excuse occasional lack of abstraction.
"""

import torch
import pandas as pd

from loader import *
from features import *


# ================================== Data loading and prep ===================================
movies, ratings = load_raw_data(user_limit=1000, movie_limit=500)
genre_lookup, genres = generate_genre_lookup(movies)
lookup_movie_id_to_emb, lookup_emb_to_movie_id, lookup_genre_to_emb, lookup_emb_to_genre = generate_embeddings(movies, genres)
aggregated_user_scores = generate_aggregated_user_scores(ratings)

user_features = generate_user_features(aggregated_user_scores, lookup_genre_to_emb)


# ================================== Model definition ========================================




# ================================== Model training ==========================================




# ================================== Model evaluation ========================================