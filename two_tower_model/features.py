import numpy as np
import pandas as pd

from collections import defaultdict


def generate_user_features(aggregated_user_scores: pd.DataFrame, genre_lookup: dict, lookup_genre_to_emb: dict) -> dict:
    user_features = {}
    N_features = max(lookup_genre_to_emb.values()) + 1
    
    # Generate the user's genre preferences:
    for row in aggregated_user_scores.itertuples():
        user_id = row.userId
        movies_voted = row.movieId
        ratings_voted = row.rating
    
        genre_ratings = defaultdict(list)
        
        # We first collect all ratings for a given genre, so that it is easier to calculate the average rating for this genre:
        for (movie_voted, rating) in zip(movies_voted, ratings_voted):
            for genre in genre_lookup[movie_voted]:
                genre_ratings[lookup_genre_to_emb[genre]].append(rating)
    
        # Now calculate the average, but relative to the average rating of that user:
        genre_ratings = {g: np.mean(r) - row.avg for g, r in genre_ratings.items()}
    
        # Then, we want to rescale such that the most liked genre gets a feature value of 1 and the most disliked one of -1.
        # For this, we will have to stretch positive and negative values differently:
        max_rating = max(genre_ratings.values())
        min_rating = min(genre_ratings.values())
    
        # We have to watch out for users which have always given the same ratings score (yes, there is one).
        # In order to still be able to recommend a movie to this user, we will assign them their normalised average
        # score to all genres they voted on:
        if max_rating > 0 and min_rating < 0:
            genre_ratings = {g: r / max_rating if r >= 0 else -r / min_rating for g, r in genre_ratings.items()}
        else:
            genre_ratings = {g: (row.avg - 2.5) / 2.5 for g, r in genre_ratings.items()}
    
        # Finally, for each user, we will create a feature vector which contains their genre-ratings.
        feature = np.zeros(N_features)
    
        for g, r in genre_ratings.items():
            feature[g] = r
    
        user_features[user_id] = np.array(feature, copy=True)

    return user_features