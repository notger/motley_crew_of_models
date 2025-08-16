"""Functions to generate user and movie features.

IMPORTANT: It is imperative that the user features and the movie features share the same length
AND are ordered in the same way. Our convention here will be: Features will consist of (N + M)
entries, where N is then number of movies (one-hot encoded) and M is the number
of genres.

The genres for the user feature vector will contain the weighted ratings the user gave on movies
they had watched. For movies, this part will be one-hot encoded as well, obviously.

Please note that this way of doing things assumes that the users have enough ratings to result in
a meaningful averaged genre score. I.e. their ratings are also sufficiently diverse. A user who only
watches fantasy action movies but actually only likes the fantasy part will inadvertedly be seen to
also like the action part and thus be recommended action movies as well.
"""

import numpy as np
import pandas as pd

from collections import defaultdict


def generate_user_features(aggregated_user_scores: pd.DataFrame, genre_lookup: dict, lookup_genre_to_emb: dict, lookup_movie_id_to_emb: dict) -> pd.DataFrame:
    user_features = {}
    N_features_genres = max(lookup_genre_to_emb.values()) + 1
    N_features_movies = max(lookup_movie_id_to_emb.values()) + 1
    
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
        feature_genres = np.zeros(N_features_genres)
    
        for g, r in genre_ratings.items():
            feature_genres[g] = r

        # Create the one-hot-encoded vector for the movies watched:
        feature_movies_watched = np.zeros(N_features_movies)

        for movie_voted in movies_voted:
            feature_movies_watched[lookup_movie_id_to_emb[movie_voted]] = 1.0
    
        # Now bring it all together:
        user_features[user_id] = np.concatenate(
            (np.array(feature_movies_watched, copy=True), np.array(feature_genres, copy=True),)
        )

    # We now have a dict of user_id to features, but we have to transform this dict into a DataFrame
    # where each user comprises a row:
    return pd.DataFrame.from_dict(user_features, orient='index')


def generate_movie_features(genre_lookup: dict, lookup_genre_to_emb: dict, lookup_movie_id_to_emb: dict) -> pd.DataFrame:
    # This follows the same logic for the user feature function above, except since we are doing
    # one-hot encoding here and don't need to calculate genre rating averages, we can directly
    # generate the parts and then glue them together.
    N_movies = max(lookup_movie_id_to_emb.values()) + 1

    feature_movie_one_hot = np.zeros((N_movies, N_movies))
    feature_genres = np.zeros((N_movies, max(lookup_genre_to_emb.values()) + 1))

    for k, (movie_id, genres) in enumerate(genre_lookup.items()):
        feature_movie_one_hot[k, lookup_movie_id_to_emb[movie_id]] = 1.0
        for genre in genres:
            feature_genres[k, lookup_genre_to_emb[genre]] = 1.0

    return np.concatenate((feature_movie_one_hot, feature_genres), axis=1)
