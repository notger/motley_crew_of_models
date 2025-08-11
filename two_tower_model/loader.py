"""Module containing utility functions for loading and light pre-processing of the MovieLens dataset.

The most important feature here is that you can specify how many users and movies you want to load, in case your memory is not sufficient to load the entire dataset.

Please see the README.md file for more information on how to use this module.
"""

import pandas as pd

def load_raw_data(user_limit: int=None, movie_limit: int=None, path: str='two_tower_model/data/ml-32m/') -> tuple[dict, dict]:
    """Load raw MovieLens data with optional limits on users and movies.

    Args:
        user_limit (int, optional): Maximum number of users to load. If None, all users are loaded.
        movie_limit (int, optional): Maximum number of movies to load. If None, all movies are loaded.

    Returns:
        tuple: A tuple containing the user data and movie data.
    """
    # Implementation of data loading logic goes here
    movies = pd.read_csv(path + 'movies.csv')
    ratings = pd.read_csv(path + 'ratings.csv')

    # If we either have to apply a limit on the users or the movies, we need statistics to filter 
    # the most important ones. What is important to know here is that if we filter out movies, then
    # we also need to filter out the respective ratings, and vice versa. This means that the data
    # which remains will be the union of ratings and movies, which are both amongst the most prolific
    # users and most frequently rated movies.
    if user_limit or movie_limit:
        # Get the top users based on the number of ratings
        user_counts = ratings['userId'].value_counts()
        if user_limit:
            top_users = user_counts.nlargest(user_limit).index
            ratings = ratings[ratings['userId'].isin(top_users)]

        # Get the top movies based on the number of ratings
        movie_counts = ratings['movieId'].value_counts()
        if movie_limit:
            top_movies = movie_counts.nlargest(movie_limit).index
            ratings = ratings[ratings['movieId'].isin(top_movies)]
            movies = movies[movies['movieId'].isin(top_movies)]

    # Add a column with the number of votes to the movies-DataFrame:
    movies = movies.join(ratings.groupby('movieId').rating.count()).set_index('movieId', drop=True)

    return movies, ratings


def generate_genre_lookup(movies: pd.DataFrame) -> tuple[dict, list]:
    """We want a lookup which maps a given movie (by id) to its genres."""
    lookup_movie_id_to_genres = {row.Index: row.genres.split('|') for row in movies.itertuples()}
    genres = set([genre for genre_list in lookup_movie_id_to_genres.values() for genre in genre_list])
    return lookup_movie_id_to_genres, genres


def generate_embeddings(movies: pd.DataFrame, genres: list) -> tuple[dict, dict, dict, dict]:
    """Generates the embeddings and the reverse lookup for those embeddings as dicts for the movies and the genres."""
    lookup_movie_id_to_emb = {movie_id: k for k, movie_id in enumerate(movies.index.unique())}
    lookup_emb_to_movie_id = {v: k for k, v in lookup_movie_id_to_emb.items()}

    lookup_genre_to_emb = {genre: k for k, genre in enumerate(genres)}
    lookup_emb_to_genre = {v: k for k, v in lookup_genre_to_emb.items()}

    return lookup_movie_id_to_emb, lookup_emb_to_movie_id, lookup_genre_to_emb, lookup_emb_to_genre


def generate_aggregated_user_scores(ratings: pd.DataFrame) -> pd.DataFrame:
    """Generate a DataFrame which contains the list of movies and a list with the assorted ratings, per user-id.
    
    Could also have been a regular dict, but if we have things as DataFrame, then doing it like this is quicker.
    """
    return ratings.groupby('userId').agg({'movieId': lambda x: list(x), 'rating': lambda y: list(y)}).reset_index()


if __name__ == "__main__":
    # Example usage, also useful to test that stuff works:
    movies, ratings = load_raw_data(user_limit=1000, movie_limit=500)
    genre_lookup, genres = generate_genre_lookup(movies)
    lookup_movie_id_to_emb, lookup_emb_to_movie_id, lookup_genre_to_emb, lookup_emb_to_genre = generate_embeddings(movies, genres)
    aggregated_user_scores = generate_aggregated_user_scores(ratings)

    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings from {len(ratings.userId.unique())} distinct users.")
    print()
    print(movies.head(3))
    print()
    print(ratings.head(3))
    print()
    print(aggregated_user_scores.head(3))

