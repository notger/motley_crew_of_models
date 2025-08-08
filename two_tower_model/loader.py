"""Module containing utility functions for loading and light pre-processing of the MovieLens dataset.

The most important feature here is that you can specify how many users and movies you want to load, in case your memory is not sufficient to load the entire dataset.

Please see the README.md file for more information on how to use this module.
"""

import pandas as pd

def load_raw_data(user_limit=None, movie_limit=None):
    """Load raw MovieLens data with optional limits on users and movies.

    Args:
        user_limit (int, optional): Maximum number of users to load. If None, all users are loaded.
        movie_limit (int, optional): Maximum number of movies to load. If None, all movies are loaded.

    Returns:
        tuple: A tuple containing the user data and movie data.
    """
    # Implementation of data loading logic goes here
    movies = pd.read_csv('two_tower_model/data/ml-32m/movies.csv')
    ratings = pd.read_csv('two_tower_model/data/ml-32m/ratings.csv')

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

    return movies, ratings


if __name__ == "__main__":
    # Example usage, also useful to test that stuff works:
    movies, ratings = load_raw_data(user_limit=1000, movie_limit=500)
    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings from {len(ratings.userId.unique())} distinct users.")
