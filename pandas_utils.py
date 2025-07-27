import pandas as pd
from pandas import DataFrame

def get_data_frame():
    df1 = pd.read_csv('movie_recommender/data/tmdb_5000_credits.csv')
    df2 = pd.read_csv('movie_recommender/data/tmdb_5000_movies.csv')
    df1.columns = ['id','title','cast','crew']
    df2 = df2.merge(df1, left_on='id', right_on='movie_id').drop(columns='movie_id')
    df2['poster_path'] = df2['poster_path'].fillna('')
    df2['vote_average'] = df2['vote_average'].fillna('0')
    df2['release_date'] = df2['release_date'].fillna('')
    return df2
