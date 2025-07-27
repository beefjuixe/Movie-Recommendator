import pandas as pd
from pandas import DataFrame
from ast import literal_eval

def get_data_frame():
    df1 = pd.read_csv('data/tmdb_5000_credits.csv')
    df2 = pd.read_csv('data/tmdb_5000_movies.csv')
    df1.columns = ['id','title','cast','crew']
    df2 = df2.merge(df1, on='id')
    #df2 = df2.merge(df1, left_on='id', right_on='movie_id').drop(columns='movie_id')
    #df2['poster_path'] = df2['poster_path'].fillna('')
    df2['vote_average'] = df2['vote_average'].fillna('0')
    df2['release_date'] = df2['release_date'].fillna('')
    return df2

def convert_string_to_obj(pd: DataFrame, column:str):
    pd[column] = pd[column].apply(literal_eval)
    return pd