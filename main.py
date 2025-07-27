from recommender.app import get_recommendations
from recommender.app import preprocess_data

def main():
    df = preprocess_data()
    movies = get_recommendations('The Dark Knight Rises', df)
    print(movies)

if __name__ == '__main__':
    main()

