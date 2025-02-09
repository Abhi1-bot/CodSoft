import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings_data = {
    'User': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'],
    'Movie': ['Titanic', 'Avatar', 'Titanic', 'Inception', 'Avatar', 'Joker', 'Titanic', 'Joker', 'Avatar', 'Inception'],
    'Rating': [5, 4, 5, 4, 3, 5, 3, 4, 2, 3]
}

ratings_df = pd.DataFrame(ratings_data)
movie_ratings = ratings_df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)
user_sim = cosine_similarity(movie_ratings)
user_similarity_df = pd.DataFrame(user_sim, index=movie_ratings.index, columns=movie_ratings.index)

def recommend_movie(user):
    if user not in movie_ratings.index:
        return "User not found."
    else:
        similar_users = user_similarity_df[user].drop(index=user)
        most_similar_user = similar_users.idxmax()
        
        if most_similar_user not in movie_ratings.index:
            return "No similar user found."
        else:
            unseen_movies = movie_ratings.loc[most_similar_user][movie_ratings.loc[user] == 0]
            
            if unseen_movies.empty:
                return "No new recommendations."
            else:
                return unseen_movies.sort_values(ascending=False).index.tolist()

user_to_recommend = 'A'
print(f"Recommended movies for {user_to_recommend}: {recommend_movie(user_to_recommend)}")