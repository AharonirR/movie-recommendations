import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load MovieLens dataset
ratings = pd.read_csv('MovieRecommendations.csv')
movies = pd.read_csv('movieIdTitles.csv')

# Prepare data for Surprise library
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split data into training and test sets
trainset, testset = train_test_split(data, test_size=0.25)

# Build collaborative filtering model using SVD
algo = SVD()
algo.fit(trainset)

# Evaluate the model
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Function to get movie recommendations for a specific user
def get_movie_recommendations(user_id, num_recommendations=10):
    # Get a list of all movie IDs
    all_movie_ids = movies['movieId'].unique()
    
    # Get a list of movie IDs that the user has already rated
    rated_movie_ids = ratings[ratings['userId'] == user_id]['movieId'].unique()
    
    # Get a list of movie IDs that the user has not rated yet
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
    
    # Predict ratings for the unrated movies
    predicted_ratings = [algo.predict(user_id, movie_id).est for movie_id in unrated_movie_ids]
    
    # Create a DataFrame with movie IDs and predicted ratings
    recommendations = pd.DataFrame({'movieId': unrated_movie_ids, 'predicted_rating': predicted_ratings})
    
    # Sort the DataFrame by predicted rating in descending order
    recommendations = recommendations.sort_values(by='predicted_rating', ascending=False)
    
    # Get the top N movie recommendations
    top_recommendations = recommendations.head(num_recommendations)
    
    # Merge with the movies DataFrame to get movie titles
    top_recommendations = top_recommendations.merge(movies, on='movieId', how='inner')
    
    return top_recommendations[['title', 'predicted_rating']]

# Example: Get movie recommendations for user with user_id=1
user_id = 1
recommendations = get_movie_recommendations(user_id)
print(recommendations)
