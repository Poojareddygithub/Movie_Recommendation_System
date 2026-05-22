import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pickle.load(open('movie_data.pkl', 'rb'))

# If movie_data.pkl is dictionary convert to dataframe
if isinstance(movies, dict):
    movies = pd.DataFrame(movies)

# Fill null values
movies = movies.fillna('')

# Create tags column
movies['tags'] = (
    movies['genres'].astype(str) + ' ' +
    movies['keywords'].astype(str) + ' ' +
    movies['cast'].astype(str) + ' ' +
    movies['crew'].astype(str)
)

# Convert text into vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# Save files
pickle.dump(movies, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Files generated successfully!")