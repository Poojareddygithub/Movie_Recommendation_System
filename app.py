import streamlit as st
import pickle
import pandas as pd

# Load pickle file
data = pickle.load(open('movie_data.pkl', 'rb'))

# Handle tuple data
if isinstance(data, tuple):
    movies = data[0]
    similarity = data[1]
else:
    movies = data
    similarity = None

# Fill missing values
movies = movies.fillna('')

st.title("Movie Recommendation System")

# Movie dropdown
movie_list = movies['title'].values

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    # Get top 5 similar movies
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# Button
if st.button("Recommend"):

    if similarity is None:
        st.error("Similarity data not found in pickle file.")
    else:
        recommendations = recommend(selected_movie)

        st.subheader("Recommended Movies")

        for movie in recommendations:
            st.write(movie)