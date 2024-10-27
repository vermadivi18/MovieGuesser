import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data
movies = pickle.load(open('movie_list.pkl','rb'))
cosine_sim = pickle.load(open('similarity.pkl','rb'))

def recommend(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]


# Streamlit app
st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="wide")
st.title('🎬 Movie Recommender System')
st.sidebar.header('Movie Selection')

# Select box for movie selection
movie_list = movies['title'].values
selected_movie = st.sidebar.selectbox("Select a movie:", movie_list)

if st.sidebar.button('Show Recommendations'):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies:")
    cols = st.columns(5)

    for idx, (col, row) in enumerate(zip(cols, recommendations.iterrows())):
        with col:
            st.text(row[1]['title'])



# Style adjustments
st.markdown("""
<style>
    body {
        background-color: #FFC0CB;  /* Change to your desired color */
    }
    .css-1d391kg {
        background-color: #FFC0CB;  /* Change sidebar color */
    }
    .css-1d391kg h2 {
        color: #3c3c3c;  /* Change header color */
    }
</style>
""", unsafe_allow_html=True)
