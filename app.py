import streamlit as st
import joblib
import pandas as pd
from fuzzywuzzy import process
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

def get_img_as_base64(file):
    with open(file, "rb") as f:
        img = f.read()
    return base64.b64encode(img).decode()

image = get_img_as_base64("img.jpg")


cosine_sim_df = joblib.load('cosine_similarity_matrix.pkl')
mlb = joblib.load('genre_binarizer.pkl')
movies_with_mode = pd.read_csv('movies_with_mode.csv')


def recommend_movies(user_input, user_rating, cosine_sim_df, movies_with_mode, top_k=5):

    exact_genre_movies = movies_with_mode[movies_with_mode['genres'].str.contains(user_input, case=False, na=False)]
    exact_title_match = movies_with_mode[movies_with_mode['normalized_title'].str.lower() == user_input.lower()]

    if not exact_title_match.empty:  
        movie_id = exact_title_match.iloc[0]['movieId']
        similar_movies = cosine_sim_df[movie_id].sort_values(ascending=False).iloc[1:top_k + 1]
    elif not exact_genre_movies.empty:  
        movie_ids = exact_genre_movies['movieId'].values
        genre_similarities = cosine_sim_df.loc[movie_ids].mean(axis=0).sort_values(ascending=False)
        similar_movies = genre_similarities.iloc[:top_k]
    else:  
        genre_match = process.extractOne(user_input, '|'.join(movies_with_mode['genres'].unique()).split('|'))
        if genre_match and genre_match[1] > 80:
            matched_genre = genre_match[0]
            fuzzy_genre_movies = movies_with_mode[movies_with_mode['genres'].str.contains(matched_genre)]
            movie_ids = fuzzy_genre_movies['movieId'].values
            genre_similarities = cosine_sim_df.loc[movie_ids].mean(axis=0).sort_values(ascending=False)
            similar_movies = genre_similarities.iloc[:top_k]
        else:
            print("Sorry, we couldn't find the movie or genre. Please check your input.")
            return pd.DataFrame()

    
    recommended_movies = movies_with_mode[movies_with_mode['movieId'].isin(similar_movies.index)].copy()

    
    print(f"\nRecommended Movies based on your input '{user_input}' (Genre/Name) with rating '{user_rating}':")
    print(recommended_movies[['normalized_title', 'genres']])
    return recommended_movies


def predict_rating_for_unseen(user_rating, unseen_movie_name, unseen_movie_genre, cosine_sim_df, mode_ratings, movies_with_mode, top_k=5):
    
    """
    Predict the rating for an unseen movie based on the genre and user rating, 
    using the mode ratings of similar movies and movie recommendations based on the genre.
    """
    
    recommended_movies = recommend_movies(unseen_movie_genre, user_rating, cosine_sim_df, movies_with_mode, top_k=top_k)
    
    if recommended_movies.empty:
        print("Sorry, no recommendations found based on the provided genre.")
        return None

    recommended_movie_ids = recommended_movies['movieId'].values
    genre_similarities = cosine_sim_df.loc[recommended_movie_ids].mean(axis=0).sort_values(ascending=False)
    
    similar_movies = genre_similarities.iloc[1:top_k+1]

    predicted_ratings = []
    for movie_id in similar_movies.index:
        mode_rating = mode_ratings[mode_ratings['movieId'] == movie_id].mode_rating.values[0]
        
        predicted_rating = (user_rating + mode_rating) / 2
        predicted_ratings.append(predicted_rating)

    predicted_movies = movies_with_mode[movies_with_mode['movieId'].isin(similar_movies.index)].copy()
    
    predicted_movies['predicted_rating'] = predicted_ratings

    print(f"\nPredicted Rating for '{unseen_movie_name}': {predicted_ratings[0]:.2f}")

    return predicted_movies

st.markdown(f"""
    <style>
        /* Style the background image at the top 25% of the screen */
        [data-testid="stAppViewContainer"] {{
            height: 100vh;  /* Make sure the app takes the full height of the viewport */
            display: flex;
            flex-direction: column;
            justify-content: flex-start;  /* Align the content at the top */
        }}
        
        .background-container {{
            background-image: url("data:image/png;base64,{image}");
            background-size: cover;  /* Ensure the image covers the area */
            background-position: center;
            height: 25vh;  /* Set the background image to cover the top 25% */
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        
        .heading {{
            font-size: 2rem;
            font-weight: bold;
            color: black;
        }}
        
        .content-container {{
            display: flex;
            padding: 20px;
        }}
        
        /* Left Column (Movie Recommendation) */
        .left-column {{
            flex: 2;
            margin-right: 20px;
        }}
        
        /* Right Column (Unseen Movie Prediction) */
        .right-column {{
            flex: 1;
        }}

        /* Set a divider between columns */
        .divider {{
            border-left: 2px solid #ccc;
            height: 100%;
            margin: 0 20px;
        }}

        /* Style the buttons */
        .stButton>button {{
            background-color: #f39c12;
            color: white;
            border-radius: 8px;
        }}
        .stButton>button:hover {{
            background-color: #e67e22;
        }}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="background-container"><div class="heading">Movie Recommendation System</div></div>', unsafe_allow_html=True)

st.markdown('<div class="content-container">', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])  

with col1:  
    st.subheader("Movie Recommendation")
    user_input = st.text_input("Enter a movie name or genre('|' separated):")
    user_rating = st.number_input("Rate the movie (1.0-5.0):", min_value=1.0, max_value=5.0, step=0.1)

    if st.button('Get Recommendations'):
        recommended_movies = recommend_movies(user_input, user_rating, cosine_sim_df, movies_with_mode)
        if not recommended_movies.empty:
            st.write(recommended_movies[['normalized_title', 'genres']].rename(columns={'normalized_title': 'Title'}).reset_index(drop=True)
)
        else:
            st.write("Sorry, we couldn't find the movie or genre. Please check your input.")

with col2:  
    st.subheader("Predict Rating for Unseen Movie")
    unseen_movie_name = st.text_input("Enter the name of the unseen movie:")
    unseen_movie_genre = st.text_input("Enter the genre of the unseen movie:")
    unseen_movie_rating = st.number_input("Enter your rating for the unseen movie (1.0-5.0):", min_value=1.0, max_value=5.0, step=0.1)

    if st.button('Predict Rating'):
        predicted_movies = predict_rating_for_unseen(user_rating=unseen_movie_rating,
                                                     unseen_movie_name=unseen_movie_name,
                                                     unseen_movie_genre=unseen_movie_genre,
                                                     cosine_sim_df=cosine_sim_df,
                                                     mode_ratings=movies_with_mode[['movieId', 'mode_rating']],
                                                     movies_with_mode=movies_with_mode)
        if predicted_movies is not None:
            st.write(f"Predicted Rating for '{unseen_movie_name}': {predicted_movies['predicted_rating'].iloc[0]:.2f}")

           
            st.session_state['similar_movies'] = predicted_movies[['normalized_title', 'genres']].rename(columns={'normalized_title': 'Title'}).reset_index(drop=True)

    if 'similar_movies' in st.session_state and st.button('More Movies'):
        st.write("Here are more similar movies:")
        st.write(st.session_state['similar_movies'])
