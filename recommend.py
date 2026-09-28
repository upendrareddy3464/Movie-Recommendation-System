import difflib
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_and_prepare_data(csv_path: str = "Movies.csv"):
    """Load the movies dataset and calculate cosine similarity matrix."""
    movies_data = pd.read_csv(csv_path)

    # Selecting relevant features for recommendation
    selected_features = ["genres", "keywords", "tagline", "cast", "director"]

    # Replacing null values with empty string
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna("")

    # Combining selected features
    combined_features = (
        movies_data["genres"]
        + " "
        + movies_data["keywords"]
        + " "
        + movies_data["tagline"]
        + " "
        + movies_data["cast"]
        + " "
        + movies_data["director"]
    )

    # Convert text to feature vectors using TF-IDF
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Compute cosine similarity
    similarity = cosine_similarity(feature_vectors)

    return movies_data, similarity


def recommend_movies(movie_name: str, movies_data: pd.DataFrame, similarity: np.ndarray, top_n: int = 15):
    """Recommend similar movies based on cosine similarity."""
    list_of_all_titles = movies_data["title"].tolist()

    # Find closest match for input movie name
    close_matches = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not close_matches:
        print(f"No close match found for '{movie_name}'. Please try another title.")
        return []

    close_match = close_matches[0]
    print(f"\nTop match in database: '{close_match}'\n")

    movie_rows = movies_data[movies_data.title == close_match]
    if movie_rows.empty:
        print(f"Could not find details for '{close_match}'.")
        return []

    index_of_movie = movie_rows["index"].values[0]
    similarity_scores = list(enumerate(similarity[index_of_movie]))

    # Sort movies based on similarity score (descending)
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    print(f"Movies suggested for you (based on '{close_match}'):\n" + "-" * 50)
    count = 1
    for movie in sorted_similar_movies:
        idx = movie[0]
        score = movie[1]
        title = movies_data[movies_data.index == idx]["title"].values[0]
        if count <= top_n:
            print(f"{count:2d}. {title:<40} (similarity: {score:.3f})")
            recommendations.append(title)
            count += 1

    return recommendations


def main():
    movies_data, similarity = load_and_prepare_data("Movies.csv")

    if len(sys.argv) > 1:
        movie_query = " ".join(sys.argv[1:])
    else:
        movie_query = input("Enter your favourite movie name: ").strip()

    if not movie_query:
        print("No movie name provided. Exiting.")
        return

    recommend_movies(movie_query, movies_data, similarity)


if __name__ == "__main__":
    main()
