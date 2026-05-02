from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from movie_recommender import ContentBasedRecommender, fetch_movies, fetch_bollywood

app = Flask(__name__)
CORS(app)

print("Loading movies...")

# Fetch data
hollywood = fetch_movies(5)
bollywood = fetch_bollywood(3)
movies = pd.concat([hollywood, bollywood]).drop_duplicates("movieId")

# Build model
cb = ContentBasedRecommender(movies)

print("Model ready!")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    movie = data.get("movie")

    res = cb.recommend(movie, top_n=10)

    if res.empty:
        return jsonify([])

    return jsonify(res.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)