🎬 Movie Recommendation System
A complete Machine Learning project powered by TMDB API — includes Hollywood + Bollywood + latest 2024 movies!
📁 Project Structure
movie_recommender/
├── movie_recommender.py   ← Main ML system
├── eda_visualization.py   ← Charts & graphs
├── requirements.txt       ← Dependencies
└── README.md              ← This file
🚀 How to Run
Step 1 — Install dependencies
powershellpip install -r requirements.txt
Step 2 — Run the main recommender
powershellpython movie_recommender.py
Step 3 — Run EDA visualizations
powershellpython eda_visualization.py
🤖 Algorithms Used
1. Content-Based Filtering

TF-IDF on movie genres + overview text
Cosine similarity between movies
Finds movies with similar content

2. Popularity-Based (IMDb Formula)

Weighted Rating = (v/(v+m)) × R + (m/(v+m)) × C
Filters by genre and language
Best for discovering top-rated movies

3. Hybrid Recommender

Combines Content-Based (60%) + Popularity (40%)
Best overall recommendations

🎬 Movies You Can Search
Hollywood: Inception, Avatar, Oppenheimer, Deadpool, Batman
Bollywood: Pathaan, KGF, Jawan, Animal, RRR
Genres: Action, Comedy, Romance, Horror, Thriller, Drama
📊 Dataset — TMDB API

Source: The Movie Database
900,000+ movies including latest 2024 releases
Hollywood + Bollywood + International
Free public API — no sign up needed