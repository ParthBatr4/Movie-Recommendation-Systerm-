"""
🎬 Movie Recommendation System — Powered by TMDB API
✅ Hollywood + Bollywood + Latest 2024 Movies
✅ Content-Based + Popularity-Based + Hybrid filtering
"""
 
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
 
API_KEY  = "988271023420489e1f239628ed9eff81"
BASE_URL = "https://api.themoviedb.org/3"
 
 
def get_genre_map():
    try:
        r = requests.get(f"{BASE_URL}/genre/movie/list",
                         params={"api_key": API_KEY, "language": "en-US"}, timeout=10)
        return {g["id"]: g["name"] for g in r.json().get("genres", [])}
    except:
        return {}
 
 
def fetch_movies(pages=20):
    print(f"🌐 Fetching Hollywood movies...")
    all_movies = []
    genre_map  = get_genre_map()
    for page in range(1, pages + 1):
        try:
            r = requests.get(f"{BASE_URL}/movie/popular",
                             params={"api_key": API_KEY, "language": "en-US", "page": page},
                             timeout=10)
            for m in r.json().get("results", []):
                genres = "|".join(genre_map.get(gid, "Unknown") for gid in m.get("genre_ids", [])) or "Unknown"
                all_movies.append({
                    "movieId":      m["id"],
                    "title":        m.get("title", ""),
                    "genres":       genres,
                    "overview":     m.get("overview", ""),
                    "vote_average": float(m.get("vote_average", 0)),
                    "vote_count":   int(m.get("vote_count", 0)),
                    "release_date": m.get("release_date", ""),
                    "popularity":   float(m.get("popularity", 0)),
                    "language":     "en",
                })
        except Exception as e:
            print(f"  ⚠️  Page {page}: {e}")
    df = pd.DataFrame(all_movies).drop_duplicates("movieId").reset_index(drop=True)
    print(f"✅  {len(df)} Hollywood movies fetched!")
    return df
 
 
def fetch_bollywood(pages=10):
    print(f"🎭 Fetching Bollywood movies...")
    all_movies = []
    genre_map  = get_genre_map()
    for page in range(1, pages + 1):
        try:
            r = requests.get(f"{BASE_URL}/discover/movie",
                             params={
                                 "api_key": API_KEY,
                                 "with_original_language": "hi",
                                 "sort_by": "popularity.desc",
                                 "page": page,
                                 "vote_count.gte": 10
                             }, timeout=10)
            data = r.json()
            results = data.get("results", [])
            for m in results:
                genres = "|".join(genre_map.get(gid, "Unknown") for gid in m.get("genre_ids", [])) or "Unknown"
                all_movies.append({
                    "movieId":      m["id"],
                    "title":        m.get("title", ""),
                    "genres":       genres,
                    "overview":     m.get("overview", ""),
                    "vote_average": float(m.get("vote_average", 0)),
                    "vote_count":   int(m.get("vote_count", 0)),
                    "release_date": m.get("release_date", ""),
                    "popularity":   float(m.get("popularity", 0)),
                    "language":     "hi",
                })
        except Exception as e:
            print(f"  ⚠️  Page {page}: {e}")
 
    df = pd.DataFrame(all_movies).drop_duplicates("movieId").reset_index(drop=True)
    print(f"✅  {len(df)} Bollywood movies fetched!")
    return df
 
 
def search_movie_tmdb(query):
    try:
        r = requests.get(f"{BASE_URL}/search/movie",
                         params={"api_key": API_KEY, "query": query},
                         timeout=10)
        return [m["title"] for m in r.json().get("results", [])[:5]]
    except:
        return []
 
 
class ContentBasedRecommender:
    def __init__(self, movies):
        self.movies = movies.copy().reset_index(drop=True)
        self._build()
 
    def _build(self):
        self.movies["soup"] = (
            self.movies["genres"].str.replace("|", " ", regex=False) + " " +
            self.movies["overview"].fillna("")
        )
        tfidf  = TfidfVectorizer(stop_words="english", max_features=5000)
        matrix = tfidf.fit_transform(self.movies["soup"])
        self.sim     = cosine_similarity(matrix, matrix)
        self.indices = pd.Series(self.movies.index, index=self.movies["title"].str.lower())
 
    def recommend(self, title, top_n=10):
        tl = title.lower()
        matches = [t for t in self.indices.index if tl in t]
        if not matches:
            live = search_movie_tmdb(title)
            print(f"\n  💡 '{title}' not found. Did you mean:")
            for i, m in enumerate(live, 1):
                print(f"     {i}. {m}")
            return pd.DataFrame()
        idx    = self.indices[matches[0]]
        sims   = sorted(enumerate(self.sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
        idxs   = [i[0] for i in sims]
        scores = [round(i[1], 3) for i in sims]
        res = self.movies.iloc[idxs][["title","genres","vote_average","release_date","language"]].copy()
        res.insert(0, "match_score", scores)
        res.reset_index(drop=True, inplace=True)
        res.index += 1
        return res
 
 
class PopularityRecommender:
    def __init__(self, movies):
        self.movies = movies.copy()
        self._build()
 
    def _build(self):
        # Separate thresholds for Hollywood and Bollywood
        en = self.movies[self.movies["language"] == "en"].copy()
        hi = self.movies[self.movies["language"] == "hi"].copy()
 
        def score_df(df):
            if df.empty:
                return df
            m = df["vote_count"].quantile(0.5)   # lower threshold = more results
            C = df["vote_average"].mean()
            qualified = df[df["vote_count"] >= m].copy()
            qualified["weighted_score"] = (
                qualified["vote_count"] / (qualified["vote_count"] + m) * qualified["vote_average"] +
                m / (qualified["vote_count"] + m) * C
            ).round(3)
            return qualified
 
        self.ranked = pd.concat([score_df(en), score_df(hi)]).sort_values("weighted_score", ascending=False)
        print(f"  📊 Ranked pool: {len(self.ranked)} movies "
              f"(Hollywood: {len(self.ranked[self.ranked['language']=='en'])}, "
              f"Bollywood: {len(self.ranked[self.ranked['language']=='hi'])})")
 
    def recommend(self, genre=None, language=None, top_n=10):
        df = self.ranked.copy()
        if genre:
            df = df[df["genres"].str.contains(genre, case=False, na=False)]
        if language:
            df = df[df["language"] == language]
        res = df.head(top_n)[["title","genres","vote_average","weighted_score","release_date","language"]]
        res = res.reset_index(drop=True)
        res.index += 1
        return res
 
 
class HybridRecommender:
    def __init__(self, cb, pop):
        self.cb  = cb
        self.pop = pop
 
    def recommend(self, title, genre=None, top_n=10):
        cb_recs  = self.cb.recommend(title, top_n=50)
        pop_recs = self.pop.recommend(genre=genre, top_n=50)
        if cb_recs.empty:
            return pop_recs.head(top_n)
        scaler = MinMaxScaler()
        cb_recs["cb_norm"]   = scaler.fit_transform(cb_recs[["match_score"]])
        pop_recs["pop_norm"] = scaler.fit_transform(pop_recs[["weighted_score"]])
        merged = pd.merge(
            cb_recs[["title","genres","release_date","language","cb_norm"]],
            pop_recs[["title","pop_norm"]], on="title", how="outer"
        ).fillna(0)
        merged["hybrid_score"] = (0.6*merged["cb_norm"] + 0.4*merged["pop_norm"]).round(4)
        merged = merged.sort_values("hybrid_score", ascending=False).head(top_n)
        merged.reset_index(drop=True, inplace=True)
        merged.index += 1
        return merged[["title","genres","release_date","language","hybrid_score"]]
 
 
def print_stats(movies):
    print("\n" + "="*60)
    print("📊  TMDB DATASET STATISTICS")
    print("="*60)
    print(f"  Total Movies     : {len(movies)}")
    print(f"  Hollywood        : {len(movies[movies['language']=='en'])}")
    print(f"  Bollywood        : {len(movies[movies['language']=='hi'])}")
    print(f"  Avg Rating       : {movies['vote_average'].mean():.2f} / 10")
    yr = movies["release_date"].str[:4]
    print(f"  Newest Year      : {yr.max()}")
    top_genres = movies["genres"].str.split("|").explode().value_counts().head(5)
    print("\n  Top 5 Genres:")
    for g, c in top_genres.items():
        print(f"    {g:<22} {c} movies")
    print("="*60 + "\n")
 
 
def main():
    print("\n" + "🎬 "*18)
    print("   MOVIE RECOMMENDATION SYSTEM")
    print("   Hollywood + Bollywood + Latest 2024")
    print("🎬 "*18 + "\n")
 
    hollywood = fetch_movies(pages=20)
    bollywood = fetch_bollywood(pages=10)
    movies    = pd.concat([hollywood, bollywood]).drop_duplicates("movieId").reset_index(drop=True)
 
    print_stats(movies)
 
    print("🔧 Building models...")
    cb     = ContentBasedRecommender(movies)
    pop    = PopularityRecommender(movies)
    hybrid = HybridRecommender(cb, pop)
    print("✅ All models ready!\n")
 
    # Show demo results
    print("─"*60)
    print("🏆 TOP 10 POPULAR MOVIES (Hollywood)")
    print("─"*60)
    print(pop.recommend(language="en", top_n=10).to_string())
 
    print("\n" + "─"*60)
    print("🎭 TOP 10 BOLLYWOOD MOVIES")
    print("─"*60)
    print(pop.recommend(language="hi", top_n=10).to_string())
 
    # Interactive
    print(f"\n{'═'*60}")
    print("🕹️  INTERACTIVE MODE  (type 'quit' to exit)")
    print("═"*60)
 
    while True:
        print("\nOptions:")
        print("  1 → Similar movies    (Inception / Pathaan / KGF)")
        print("  2 → Top Hollywood     (best rated English movies)")
        print("  3 → Top Bollywood     (best rated Hindi movies)")
        print("  4 → Browse by genre   (Action/Comedy/Romance/Horror)")
        print("  5 → Hybrid            (movie + genre combo)")
 
        choice = input("\nChoose [1-5/quit]: ").strip().lower()
 
        if choice in ("quit","q","exit"):
            print("👋 Goodbye!")
            break
 
        elif choice == "1":
            movie = input("  Enter movie name: ").strip()
            res   = cb.recommend(movie, top_n=10)
            if not res.empty:
                print(f"\n🎬 Movies similar to '{movie}':")
                print(res.to_string())
 
        elif choice == "2":
            res = pop.recommend(language="en", top_n=10)
            print("\n🏆 Top Hollywood Movies:")
            print(res.to_string())
 
        elif choice == "3":
            res = pop.recommend(language="hi", top_n=10)
            print("\n🎭 Top Bollywood Movies:")
            print(res.to_string())
 
        elif choice == "4":
            genre = input("  Genre (Action/Comedy/Romance/Horror/Thriller/Drama): ").strip()
            lang  = input("  Language (en=Hollywood / hi=Bollywood / blank=both): ").strip() or None
            res   = pop.recommend(genre=genre, language=lang, top_n=10)
            if res.empty:
                print(f"  ⚠️  No movies found for genre '{genre}'. Try: Action, Drama, Comedy, Thriller")
            else:
                print(f"\n🎬 Top {genre} movies:")
                print(res.to_string())
 
        elif choice == "5":
            movie = input("  Movie you liked: ").strip()
            genre = input("  Genre filter (blank=none): ").strip() or None
            res   = hybrid.recommend(movie, genre=genre, top_n=10)
            if not res.empty:
                print("\n🔀 Hybrid Recommendations:")
                print(res.to_string())
 
        else:
            print("  ❌ Invalid. Enter 1-5 or quit.")
 
 
if __name__ == "__main__":
    main()