"""
📊 Movie Recommender — EDA & Visualizations
Powered by TMDB API (Hollywood + Bollywood)
Run this after movie_recommender.py
"""
 
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
 
API_KEY  = "988271023420489e1f239628ed9eff81"
BASE_URL = "https://api.themoviedb.org/3"
 
# ── Fetch data ────────────────────────────────────────────
def get_genre_map():
    try:
        r = requests.get(f"{BASE_URL}/genre/movie/list",
                         params={"api_key": API_KEY}, timeout=10)
        return {g["id"]: g["name"] for g in r.json().get("genres", [])}
    except:
        return {}
 
def fetch(pages=15, language=None):
    genre_map = get_genre_map()
    movies = []
    for page in range(1, pages + 1):
        try:
            params = {"api_key": API_KEY, "page": page, "sort_by": "popularity.desc"}
            if language:
                params["with_original_language"] = language
                url = f"{BASE_URL}/discover/movie"
            else:
                url = f"{BASE_URL}/movie/popular"
                params["language"] = "en-US"
            r = requests.get(url, params=params, timeout=10)
            for m in r.json().get("results", []):
                genres = "|".join(genre_map.get(gid,"?") for gid in m.get("genre_ids",[]))
                movies.append({
                    "title":        m.get("title",""),
                    "genres":       genres or "Unknown",
                    "vote_average": m.get("vote_average", 0),
                    "vote_count":   m.get("vote_count", 0),
                    "popularity":   m.get("popularity", 0),
                    "release_date": m.get("release_date",""),
                    "language":     m.get("original_language",""),
                })
        except: pass
    return pd.DataFrame(movies).drop_duplicates("title").reset_index(drop=True)
 
print("📥 Fetching data for EDA...")
hollywood = fetch(pages=15)
bollywood = fetch(pages=8, language="hi")
movies    = pd.concat([hollywood, bollywood]).drop_duplicates("title").reset_index(drop=True)
movies["year"] = movies["release_date"].str[:4]
print(f"✅ {len(movies)} movies loaded\n")
 
# ── Plot ──────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 24), facecolor="#0d0d1a")
fig.suptitle("🎬 Movie Recommendation System — TMDB EDA Dashboard",
             fontsize=22, fontweight="bold", color="white", y=0.98)
 
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
 
ACCENT = "#f5c518"
BLUE   = "#00b4d8"
PINK   = "#ff6b9d"
GREEN  = "#06d6a0"
BG     = "#0d0d1a"
CARD   = "#1a1a2e"
 
def style_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
 
# 1. Rating Distribution
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "⭐ TMDB Rating Distribution")
ax1.hist(movies["vote_average"], bins=20, color=ACCENT, edgecolor=BG, alpha=0.9)
ax1.axvline(movies["vote_average"].mean(), color=PINK, linestyle="--", linewidth=2,
            label=f"Mean: {movies['vote_average'].mean():.2f}")
ax1.set_xlabel("Rating", color="white")
ax1.set_ylabel("Count", color="white")
ax1.legend(facecolor=CARD, edgecolor="#444", labelcolor="white")
 
# 2. Top Genres
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "🎭 Top 15 Genres")
gc = movies["genres"].str.split("|").explode().value_counts().drop("Unknown", errors="ignore").head(15)
colors = plt.cm.cool(np.linspace(0.2, 0.9, len(gc)))
ax2.barh(gc.index[::-1], gc.values[::-1], color=colors[::-1], edgecolor=BG)
ax2.set_xlabel("Count", color="white")
 
# 3. Movies per Year
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "📅 Movies Released Per Year")
yc = movies[movies["year"] > "1990"]["year"].value_counts().sort_index()
ax3.fill_between(yc.index, yc.values, alpha=0.3, color=BLUE)
ax3.plot(yc.index, yc.values, color=BLUE, linewidth=2.5, marker="o", markersize=4)
ax3.set_xlabel("Year", color="white")
ax3.set_ylabel("Count", color="white")
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
 
# 4. Hollywood vs Bollywood
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "🌍 Hollywood vs Bollywood — Avg Rating")
lang_avg = movies[movies["language"].isin(["en","hi"])].groupby("language")["vote_average"].mean()
bars = ax4.bar(["Hollywood (en)", "Bollywood (hi)"], lang_avg.values,
               color=[BLUE, PINK], edgecolor=BG, width=0.5)
for bar, val in zip(bars, lang_avg.values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             f"{val:.2f}", ha="center", color="white", fontsize=11, fontweight="bold")
ax4.set_ylabel("Avg Rating", color="white")
ax4.set_ylim(0, 10)
 
# 5. Top 15 Most Popular Movies
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, "🏆 Top 15 Most Popular Movies")
top15 = movies.nlargest(15, "popularity")[["title","popularity"]]
top15["short"] = top15["title"].str[:28]
colors2 = plt.cm.autumn(np.linspace(0.2, 0.85, 15))
ax5.barh(range(15), top15["popularity"].values, color=colors2, edgecolor=BG)
ax5.set_yticks(range(15))
ax5.set_yticklabels(top15["short"].values, fontsize=8)
ax5.set_xlabel("Popularity Score", color="white")
 
# 6. Popularity vs Rating scatter
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, "🔵 Popularity vs Rating")
sc = ax6.scatter(movies["popularity"], movies["vote_average"],
                 c=movies["vote_average"], cmap="RdYlGn",
                 alpha=0.5, s=15, edgecolors="none")
cb = plt.colorbar(sc, ax=ax6)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white", fontsize=8)
ax6.set_xlabel("Popularity", color="white")
ax6.set_ylabel("Rating", color="white")
ax6.set_xlim(0, movies["popularity"].quantile(0.97))
 
plt.savefig("eda_dashboard.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("✅ EDA dashboard saved → eda_dashboard.png")
plt.show()