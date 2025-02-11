import joblib
import requests
import feedparser
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Google Fact Check API Key
API_KEY = "AIzaSyDjZDNPL-Y9DFNiV6Ap1Y6qhBf3QCc7_90"

# RSS Feeds for News Sources
RSS_FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
    "AajTak": "https://www.aajtak.in/rss.xml"
}

# Similarity threshold
SIMILARITY_THRESHOLD = 0.7  # Set your desired threshold here

# Function to fetch news from RSS feeds
def fetch_news_from_rss(feed_url):
    try:
        feed = feedparser.parse(feed_url)
        articles = [entry.title for entry in feed.entries]
        return articles
    except Exception as e:
        print(f"Error fetching RSS feed from {feed_url}: {e}")
        return []

# Function to check similarity between input news and fetched news
def check_similarity(input_text, news_articles):
    if not news_articles:  # Handle empty articles list
        return []

    # Combine input text and articles for TF-IDF transformation
    all_texts = [input_text] + news_articles
    tfidf = TfidfVectorizer().fit_transform(all_texts)

    # Compute cosine similarity between input text and articles
    similarity_scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return similarity_scores

# Function to check news using Google Fact Check API
def check_fact_with_google(news_text):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={news_text}&key={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "claims" in data and data["claims"]:
            results = []
            for claim in data["claims"]:
                claim_text = claim.get("text", "No claim text available")
                rating = claim.get("claimReview", [{}])[0].get("textualRating", "No rating available")
                publisher = claim.get("claimReview", [{}])[0].get("publisher", {}).get("name", "Unknown publisher")
                results.append(f"⚠️ Fact-Checked by {publisher}: {rating} → {claim_text}")
            return "\n".join(results)
        else:
            return "⚠️ Not Found in Google Fact Check"
    except requests.exceptions.RequestException as e:
        return f"⚠️ Error accessing Google Fact Check API: {e}"

# Function to predict news
def predict_news(news_text):
    # Check with Google Fact Check API
    google_fact = check_fact_with_google(news_text)

    # Fetch news from multiple sources
    source_results = {}
    for source, feed_url in RSS_FEEDS.items():
        articles = fetch_news_from_rss(feed_url)
        if articles:  # Only compute similarity if articles are fetched
            similarity_scores = check_similarity(news_text, articles)
            if len(similarity_scores) > 0:  # Check if similarity_scores is non-empty
                max_similarity = max(similarity_scores)
                # Check if max_similarity meets the threshold
                if max_similarity >= SIMILARITY_THRESHOLD:
                    source_results[source] = f"Match (Similarity = {max_similarity:.2f})"
                else:
                    source_results[source] = f"No Match (Similarity = {max_similarity:.2f})"
            else:
                source_results[source] = "No Match (No articles fetched)"
        else:
            source_results[source] = "No Match (No articles fetched)"

    # AI Model Prediction
    try:
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)
        prediction_proba = model.predict_proba(transformed_text)[0]
        ai_prediction = f"❌ Fake News (Confidence: {prediction_proba[1]:.2f})" if prediction[0] == 1 else f"✅ Real News (Confidence: {prediction_proba[0]:.2f})"
    except Exception as e:
        ai_prediction = f"⚠️ Error in AI Model Prediction: {e}"

    # Aggregate results from multiple sources
    source_verification = "\n".join([f"🔍 {source}: {result}" for source, result in source_results.items()])

    return f"{ai_prediction}\n{source_verification}\n{google_fact}"

# Example usage
news_article = input("Enter a news article: ")
result = predict_news(news_article)
print("Prediction:", result)