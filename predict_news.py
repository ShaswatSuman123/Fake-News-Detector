import joblib
import requests

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Google Fact Check API Key
API_KEY = "YOUR_GOOGLE_FACT_CHECK_API_KEY"

# Function to check news using Google Fact Check API
def check_fact_with_google(news_text):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={news_text}&key={API_KEY}"
    response = requests.get(url)
    data = response.json()

    # If Google has fact-checked this news
    if "claims" in data:
        for claim in data["claims"]:
            claim_text = claim["text"]
            rating = claim["claimReview"][0]["textualRating"]
            publisher = claim["claimReview"][0]["publisher"]["name"]
            return f"⚠️ Fact-Checked by {publisher}: {rating} → {claim_text}"

    return "⚠️ Not Found in Google Fact Check"

# Function to predict news
def predict_news(news_text):
    google_fact = check_fact_with_google(news_text)

    transformed_text = vectorizer.transform([news_text])
    prediction = model.predict(transformed_text)

    ai_prediction = "❌ Fake News" if prediction[0] == 1 else "✅ Real News (Based on AI Model)"

    return f"{ai_prediction}\n{google_fact}"

# Example usage
news_article = input("Enter a news article: ")
result = predict_news(news_article)
print("Prediction:", result)
