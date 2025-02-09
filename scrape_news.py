from newspaper import Article
import pandas as pd

# List of trusted news URLs
trusted_urls = [
    "https://www.bbc.com/news/world-asia-67305143",
    "https://edition.cnn.com/2025/02/08/politics/us-election-update/index.html",
    "https://www.reuters.com/world/india/latest-news-2025-02-08/",
    "https://www.thehindu.com/news/national/",
    "https://aajtak.intoday.in/",
    "https://timesofindia.indiatimes.com/",
]

# Function to extract news text
def get_news_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None

# Scrape articles from trusted sources
news_data = []
for url in trusted_urls:
    text = get_news_text(url)
    if text:
        news_data.append({"text": text, "source": "trusted"})

# Convert to DataFrame
df = pd.DataFrame(news_data)

# Save dataset
df.to_csv("trusted_news.csv", index=False)

print("✅ Trusted news dataset saved as 'trusted_news.csv'. Now proceed to Step 2!")
