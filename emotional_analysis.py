from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate sentiment score using TextBlob
def get_sentiment_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Apply sentiment analysis to the lyrics dataframe
def add_sentiment_scores(lyrics_df):
    lyrics_df['sentiment_score'] = lyrics_df['clean_lyrics'].apply(lambda x: get_sentiment_score(x) if isinstance(x, str) else 0)
    return lyrics_df

# Plot positive and negative sentiment proportions by year
def plot_sentiment_by_year(lyrics_df):
    lyrics_df['positive'] = lyrics_df['sentiment_score'].apply(lambda x: x if x > 0 else 0)
    lyrics_df['negative'] = lyrics_df['sentiment_score'].apply(lambda x: -x if x <= 0 else 0)
    
    plt.figure(figsize=(14, 7))
    
    # Plot negative sentiment
    plt.subplot(1, 2, 1)
    sns.boxplot(x='year', y='negative', data=lyrics_df)
    plt.title('Negative Sentiment Proportion in R&B Music by Year')
    plt.xticks(rotation=45)
    plt.ylabel('Negative Proportion')
    
    # Plot positive sentiment
    plt.subplot(1, 2, 2)
    sns.boxplot(x='year', y='positive', data=lyrics_df)
    plt.title('Positive Sentiment Proportion in R&B Music by Year')
    plt.xticks(rotation=45)
    plt.ylabel('Positive Proportion')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filepath = './lyrics.RData'
    lyrics_df = pd.read_csv(filepath)  # Assuming lyrics_df is saved as CSV after preprocessing
    lyrics_df = add_sentiment_scores(lyrics_df)
    plot_sentiment_by_year(lyrics_df)