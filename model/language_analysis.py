import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')

# Function to analyze narrative persona by counting first and second person pronouns
def analyze_pronouns(lyrics):
    first_person = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
    second_person = ['you', 'your', 'yours']

    tokens = word_tokenize(lyrics.lower())
    first_person_count = sum([1 for word in tokens if word in first_person])
    second_person_count = sum([1 for word in tokens if word in second_person])

    total_words = len(tokens)
    return first_person_count / total_words, second_person_count / total_words

# Apply narrative persona analysis to the lyrics dataframe
def add_pronoun_ratios(lyrics_df):
    lyrics_df['first_person_ratio'], lyrics_df['second_person_ratio'] = zip(*lyrics_df['clean_lyrics'].apply(analyze_pronouns))
    return lyrics_df

# Function to analyze language style: average word length and sentence length
def analyze_language_style(lyrics):
    tokens = word_tokenize(lyrics.lower())
    words = [word for word in tokens if word.isalpha()]
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

    sentences = nltk.sent_tokenize(lyrics)
    avg_sentence_length = len(words) / len(sentences) if sentences else 0

    return avg_word_length, avg_sentence_length

# Apply language style analysis to the lyrics dataframe
def add_language_style_metrics(lyrics_df):
    lyrics_df['avg_word_length'], lyrics_df['avg_sentence_length'] = zip(*lyrics_df['clean_lyrics'].apply(analyze_language_style))
    return lyrics_df

# Plot narrative persona ratios and language style metrics
def plot_language_analysis(lyrics_df):
    plt.figure(figsize=(14, 6))

    # Scatterplot: First vs. Second Person Pronoun Usage
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='first_person_ratio', y='second_person_ratio', hue='genre', data=lyrics_df, palette='coolwarm')
    plt.title('First vs. Second Person Pronoun Usage')
    plt.xlabel('First Person Pronoun Ratio')
    plt.ylabel('Second Person Pronoun Ratio')
    plt.legend(title='Genre')

    # Boxplot: Average Word Length by Genre
    plt.subplot(1, 2, 2)
    sns.boxplot(x='genre', y='avg_word_length', data=lyrics_df, palette='coolwarm')
    plt.title('Average Word Length by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Average Word Length')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filepath = './lyrics.RData'
    lyrics_df = pd.read_csv(filepath)  # Assuming lyrics_df is saved as CSV after preprocessing
    lyrics_df = add_pronoun_ratios(lyrics_df)
    lyrics_df = add_language_style_metrics(lyrics_df)
    plot_language_analysis(lyrics_df)
