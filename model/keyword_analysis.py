import itertools
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from nltk import ngrams
from nltk.corpus import stopwords
from collections import Counter

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define function to tokenize lyrics and create bigrams (two-word phrases)
def tokenize_and_bigrams(text):
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in nltk.word_tokenize(text.lower()) if word.isalpha() and word not in stop_words]
    return list(ngrams(tokens, 2))

# Collect all bigrams from the lyrics dataframe
def extract_bigrams(lyrics_df):
    bigrams = list(itertools.chain.from_iterable(lyrics_df['clean_lyrics'].dropna().apply(tokenize_and_bigrams)))
    return bigrams

# Create a bigram frequency dictionary and filter low-frequency bigrams
def filter_bigrams(bigrams, threshold=30):
    bigram_freq = Counter(bigrams)
    filtered_bigrams = {bigram: count for bigram, count in bigram_freq.items() if count > threshold}
    return filtered_bigrams

# Create a network graph of word pairs
def create_bigram_network(filtered_bigrams):
    G = nx.Graph()
    for (word1, word2), count in filtered_bigrams.items():
        G.add_edge(word1, word2, weight=count)
    return G

# Plot the word network using networkx and matplotlib
def plot_bigram_network(G, title="Bigram Word Network of Lyrics"):
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes and edges with sizes based on degree and edge weights
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray', width=[G[u][v]['weight']*0.2 for u, v in G.edges()])
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    
    plt.title(title, size=16)
    plt.axis('off')  # Turn off the axis
    plt.show()

if __name__ == "__main__":
    filepath = './lyrics.RData'
    lyrics_df = pd.read_csv(filepath)  # Assuming lyrics_df is saved as CSV after preprocessing
    bigrams = extract_bigrams(lyrics_df)
    filtered_bigrams = filter_bigrams(bigrams)
    bigram_network = create_bigram_network(filtered_bigrams)
    plot_bigram_network(bigram_network)
