import pandas as pd
from emotional_analysis import add_sentiment_scores, plot_sentiment_by_year
from language_analysis import add_pronoun_ratios, add_language_style_metrics, plot_language_analysis
from keyword_analysis import extract_bigrams, filter_bigrams, create_bigram_network, plot_bigram_network
from network_centrality import calculate_centrality, filter_nodes_by_centrality, plot_centrality_network

if __name__ == "__main__":
    filepath = './lyrics.csv'  # Assuming lyrics_df is saved as CSV after preprocessing
    lyrics_df = pd.read_csv(filepath)
    
    # Sentiment Analysis
    lyrics_df = add_sentiment_scores(lyrics_df)
    plot_sentiment_by_year(lyrics_df)
    
    # Language Analysis
    lyrics_df = add_pronoun_ratios(lyrics_df)
    lyrics_df = add_language_style_metrics(lyrics_df)
    plot_language_analysis(lyrics_df)
    
    # Keyword Analysis
    bigrams = extract_bigrams(lyrics_df)
    filtered_bigrams = filter_bigrams(bigrams)
    bigram_network = create_bigram_network(filtered_bigrams)
    plot_bigram_network(bigram_network)
    
    # Network Centrality Analysis
    degree_centrality, eigenvector_centrality = calculate_centrality(bigram_network)
    filtered_network = filter_nodes_by_centrality(bigram_network, degree_centrality)
    plot_centrality_network(filtered_network, degree_centrality, eigenvector_centrality)