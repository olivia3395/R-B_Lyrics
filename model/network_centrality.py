import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Function to calculate degree centrality and eigenvector centrality
def calculate_centrality(G):
    degree_centrality = nx.degree_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)
    return degree_centrality, eigenvector_centrality

# Function to filter nodes based on degree centrality threshold
def filter_nodes_by_centrality(G, degree_centrality, threshold=0.02):
    filtered_nodes = {node: degree for node, degree in degree_centrality.items() if degree > threshold}
    return G.subgraph(filtered_nodes)

# Plot the network with nodes colored by centrality
def plot_centrality_network(G, degree_centrality, eigenvector_centrality, title="Centrality Network of Lyrics"):
    node_size = [v * 3000 for v in degree_centrality.values()]  # Scale degree centrality for node size
    node_color = [v for v in eigenvector_centrality.values()]   # Color based on eigenvector centrality
    
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.7, iterations=50)
    
    # Draw nodes with color and size based on centrality measures
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, cmap=plt.cm.Blues, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray', width=1)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    
    plt.colorbar(nodes, label="Eigenvector Centrality")
    plt.title(title, size=16)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    filepath = './lyrics.RData'
    lyrics_df = pd.read_csv(filepath)  # Assuming lyrics_df is saved as CSV after preprocessing
    bigrams = extract_bigrams(lyrics_df)
    filtered_bigrams = filter_bigrams(bigrams)
    bigram_network = create_bigram_network(filtered_bigrams)
    
    # Calculate centralities
    degree_centrality, eigenvector_centrality = calculate_centrality(bigram_network)
    
    # Filter nodes based on degree centrality threshold
    filtered_network = filter_nodes_by_centrality(bigram_network, degree_centrality)
    
    # Plot the centrality network
    plot_centrality_network(filtered_network, degree_centrality, eigenvector_centrality)
