import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pyreadr
import pandas as pd
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Activate the pandas <-> R dataframe conversion
pandas2ri.activate()

# Load the RData file
def load_rdata_file(filepath):
    result = pyreadr.read_r(filepath)  # Load the RData file
    lyrics_df = result['dt_lyrics']  # Extract the dataframe
    return lyrics_df

# Preprocess lyrics to remove stopwords and punctuation
def preprocess_lyrics(text, stop_words):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
        words = text.split()  # Split the text into words
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        return ' '.join(words)  # Join words back into a string
    else:
        return ""

# Get the default list of English stopwords
def get_stopwords():
    stop_words = set(stopwords.words('english'))
    custom_stop_words = {'oh', 'yeah', 'la', 'uh', 'ah', 'ooh', 'na', 'yo', 'hey', 'im', 'dont', 'didnt', 'youre', 'hes', 'shes', 'theyre', 'weve', 'ive'} 
    stop_words.update(custom_stop_words)  # Add custom stop words to the default list
    return stop_words

# Apply preprocessing to the lyrics dataframe
def preprocess_lyrics_dataframe(lyrics_df):
    stop_words = get_stopwords()
    lyrics_df['clean_lyrics'] = lyrics_df['lyrics'].apply(lambda x: preprocess_lyrics(x, stop_words))
    return lyrics_df

if __name__ == "__main__":
    filepath = './lyrics.RData'
    lyrics_df = load_rdata_file(filepath)
    lyrics_df = preprocess_lyrics_dataframe(lyrics_df)
    print(lyrics_df.head())
