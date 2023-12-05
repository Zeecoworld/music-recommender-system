import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("music-dataset/song_data.csv")

feature_cols=['acousticness', 'danceability', 'song_duration_ms', 'energy',
              'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode',
              'speechiness', 'tempo', 'time_signature', 'audio_valence']

scaler = MinMaxScaler()
normalized_df =scaler.fit_transform(df[feature_cols])

# print(normalized_df[:2])
indices = pd.Series(df.index, index=df['song_name']).drop_duplicates()

# Create cosine similarity matrix based on given matrix
cosine = cosine_similarity(normalized_df)

def generate_recommendation(song_name, model_type=cosine):
    """
    Purpose: Function for song recommendations 
    Inputs: song title and type of similarity model
    Output: Pandas series of recommended songs
    """
    # Get song indices
    index=indices[song_name]
    # Get list of songs for given songs
    score=list(enumerate(model_type[indices['The Middle']]))
    # Sort the most similar smilaritongs
    similarity_score = sorted(score, key=lambda x: x[1][0] if len(x[1]) > 0 else None, reverse=True)

    # Select the top-10 recommend songs
    similarity_score = similarity_score[1:11]
    top_songs_index = [i[0] for i in similarity_score]
    # Top 10 recommende songs
    top_songs=df['song_name'].iloc[top_songs_index]
    return top_songs

print("Recommended Songs:")
print(generate_recommendation('The Middle',cosine).values)