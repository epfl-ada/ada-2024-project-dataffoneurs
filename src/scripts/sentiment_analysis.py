import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
import nltk
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import pipeline
from scipy.special import softmax
from sentence_transformers import SentenceTransformer, util
from scipy.interpolate import interp1d

nltk.download('punkt')

DATA_PATH = "data/final_dataset.csv"

# Preprocessing
def preprocess():
    
    df_extended = pd.read_csv(DATA_PATH)[["Wikipedia_movie_ID", "summary", "category"]]
    df_extended.dropna(subset=["summary"], inplace=True)
    return df_extended

"""
Sentiment analysis segment-emotion version
"""
def sentiment_analysis_segment(limit=1000, threshold=0.5):

    df_extended = preprocess()

    #Model
    device = 0 if torch.cuda.is_available() else 'cpu'
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True, device=device)
    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

    big_df = pd.DataFrame()

    #iterate over summaries
    for idx, t in tqdm(df_extended['summary'].iloc[:limit].items(), total=len(df_extended['summary'].iloc[:limit])):
        sentences = nltk.sent_tokenize(t)
        embeddings = sentence_model.encode(sentences, convert_to_tensor=True)

        # Segment the text based on cosine similarity
        segments = []
        current_segment = [sentences[0]]

        for i in range(len(sentences) - 1):
            similarity = util.pytorch_cos_sim(embeddings[i], embeddings[i + 1]).item()

            if similarity < threshold:
                segments.append(" ".join(current_segment))
                current_segment = []

            current_segment.append(sentences[i + 1])

        #final segment join
        segments.append(" ".join(current_segment))

        # Classify the segments for emotions
        out = classifier(segments)
        emotions_flattened = [{item['label']: item['score'] for item in entry} for entry in out]
        emotions = pd.DataFrame(emotions_flattened)

        emotions['segment_id'] = [i for i in range(len(segments))]
        emotions['segment'] = [segments[i] for i in range(len(segments))]
        emotions['Wikipedia_movie_ID'] = [df_extended.iloc[idx]['Wikipedia_movie_ID']] * len(emotions)
        
        big_df = pd.concat([big_df, emotions], ignore_index=True)

    return big_df


"""
Sentiment analysis sentence-emotion version
"""
def sentiment_analysis_sentences(limit=1000):
    
    df_extended = preprocess()

    #model
    device = 0 if torch.cuda.is_available() else 'cpu'
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True, device=device)
    big_df = pd.DataFrame()


    for idx, t in tqdm(df_extended['summary'].iloc[:limit].items(), total=len(df_extended['summary'].iloc[:limit])):
        sentences = nltk.sent_tokenize(t)

        out = classifier(sentences)

        emotions_flattened = [{item['label']: item['score'] for item in entry} for entry in out]
        emotions = pd.DataFrame(emotions_flattened)

        emotions['sentence_id'] = [i for i in range(len(sentences))]
        emotions['sentence'] = [sentences[i] for i in range(len(sentences))]
        emotions['Wikipedia_movie_ID'] = [df_extended.iloc[idx]['Wikipedia_movie_ID']] * len(emotions) 

        big_df = pd.concat([big_df, emotions], ignore_index=True)
    
    return big_df


"""
Sentiment analysis sentence-positive/negative version
"""
def sentiment_analysis_posneg(limit=1000):
    
    df_extended = preprocess()

    # model
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL) 
    model = AutoModelForSequenceClassification.from_pretrained(MODEL) 
    dict_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}  

    big_df = pd.DataFrame()


    for idx, t in tqdm(df_extended['summary'].iloc[:limit].items(), total=len(df_extended['summary'].iloc[:limit])):
        #separate the text into sentences
        sentences = nltk.sent_tokenize(t) 
        for idx_s, s in enumerate(sentences):
            t_encoded = tokenizer(s, return_tensors='pt')
            t_output = model(**t_encoded)  
            scores = softmax(t_output.logits.detach().numpy(), axis=1) 
            
            # Create a DataFrame to store the scores for this sentence
            emotions = pd.DataFrame(scores, columns=[dict_labels[i] for i in range(3)]) 
            emotions['sentence_id'] = idx_s  
            emotions['Wikipedia_movie_ID'] = [df_extended.iloc[idx]['Wikipedia_movie_ID']] * len(emotions)  # Add movie ID

            big_df = pd.concat([big_df, emotions], ignore_index=True)

    return big_df

"""
Interpolates emotion values over a fixed number of timesteps for a given movie data.
"""
def interpolate_emotions(movie_data, target_timesteps):
    
    emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"] 

    # Define the original positions based on the length of the input data
    original_positions = np.arange(len(movie_data))
    # Define the target positions for interpolation based on the target timesteps
    target_positions = np.linspace(0, len(movie_data) - 1, target_timesteps)
    
    # Initialize a dictionary to store the interpolated data
    interpolated_data = {emotion: [] for emotion in emotions}
    # Add the Wikipedia_movie_ID to the dictionary, repeating it for each target timestep
    interpolated_data["Wikipedia_movie_ID"] = [movie_data["Wikipedia_movie_ID"].iloc[0]] * target_timesteps
    
    # Interpolate each emotion's values over the target positions
    for emotion in emotions:
        # Create a linear interpolation function for the current emotion
        interp_function = interp1d(original_positions, movie_data[emotion], kind="linear", fill_value="extrapolate")
        interpolated_data[emotion] = interp_function(target_positions)

    return pd.DataFrame(interpolated_data)


"""
Perform interpolation for all movies
"""
def interpolate_df(df, target_timesteps=20):
    interpolated_movies = []

    for movie_id, movie_data in df.groupby("Wikipedia_movie_ID"):
        # Apply the interpolate_emotions function to get a fixed number of timesteps for this movie
        interpolated_movie = interpolate_emotions(movie_data, target_timesteps)
        interpolated_movies.append(interpolated_movie)

    interpolated_df = pd.concat(interpolated_movies, ignore_index=True)

    # Add a timestep column to indicate the timestep index within each movie
    interpolated_df['timestep'] = interpolated_df.groupby("Wikipedia_movie_ID").cumcount()
    return interpolated_df