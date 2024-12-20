import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.figure_factory as ff
import pandas as pd
import nltk
from IPython.display import HTML
from nltk.tokenize import sent_tokenize

def load_data():
   # loading the data
    file_path = "data/final_dataset.pkl"
    file_path_raw = "data/emotions_data_raw.pkl"
    final_dataset = pd.read_pickle(file_path)
    raw_emotions = pd.read_pickle(file_path_raw)
    file_path_interpolated = "data/emotions_interpolated_20.pkl"
    emotions_interpolated = pd.read_pickle(file_path_interpolated)
    return final_dataset, raw_emotions, emotions_interpolated

def emotions_colors():
    # Map emotions to colors
    emotion_colors = {
        "anger": "#FF6666",    # Medium red
        "disgust": "#B266FF",  # Medium purple
        "fear": "#FFB266",     # Medium orange
        "joy": "#66FF66",      # Medium green
        "neutral": "#A9A9A9",  # Medium gray
        "sadness": "#66B2FF",  # Medium blue
        "surprise": "#FFFF66"  # Medium yellow
    }
    return emotion_colors

def plot_length_summaries(save=False):
    final_dataset, raw_emotions, emotions_interpolated = load_data()

    df_plot_length_sentences = final_dataset["summary"].dropna().apply(
    lambda x: len(sent_tokenize(x))
    )

    fig = px.histogram(df_plot_length_sentences, x="summary", marginal = 'box') 
    fig.update_layout(title="Distribution of the number of sentences in the summaries",
                  xaxis_title="Number of sentences",
                  yaxis_title="Number of summaries")
    fig.show()

    if save:
        fig.write_html("summary_length_distribution.html")

def plot_example_summary_classification(save=False):
    
    #load
    final_dataset, raw_emotions, emotions_interpolated = load_data()
    movie_id = 3333
    raw_emotions_selected = raw_emotions[raw_emotions["Wikipedia_movie_ID"] == movie_id]
    
    #colors
    emotion_colors = emotions_colors()

    raw_emotions_selected["dominant_emotion"] = raw_emotions_selected[[
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
    ]].idxmax(axis=1)

    raw_emotions_selected["color"] = raw_emotions_selected["dominant_emotion"].map(emotion_colors)

    #create a table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Sentence</b>", "<b>Dominant emotion</b>"],
            fill_color='lightgrey',
            align='left',
            font=dict(size=16)
        ),
        cells=dict(
            values=[
                raw_emotions_selected["sentence"],
                raw_emotions_selected["dominant_emotion"]
            ],
            fill_color=[
                ['white'] * len(raw_emotions_selected),  
                raw_emotions_selected["color"]  
            ],
            align='left',
            font=dict(size=14)
        )
    )])

    fig.update_layout(
        title="Sentences with dominant emotions highlighted",
        title_font=dict(size=20),
        width=1150,  
        height=900 
    )

    fig.show()
    if save:
        fig.write_html("sentences_with_dominant_emotions_highlighted.html")

def smooth_line(x, y, points=20):
    """Interpolates and smooths the line."""
    x_smooth = np.linspace(min(x), max(x), points)
    y_smooth = np.interp(x_smooth, x, y)
    return x_smooth, y_smooth

def plot_example_emotions_interpolated(save=False):
    movie_id = 3333
    final_dataset, raw_emotions, emotions_interpolated = load_data()
    df = emotions_interpolated[emotions_interpolated["Wikipedia_movie_ID"] == movie_id]
  
    fig = go.Figure()

    # smoothed line
    emotion_colors = emotions_colors()
    for emotion, color in emotion_colors.items():
        x_smooth, y_smooth = smooth_line(df["timestep"], df[emotion])
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode="lines",
            name=emotion.capitalize(),
            line=dict(color=color, width=2)
        ))

    fig.update_layout(
        title="Emotional arc of the movie",
        xaxis_title="Timestep",
        yaxis_title="Emotion intensity",
        template="plotly_white",
        width=900,
        height=600,
        legend_title="Emotions"
    )

    fig.show()
    if save:
        fig.write_html("emotional_arc_of_the_movie.html")

def plot_emotions_interpolated_strongest(save=False):
    final_dataset, raw_emotions, emotions_interpolated = load_data()
    emotion_colors = emotions_colors()

    movie_id = 3333
    df = emotions_interpolated[emotions_interpolated["Wikipedia_movie_ID"] == movie_id]

    fig = go.Figure()

    #legend
    for emotion, color in emotion_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color=color, width=2),
            name=emotion
        ))

    #strongest
    strongest_emotions = df[["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]].idxmax(axis=1)

    # staircase lines
    for idx in range(len(df)):
        emotion = strongest_emotions[idx]
        color = emotion_colors[emotion]
        fig.add_trace(go.Scatter(
            x=[df["timestep"][idx], df["timestep"][idx] + 1],
            y=[df[emotion][idx], df[emotion][idx]],
            mode='lines',
            line_shape='hv',
            fill='tozeroy',
            fillcolor=color,
            line=dict(color=color, width=2),
            showlegend=False, 
            name=emotion
        ))

    fig.update_layout(
        title="Strongest emotion highlighted across timesteps",
        xaxis_title="Timestep",
        yaxis_title="Emotion score",
        legend_title="Emotions",
        template="plotly_white",
        xaxis=dict(tickmode="linear", dtick=1),
        yaxis=dict(range=[0, 1], gridcolor="lightgray"),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.2
        )
    )

    fig.show()
    if save:
        fig.write_html("strongest_emotion_highlighted_across_timesteps.html")


def general_emotion_average(save=False):
    final_dataset, raw_emotions, emotions_interpolated = load_data()
    emotion_colors = emotions_colors()

    emotions_interpolated_2 = emotions_interpolated.drop(columns=["timestep"]).groupby("Wikipedia_movie_ID").mean().reset_index()

    # melt the df for box plot visualization
    melted_df = emotions_interpolated_2.melt(
        id_vars=["Wikipedia_movie_ID"],
        var_name="Emotion",
        value_name="Intensity"
    )

    #colors
    melted_df['Color'] = melted_df['Emotion'].map(emotion_colors)

    # boxplots
    fig = px.box(
        melted_df,
        x="Emotion",
        y="Intensity",
        color="Emotion",
        color_discrete_map=emotion_colors,
        title="Average Emotion Intensity Across Movies",
        width=900,
        height=600
    )

    
    fig.update_layout(
        yaxis=dict(
            title="Intensity",
            range=[0, 1], #focus on the 0 to 1 range 
            tickformat=".2f"
        ),
        xaxis=dict(title="Emotion"),
        boxmode="overlay",
        plot_bgcolor="white"
    )

    fig.show()
    if save:
        fig.write_html("average_emotion_intensity_across_movies.html")

def plot_emotion_heatmap_general(save=False):
    final_dataset, raw_emotions, emotions_interpolated = load_data()
    categories_df = final_dataset[["category", "Wikipedia_movie_ID"]]
    average_emotion_df = emotions_interpolated.drop(columns=["timestep"]).groupby("Wikipedia_movie_ID").mean().reset_index() 

    #merge with the categories and grooupby
    categories_exploded = categories_df.explode("category")
    merged_df = pd.merge(categories_exploded, average_emotion_df, on="Wikipedia_movie_ID")
    category_emotion_avg = merged_df.groupby("category").mean().reset_index().drop(columns=["Wikipedia_movie_ID", "neutral"])

    #heatmap
    melted_category_emotions = category_emotion_avg.melt(
        id_vars=["category"],
        var_name="Emotion",
        value_name="Average Intensity"
    )

    fig = px.imshow(
        category_emotion_avg.set_index("category").T,
        labels=dict(x="Category", y="Emotion", color="Average Intensity"),
        title="Average Emotion Intensity per Movie Category",
        color_continuous_scale="Viridis",
        aspect="auto"
    )

    fig.update_layout(
        xaxis=dict(title="Category"),
        yaxis=dict(title="Emotion"),
        coloraxis_colorbar=dict(title="Average Intensity"),
        width=900,
        height=600
    )

    fig.show()
    if save:
        fig.write_html("heatmap_avg_emotion.html") 