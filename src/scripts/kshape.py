import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from tqdm import tqdm
sns.set(style="darkgrid")
nltk.download('punkt')
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from tslearn.clustering import KShape
plt.style.use('seaborn-whitegrid')
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#utils hex to rgba
def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#') 
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) 
    return f"rgba({r}, {g}, {b}, {alpha})"

#function to plot the elbow for k-shape clustering
def plot_elbow_kshape(X, subsample, emotion_title):
    subsample_size = min(subsample, len(X))
    X_subsample = resample(X, n_samples=subsample_size, random_state=0)

    inertias = []
    for k in range(2, 10): 
        kshape_model = KShape(n_clusters=k, verbose=False, random_state=0)
        kshape_model.fit(X_subsample)
        inertias.append(kshape_model.inertia_)

   #plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 10), inertias, 'o-', markerfacecolor='red')
    plt.title(f"Elbow Method for {emotion_title}")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.grid(True, linestyle=":")
    plt.show()

#function to one shot all k-shape elbow plots
def plot_elbow_for_all_emotions(emotion_data, subsample):
    for emotion, X in tqdm(emotion_data.items(), desc="Processing all emotions", unit="emotion"):
        plot_elbow_kshape(X, subsample, emotion_title=emotion)


#function to run the k-shape clustering pipeline
def clustering_k_shape_pipeline(emotion_to_cluster, n_cluster, recalculate_clusters=False):

    # load the data needed
    df_emotions = pd.read_pickle("data/emotions_interpolated_20.pkl")
    df_emotions_raw = pd.read_pickle("data/emotions_data_raw.pkl")
    df_metadata = pd.read_pickle("data/final_dataset.pkl")    
    
    # let's get rid of movies that are too short or too long, firstly to reduce the computational cost
    # and secondly because they are less interesting (extrapolated or squished data)
    df_emotions_raw['sentence_id'] = (df_emotions_raw.groupby('Wikipedia_movie_ID').cumcount() + 1)
    sentence_counts = df_emotions_raw.groupby('Wikipedia_movie_ID').size()
    valid_movies = sentence_counts[(sentence_counts >= 10) & (sentence_counts <= 80)].index
    df_emotions = df_emotions[df_emotions['Wikipedia_movie_ID'].isin(valid_movies)]
    df_emotions.reset_index(drop=True, inplace=True)

    # normalize the data
    emotion_df = df_emotions[[emotion_to_cluster, "Wikipedia_movie_ID", "timestep"]]
    emotion_df = emotion_df.pivot(index='Wikipedia_movie_ID', columns='timestep', values=emotion_to_cluster)
    X = emotion_df.values
    X = TimeSeriesScalerMinMax().fit_transform(X)

    #join with the genre data
    df_genres = df_metadata[['Wikipedia_movie_ID', 'category']]
    df_genres = df_genres[df_genres['Wikipedia_movie_ID'].isin(valid_movies)]


    #to avoid recalculating the clusters, we will load the cluster assignments
    if recalculate_clusters:
        model = KShape(n_clusters=4, verbose=True, random_state=0)
        y = model.fit(X).predict(X)
        y_df = pd.DataFrame({'Wikipedia_movie_ID': df_genres["Wikipedia_movie_ID"], 'cluster_label': y})

    else:
        cluster_assignment = pd.read_pickle("data/emotions_cluster_assignments.pkl")
        y = cluster_assignment[emotion_to_cluster]
    return X, y


#function to plot the cluster evolution for each cluster in an emotion
def plot_cluster_evolution_plotly(X, y, nb_clusters, title, emotion, save=False):
    
    fig = make_subplots(
        rows=1, cols=nb_clusters,
        subplot_titles=[f"Cluster {i}" for i in range(nb_clusters)]
    )

    time_steps = np.arange(20)
    colors = ["#ff6666", "#b266ff", "#ffb266", "#66ff66", "#a9a9a9", "#66b2ff", "#ffff66"]
    
    #iterate over clusters
    for cluster_id in range(nb_clusters):

        # extract movies in the cluster
        cluster_indices = np.where(y == cluster_id)[0]
        cluster_data = X[cluster_indices]

        #calculate std and mean
        avg_evolution = np.mean(cluster_data, axis=0).reshape(-1)
        std_deviation = np.std(cluster_data, axis=0).reshape(-1)

        #plot mean
        fill_color_rgba = hex_to_rgba(colors[cluster_id], alpha=0.2)
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=avg_evolution,
                mode='lines',
                name=f'avg',
                line=dict(color=colors[cluster_id], width=3)
            ),
            row=1, col=cluster_id + 1
        )

        #plot std
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time_steps, time_steps[::-1]]),
                y=np.concatenate([
                    avg_evolution + std_deviation,
                    (avg_evolution - std_deviation)[::-1]
                ]),
                fill='toself',
                fillcolor=fill_color_rgba, 
                line=dict(color='rgba(255,255,255,0)'),
                name=f'std',
                hoverinfo='skip'
            ),
            row=1, col=cluster_id + 1
        )

    #uniform y axis
    for i in range(1, nb_clusters + 1):
        fig.update_yaxes(range=[0, 1], row=1, col=i)

    #global layout
    fig.update_layout(
        title=title,
        xaxis_title="Timesteps",
        plot_bgcolor='rgba(240,240,240,1)',
        showlegend=False, 
        width=2000, 
        height=500 
    )

    if save:
        fig.write_html(f"cluster_evolution_{emotion}.html")
    fig.show()


#function to plot the cluster sizes
def plot_cluster_sizes(cluster_sizes, nb_clusters, emotion, save=False):
    #bar chart
    fig = go.Figure(
        go.Bar(
            x=[f"Cluster {i}" for i in range(nb_clusters)],
            y=cluster_sizes,
            text=[f"{size}" for size in cluster_sizes],
            textposition='auto',
            marker=dict(color=["#ff6666", "#b266ff", "#ffb266", "#66ff66", "#a9a9a9", "#66b2ff", "#ffff66"][:nb_clusters]),
        )
    )
    fig.update_layout(
        title=f"Cluster sizes for the {emotion} emotion",
        xaxis_title="Clusters",
        yaxis_title="Size",
        plot_bgcolor="rgba(240, 240, 240, 1)",
        width=800,
        height=500
    )

    if save:
        fig.write_html(f"cluster_sizes_{emotion}.html")
    fig.show()