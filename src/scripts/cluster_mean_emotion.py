import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from scipy.stats import pearsonr


#define colors for the cluster, the genre/category and for the emotions
cluster_colors = {
    1: "#1f77b4",  # Blue
    2: "#ff7f0e",  # Orange
    3: "#2ca02c",  # Green
    4: "#d62728",  # Red
    5: "#9467bd",  # Purple
    6: "#8c564b",  # Brown
    7: "#e377c2",  # Pink
}
genre_colors = {
    "Action/Adventure": "#6699FF",  # Soft Blue
    "Comedy": "#FFCC66",           # Soft Yellow/Orange
    "Drama": "#66CC99",            # Soft Teal
    "Family/Animation": "#FF9999", # Soft Coral/Red
    "Fantasy/Sci-Fi": "#CC99FF",   # Soft Lavender
    "Horror": "#996666",           # Muted Brown
    "Romance": "#FFB3CC",          # Soft Pink
    "Thriller": "#A9A9A9"          # Neutral Gray
}


# Define colors for emotions
emotion_colors = {
    "anger": "#FF6666",    # Medium red
    "disgust": "#B266FF",  # Medium purple
    "fear": "#FFB266",     # Medium orange
    "joy": "#66FF66",      # Medium green
    "neutral": "#A9A9A9",  # Medium gray
    "sadness": "#66B2FF",  # Medium blue
    "surprise": "#FFFF66"  # Medium yellow
}

film_exemple = {1: 'Into the Night', 2:'Gross Anatomy', 3: 'A Good Day to Have an Affair', 4: 'The Second Woman', 5:'Deeper and Deeper', 6:'Lucker', 7:'Asterix in Britain'}
cluster_emotion = {1: 'anger', 2:'joy', 3: 'surprise', 4: 'sadness', 5:'fear', 6:'disgust', 7:'neutral'}


emotion_columns = ['fear', 'sadness', 'surprise', 'neutral', 'disgust', 'anger', 'joy']


# Define the paths to the data files
DATA_PATH = "data/emotions_interpolated_20.pkl"  # Path to the CSV file containing sentence emotions data
DATA_PATH_MOVIE_METADATA = "data/final_dataset.pkl"  # Path to the pickle file containing movie metadata

def load_data():
    df_emotions = pd.read_pickle(DATA_PATH)
    with open(DATA_PATH_MOVIE_METADATA, 'rb') as f:
        df_metadata = pickle.load(f)
    return df_emotions, df_metadata


def compute_data(df_emotions, df_metadata):
    # Compute inertias and silhouette scores to find best k
    
    data = df_emotions.drop('timestep', axis=1).groupby('Wikipedia_movie_ID')[emotion_columns].mean()
    data = data.merge(df_metadata[['Wikipedia_movie_ID', 'category']], on='Wikipedia_movie_ID')
    data = data.explode('category')
    data_with_category = data.dropna()
    #a movie can be in more than 1 categories, so we need to keep only 1 sample per movie (hence drop_duplicates())
    data = data_with_category.drop(['category'], axis=1).drop_duplicates()
    data_with_category = data_with_category['category']
    return data, data_with_category

def compute_kmeans(data, df_metadata):
    scaler = StandardScaler()
    # scale the data
    data_scaled = scaler.fit_transform(data.drop('Wikipedia_movie_ID', axis=1))
    data_kmeans = data.copy(deep=True)

    n_clusters = 7
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=500)
    clusters = kmeans.fit_predict(data_scaled)

    data_kmeans['cluster_km'] = clusters+1

    pca = PCA(n_components=3)
    data_3d = pca.fit_transform(data_kmeans.drop(['Wikipedia_movie_ID', 'cluster_km'], axis=1))
    data_kmeans['PCA1_km'] = data_3d[:, 0]
    data_kmeans['PCA2_km'] = data_3d[:, 1]
    data_kmeans['PCA3_km'] = data_3d[:, 2]
    data_kmeans = data_kmeans.merge(df_metadata[['Wikipedia_movie_ID', 'category']], on='Wikipedia_movie_ID')
    return data_kmeans

def plot_elbow_and_silhouette(data):
    scaler = StandardScaler()
    # scale the data
    data_scaled = scaler.fit_transform(data.drop('Wikipedia_movie_ID', axis=1))
    
    inertias = []
    silhouette_scores = []
    k_values = range(1, 12)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data_scaled)
        
        inertias.append(kmeans.inertia_)
        if k > 1:  # k=1 does not work with the silhouette score
            silhouette_avg = silhouette_score(data_scaled, labels)
            silhouette_scores.append(silhouette_avg)

    # Create the elbow method plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(k_values),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(dash='dash'),
            marker=dict(symbol='circle', size=8),
        )
    )

    fig.update_layout(
        title='Elbow method',
        xaxis=dict(title='Number of clusters (k)', tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(title='Inertia'),
        template='plotly_white',
        showlegend=True
    )

    # Create the silhouette score plot
    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=list(k_values[1:]),
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette score',
            line=dict(color='green', dash='dash'),
            marker=dict(symbol='circle', size=8),
        )
    )

    fig2.update_layout(
        title='Silhouette score',
        xaxis=dict(title='Number of clusters (k)', tickmode='linear', tick0=2, dtick=1),
        yaxis=dict(title='Silhouette score'),
        template='plotly_white',
        showlegend=True
    )

    # Combine both plots in a subplot layout
    from plotly.subplots import make_subplots

    combined_fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Elbow method", "Silhouette score"),
        horizontal_spacing=0.15
    )

    combined_fig.add_trace(fig.data[0], row=1, col=1)
    combined_fig.add_trace(fig2.data[0], row=1, col=2)

    combined_fig.update_layout(
        title='Cluster evaluation metrics',
        template='plotly_white',
        showlegend=False,
        height=400, width=900
    )

    combined_fig.show()
    
def visualize_pca_on_kmeans(data_kmeans):
    plotly_data = pd.DataFrame({
        'PCA1_km': data_kmeans['PCA1_km'],
        'PCA2_km': data_kmeans['PCA2_km'],
        'PCA3_km': data_kmeans['PCA3_km'],
        'Cluster_km': (data_kmeans['cluster_km']+1).astype(str),
        'Wikipedia_movie_ID': data_kmeans['Wikipedia_movie_ID'],
        'category': data_kmeans['category']
    })

    cluster_colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # Créer un mapping entre les clusters et leurs couleurs
    clusters = sorted(plotly_data['Cluster_km'].unique())  # S'assurer que les clusters sont triés
    color_discrete_map = {cluster: color for cluster, color in zip(clusters, cluster_colors_list)}

    # Générer le graphique 3D
    fig = px.scatter_3d(
        plotly_data,
        x='PCA1_km',
        y='PCA2_km',
        z='PCA3_km',
        color='Cluster_km',
        title='Clustering emotions in 3D (KMeans)',
        labels={'Cluster_km': 'Cluster'},
        opacity=0.5,
        width=1000,
        height=800,
        hover_data=["Wikipedia_movie_ID", "category"],
        color_discrete_map=color_discrete_map  # Utiliser le mapping des couleurs
    )

    # Assurer que les légendes suivent l'ordre des clusters
    fig.update_layout(legend=dict(traceorder="normal"))

    fig.show()
        


def visualize_prop_genre_cluster(data_with_category, data_kmeans):
    # global category distribution
    labels, counts = np.unique(data_with_category, return_counts=True)
    global_colors = [genre_colors[label] for label in labels]

    # cluster-specific category distribution
    data_kmeans = data_kmeans.explode('category')
    category_per_cluster = data_kmeans[['cluster_km', 'category']].groupby('cluster_km').value_counts()
    clusters = category_per_cluster.index.get_level_values('cluster_km').unique()

    fig = go.Figure()

    # Add the overall distribution pie chart (left pie chart, always visible)
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=counts,
            hole=0.3,
            marker=dict(colors=global_colors),
            name="Overall distribution",
            textinfo='label+percent',
            domain=dict(x=[0, 0.5])  # Left half of the figure
        )
    )

    # add cluster-specific pie charts (right pie chart, with cluster 1 visible by default)
    for i, cluster in enumerate(clusters):
        cluster_data = category_per_cluster.loc[cluster]
        cluster_categories = cluster_data.index.get_level_values('category')
        genre_color = [genre_colors[category] for category in cluster_categories]

        fig.add_trace(
            go.Pie(
                labels=cluster_categories,
                values=cluster_data.values,
                hole=0.3,
                marker=dict(colors=genre_color),
                name=f"Cluster {cluster}",
                textinfo='label+percent',
                visible=(i == 0),  # Make cluster 1 visible by default
                domain=dict(x=[0.5, 1])  # Right half of the figure
            )
        )

    # Create buttons for dropdown (only for clusters)
    buttons = []
    for i, cluster in enumerate(clusters):
        visibility = [True] + [j == i for j in range(len(clusters))]
        buttons.append(
            dict(
                label=f"Cluster {cluster}",
                method="update",
                args=[{"visible": visibility},
                    {"title": f"Distribution of genre: overall & cluster {cluster}"}]
            )
        )

    # Update layout with dropdown menu
    fig.update_layout(
        title="Distribution of genre: overall & cluster 1",
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.5,  # Centered above the chart
                xanchor="center",
                y=1.2,  # Positioned higher above the charts
                yanchor="top"
            )
        ]
    )

    # Add margin to avoid overlapping with dropdown
    fig.update_layout(
        margin=dict(t=100, b=50),  
        height=600  
    )
    fig.show()


def visualize_prop_emotion_cluser(data_kmeans):    
    data_km  = data_kmeans.drop(['PCA1_km', 'PCA2_km', 'PCA3_km'], axis=1).copy()
    mean_emotions_per_cluster = data_km.drop(['category', 'Wikipedia_movie_ID'], axis=1).groupby('cluster_km').mean()

    fig = go.Figure()

    # find the maximum emotion per emotion category across all clusters
    max_emotion_per_category = mean_emotions_per_cluster.idxmax(axis=0)

    for emotion in mean_emotions_per_cluster.columns:
        # compute border width (highlight only if max)
        highlighted_borders = [2 if max_emotion_per_category[emotion] == cluster else 0 
                            for cluster in mean_emotions_per_cluster.index]
        fig.add_trace(
            go.Bar(
                x=(mean_emotions_per_cluster.index ),
                y=mean_emotions_per_cluster[emotion],
                name=emotion.capitalize(),
                marker=dict(color=emotion_colors[emotion],
                            line=dict(color="black", width=highlighted_borders)),
                showlegend=False if emotion == "anger" else True  # hide legend for anger bar
            )
        )

    # add a separate trace for anger in the legend without the border
    # no a really clean way of doing so, but only way I found that work 
    fig.add_trace(
        go.Bar(
            x=[None],  
            y=[None],  
            name="Anger", 
            marker=dict(color=emotion_colors["anger"], line=dict(width=0))
        )
    )

    fig.update_layout(
        barmode='stack',
        title='Emotions by cluster (highlighting cluster where each emotion is the highest)',
        xaxis=dict(title='Cluster'),
        yaxis=dict(title='Proportion'),
        legend_title='Emotion',
        height=600,
        width=900
    )

    fig.show()

def visualize_prop_cluster_by_genre(data_kmeans):
    data_kmeans = data_kmeans.explode('category')
    # Group data by category and cluster, and count occurrences
    category_cluster_counts = data_kmeans.groupby(['category', 'cluster_km']).size().reset_index(name='count')
    categories = category_cluster_counts['category'].unique()

    # Create a figure
    fig = go.Figure()

    # Add a Pie chart for each category
    for category in categories:
        category_data = category_cluster_counts[category_cluster_counts['category'] == category]
        cluster_labels = category_data['cluster_km'].astype(str)
        cluster_counts = category_data['count']
        cluster_colors_list = [cluster_colors[cluster] for cluster in category_data['cluster_km']]

        fig.add_trace(
            go.Pie(
                labels=cluster_labels,
                values=cluster_counts,
                name=category,
                textinfo='percent+label',
                marker=dict(colors=cluster_colors_list),
                visible=False  # Initially hide all traces
            )
        )

    # Make the first category visible
    fig.data[0].visible = True

    # Add dropdown buttons for each category
    buttons = []
    for i, category in enumerate(categories):
        buttons.append(dict(
            label=category,
            method='update',
            args=[{'visible': [j == i for j in range(len(categories))]},  # Show only the selected category
                {'title': f"Cluster repartition for {category}"}]  # Update title
        ))

    # Add the dropdown menu to the layout
    fig.update_layout(
        updatemenus=[{
            'active': 0,
            'buttons': buttons,
            'x': 0.5,
            'xanchor': 'center',
            'y': 1.2,
            'yanchor': 'top'
        }],
        title="Cluster repartition for Action/Adventure",
        height=600, 
        width=1000,
        showlegend=True
    )

    fig.show()


def visualize_arc_movie_cluster_overall(df_emotions, data_kmeans, df_metadata, cluster=1):
    df_em_sum_cluster = data_kmeans[['Wikipedia_movie_ID', 'category', 'cluster_km']].merge(df_emotions, on='Wikipedia_movie_ID').merge(df_metadata[['summary', 'Wikipedia_movie_ID', 'Movie_name']], on='Wikipedia_movie_ID')
    mean_cluster_category = df_em_sum_cluster.groupby(['cluster_km', 'timestep'])[emotion_columns].mean()
    x = np.arange(0, 20)

    emotion = cluster_emotion.get(cluster)
    # filter movie to get the 20 timestep of the movie associated to the cluster 
    movie = df_em_sum_cluster[df_em_sum_cluster['Movie_name'] == film_exemple.get(cluster)].iloc[:20]
    # get the summary
    summary = movie.summary.iloc[0]
    # mean emotion by timestep over every movies
    overall_emotion = df_em_sum_cluster.groupby('timestep')[emotion].mean()
    # mean emotion by timestep over movies in the cluster chosen
    mean_cluster_emotion = mean_cluster_category.loc[cluster][emotion]  
    # emotion of the movie chosen
    movie_emotion = movie[emotion]  # Émotion spécifique du film

    fig = make_subplots(
        rows=1, cols=2, 
        column_widths=[0.65, 0.35],  
        specs=[[{"type": "scatter"}, {"type": "table"}]], 
        subplot_titles=[f"{emotion.capitalize()} arc", f"{film_exemple.get(cluster)} summary"]
    )

    # plot for the movie
    fig.add_trace(
        go.Scatter(
            x=x, 
            y=movie_emotion, 
            mode='lines+markers',
            line=dict(dash='dot', color=emotion_colors.get(emotion, 'red')),
            marker=dict(symbol='circle', size=8),
            name=f"{emotion.capitalize()} arc of {film_exemple.get(cluster)}"
        ),
        row=1, col=1
    )

    # plot for the mean emotion of the cluster
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean_cluster_emotion,
            mode='lines+markers',
            line=dict(dash='dash', color='blue'),
            marker=dict(symbol='triangle-up', size=8),
            name=f"{emotion.capitalize()} arc of cluster {cluster}"
        ),
        row=1, col=1
    )

    # plot for the mean emotion over all movies
    fig.add_trace(
        go.Scatter(
            x=x, 
            y=overall_emotion, 
            mode='lines+markers',
            line=dict(dash='dash', color='black'),
            marker=dict(symbol='cross', size=8),
            name=f"Mean arc of {emotion}"
        ),
        row=1, col=1
    )

    # table for the summary
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Summary"], 
                align="center", 
                fill_color="paleturquoise",
                font=dict(size=14)
            ),
            cells=dict(
                values=[summary], 
                align="left", 
                fill_color="lavender",
                font=dict(size=12)
            ),
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=dict(
            text=f"{emotion.capitalize()} arc: comparing all films, cluster {cluster}, and ‘{film_exemple.get(cluster)}’ (from cluster {cluster})",
            x=0.5,  
            xanchor="center"
        ),
        xaxis_title="Timestep",
        yaxis_title=f"{emotion.capitalize()} intensity",
        legend=dict(
            x=0.5,  
            y=-0.2, 
            xanchor="center",
            yanchor="top",  
            orientation="h",
            font=dict(size=12)
        ),
        margin=dict(l=40, r=40, t=80, b=100),
        height=400,
        width=1000,
        template="plotly_white",
        showlegend=True
    )

    fig.show()

