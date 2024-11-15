import pandas as pd
import numpy as np 
import re
import ast

# Define data paths constants
PLOT_DATA_PATH = "src/data/plot_summaries.txt"
MOVIE_DATA_PATH = "src/data/movie.metadata.tsv"
CLUSTER_NAME_DATA_PATH = "src/data/name.clusters.txt"
CHARACTER_DATA_PATH = "src/data/character.metadata.tsv"

"""
Function to parse dictionary-like strings in the file and separate keys and values
"""
def parse_dict_column(column):
    parsed_keys = []
    parsed_values = []
    
    for item in column:
        # Convert string representation of dictionary to actual dictionary
        item_dict = ast.literal_eval(item)
        parsed_keys.append(", ".join(item_dict.keys()))
        parsed_values.append(", ".join(item_dict.values()))
    
    return parsed_keys, parsed_values


"""
Function to parse dictionary-like strings in the file and separate keys and values
"""
def parse_dict_column(column):
    parsed_keys = []
    parsed_values = []
    
    for item in column:
        # Convert string representation of dictionary to actual dictionary
        item_dict = ast.literal_eval(item)
        parsed_keys.append(", ".join(item_dict.keys()))
        parsed_values.append(", ".join(item_dict.values()))
    
    return parsed_keys, parsed_values


"""
Function to load and clean movie metadata, returns a dataframe with cleaned movie data
"""
def load_and_clean_movie_data():
    # Load the movie metadata
    df_movie_metadata = pd.read_csv(
        MOVIE_DATA_PATH, sep='\t', header=None, 
        names=[
            'Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_name', 
            'Movie_release_date', 'Movie_box_office_revenue', 'Movie_runtime',
            'Movie_languages_(Freebase ID:name tuples)', 'Movie_countries_(Freebase ID:name tuples)',
            'Movie_genres_(Freebase ID:name tuples)'
        ]
    )
    
    # Parse 'languages', 'countries', and 'genres' columns
    df_movie_metadata['id_movie_languages'], df_movie_metadata['Movie_languages'] = parse_dict_column(df_movie_metadata['Movie_languages_(Freebase ID:name tuples)'])
    df_movie_metadata['id_Movie_countries'], df_movie_metadata['Movie_countries'] = parse_dict_column(df_movie_metadata['Movie_countries_(Freebase ID:name tuples)'])
    df_movie_metadata['id_Movie_genres'], df_movie_metadata['Movie_genres'] = parse_dict_column(df_movie_metadata['Movie_genres_(Freebase ID:name tuples)'])

    # Convert dates to datetime and extract the year
    df_movie_metadata['Movie_release_date'] = pd.to_datetime(df_movie_metadata['Movie_release_date'], errors='coerce').dt.year

    # Select and rename the columns as required
    cleaned_df_movie_metadata = df_movie_metadata[[
        'Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_name', 'Movie_release_date', 
        'Movie_box_office_revenue', 'Movie_runtime', 'id_movie_languages', 
        'Movie_languages', 'id_Movie_countries', 'Movie_countries', 
        'id_Movie_genres', 'Movie_genres'
    ]]

    # Drop unwanted id columns
    columns_to_drop = ['id_movie_languages', 'id_Movie_genres', 'id_Movie_countries']
    cleaned_df_movie_metadata = cleaned_df_movie_metadata.drop(columns=columns_to_drop)

    # Convert floats of box office and years to nullable integers, keeping NaNs as np.nan
    cleaned_df_movie_metadata['Movie_box_office_revenue'] = pd.to_numeric(cleaned_df_movie_metadata['Movie_box_office_revenue'], errors='coerce')
    cleaned_df_movie_metadata['Movie_release_date'] = pd.to_numeric(cleaned_df_movie_metadata['Movie_release_date'], errors='coerce')

    #need to drop a line that has some weird encodings 
    cleaned_df_movie_metadata = cleaned_df_movie_metadata.map(lambda x: x.encode('utf-8', 'ignore').decode('utf-8') if isinstance(x, str) else x)

    # Replace any <NA> with np.nan for uniform NaNs
    cleaned_df_movie_metadata = cleaned_df_movie_metadata.replace({pd.NA: np.nan})

    return cleaned_df_movie_metadata


"""
Helper function to clean plot texts from unwanted annotations and tags
"""
def clean_plot(txt):

    #Remove URLs
    txt = re.sub(r"http\S+|www\.\S+", '', txt)

    #Remove HTML tags
    txt = re.sub(r'<.*?>', '', txt)

    #Remove {{annotations}}
    txt = re.sub(r'\{\{.*?\}\}', '', txt)

    #Remove the ([[ annotation that is never closed
    txt = re.sub(r'\(\[\[', '', txt)

    #Remove the synopsis from context
    txt = re.sub(r'Synopsis from', '', txt)

    #Remove <ref...}} tags
    txt = re.sub(r'<ref[^}]*}}', '', txt)

    return txt

#loading only clean plots
def load_and_clean_plots_data():
    df_plot_summaries = pd.read_csv(PLOT_DATA_PATH, sep='\t', header=None,  names=['Wikipedia_movie_ID', 'summary'])
    df_plot_summaries['summary'] = df_plot_summaries['summary'].apply(clean_plot)
    return df_plot_summaries

"""
Pipeline to load and clean movie metadata, returns a dataframe with cleaned movie data
"""
def movie_data_cleaning_pipeline():
    df_movie_plots = load_and_clean_plots_data()
    df_movie_metadata = load_and_clean_movie_data()
    df_movie_data = df_movie_plots.merge(df_movie_metadata, on='Wikipedia_movie_ID', how='outer')
    return df_movie_data


"""
Function to load cluster data, returns a dataframe with cleaned cluster data
"""
def load_and_clean_cluster_data():
    
    #get cluster data
    file_path = "src/data/tvtropes.clusters.txt"
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Replace `{"char": ` with a simpler delimiter like a tab
    lines = [line.replace('{"char": ', '').replace(', "movie": ', '\t')
            .replace(', "id": ', '\t').replace('}', '')
            .replace(', "actor": ', '\t').replace('\t\t', '\t') for line in lines]

    with open("src/data/pro_tvtropes.clusters.txt", "w") as file:
        file.writelines(lines)
    file_path = "src/data/pro_tvtropes.clusters.txt"

    # Load the processed file
    df_clusters_tvtropes = pd.read_csv(file_path, sep='\t', header=None, names=['character_types', 'character', 'movie','Freebase_character/actor_map_ID','Actor_name'])

    # Replace any <NA> with np.nan for uniform NaNs
    df_clusters_tvtropes = df_clusters_tvtropes.replace({pd.NA: np.nan})
    return df_clusters_tvtropes

"""
Function to load character metadata, returns a dataframe with cleaned cluster data
"""
def load_and_clean_character_data():
    
    # load from csv
    df_clusters_name = pd.read_csv(CLUSTER_NAME_DATA_PATH, sep='\t', header=None, names=['unique_character_name', 'Freebase_character/actor_map_ID'])
    df_character_metadata = pd.read_csv(CHARACTER_DATA_PATH, sep='\t', header=None, 
                                    names=[
                                        'Wikipedia_movie_ID','Freebase_movie_ID', 'Movie_release_date',
                                        'Character_name', 'Actor_date_of_birth', 'Actor_gender',
                                        'Actor_height_(in meters)', 'Actor_ethnicity_(Freebase ID)',
                                        'Actor_name', 'Actor_age_at_movie_release', 'Freebase_character/actor_map_ID',
                                        'Freebase_character_ID', 'Freebase_actor_ID'
                                        ])


    #keep only year of birth
    df_character_metadata['Actor_date_of_birth'] = pd.to_datetime(df_character_metadata['Actor_date_of_birth'], errors='coerce').dt.year

    #keep only year of release
    df_character_metadata['Movie_release_date'] = pd.to_datetime(df_character_metadata['Movie_release_date'], errors='coerce').dt.year

    #merge character info with their unique names
    df_character_metadata = df_character_metadata.merge(df_clusters_name, on='Freebase_character/actor_map_ID', how='outer')

    #Check actor age is bigger equal 0 and smaller than 110, else replace with NaN
    df_character_metadata['Actor_age_at_movie_release'] = df_character_metadata['Actor_age_at_movie_release'].apply(lambda x: x if 0 <= x <= 110 else np.nan)

    # Replace any <NA> with np.nan for uniform NaNs
    df_character_metadata = df_character_metadata.replace({pd.NA: np.nan})
        
    return df_character_metadata


"""
Pipeline to load and clean character data, returns a dataframe with cleaned character data
"""
def character_data_cleaning_pipeline():
    df_character_metadata = load_and_clean_character_data()
    df_clusters_tvtropes = load_and_clean_cluster_data()
    
    # Merge the two dataframes
    df_character_data = df_character_metadata.merge(df_clusters_tvtropes, on=['Freebase_character/actor_map_ID', 'Actor_name'], how='outer')
    df_clusters_name = pd.read_csv(CLUSTER_NAME_DATA_PATH, sep='\t', header=None, names=['unique_character_name', 'Freebase_character/actor_map_ID'])
    df_character_data = df_character_data.merge(df_clusters_name, on=['Freebase_character/actor_map_ID', "unique_character_name"], how='outer')
    return df_character_data