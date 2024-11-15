import pandas as pd
import numpy as np

DATA_PATH = "data/our_movie_data.csv"

GENRE_MAPPING = {
    'Action/Adventure': [
        'Action', 'Action Comedy', 'Action Thrillers', 'Action/Adventure', 
        'Martial Arts Film', 'Samurai cinema', 'Adventure', 'Adventure Comedy',
        'Swashbuckler films', 'Costume Adventure', 'Western', 'Hybrid Western', 'Epic', 'Epic Western', 
        'Family-Oriented Adventure', 'Historical Epic', 'Indian Western', 'Spaghetti Western', 'Revisionist Western',
        'Escape Film', 'Glamorized Spy Film', 'Prison escape', 'Movies About Gladiators'
    ],
    'Comedy': [
        'Comedy', 'Comdedy', 'Comedy Thriller', 'Comedy Western', 'Comedy film', 
        'Comedy horror', 'Comedy of Errors', 'Comedy of manners', 'Comedy-drama', 
        'Workplace Comedy', 'Stand-up comedy', 'Screwball comedy',
        'Parody', 'Slapstick', 'Gross out', 'Gross-out film', 'Black comedy', 
        'Sex comedy', 'Farce', 'Satire', 'Dark comedy', 'Camp', 'Domestic Comedy', 
        'Courtroom Comedy', 'Horror Comedy', 'Humour', 'Beach Party film', 
        'Buddy Picture', 'Buddy film', 'Heavenly Comedy', 'Media Satire', 'Ealing Comedies'
    ],
    'Drama': [
        'Drama', 'Addiction Drama', 'Childhood Drama', 'Courtroom Drama', 
        'Historical drama', 'Inspirational Drama', 'Marriage Drama', 
        'Medical fiction', 'Melodrama', 'Psychological drama', 
        'Tragicomedy', 'Political drama', 'Prison film', 'Family Drama', 'Legal drama', 
        'Family & Personal Relationships', 'Social problem film', 'Interpersonal Relationships',
        'Social issues', 'Illnesses & Disabilities', 'Existentialism', 'Costume drama', 'Prison', 'Tragedy'
    ],
    'Fantasy/Sci-Fi': [
        'Fantasy', 'Fantasy Adventure', 'Fantasy Comedy', 'Fantasy Drama', 
        'Mythological Fantasy', 'Sword and Sorcery', 'Sword and sorcery films', 
        'Fairy tale', 'Supernatural', 'Surrealism', 'Absurdism', 'Magic Realism',
        'Sci-Fi', 'Sci-Fi Adventure', 'Sci-Fi Thriller', 
        'Science Fiction', 'Space opera', 'Future noir', 'Space western', 'Cyberpunk', 
        'Alien invasion', 'Alien Film', 'Dystopia', 'Utopia', 'Time travel', 
        'Steampunk', 'Post-apocalyptic', 'Robots', 'Science fiction Western', 'Punk rock', 
        'Revisionist Fairy Tale', 'Sci Fi Pictures original films', 'Apocalyptic and post-apocalyptic fiction'
    ],
    'Horror': [
        'Horror', 'Psychological horror', 'Natural horror films', 
        'Monster movie', 'Slasher', 'Splatter film', 'Zombie Film', 'Vampire movies', 
        'Werewolf fiction', 'Creature Film', 'Body Horror', 'Ghost Story', 
        'Demonic child', 'Gothic Film', 
        'Supernatural Horror', 'Sci-Fi Horror', 'Costume Horror', 'Period Horror', 'Road-Horror', 
        'Haunted House Film', 'Sexploitation', 'Softcore Porn', 
        'Doomsday film', 'Monster', 'Plague'
    ],
    'Romance': [
        'Romance', 'Romantic comedy', 'Romantic drama', 'Romantic fantasy', 
        'Romantic thriller', 'Love Story', 'Erotic Drama', 'Erotic thriller', 
        'Romance Film', 'Coming-of-age film', 'Chick flick', 'Erotica'
    ],
    'Thriller': [
        'Thriller', 'Political thriller', 'Psychological thriller', 
        'Crime Thriller', 'Mystery', 'Spy', 'Suspense', 'Whodunit', 'Conspiracy fiction', 
        'Detective', 'Chase Movie', 'Crime Fiction', 'Gangster Film', 'Master Criminal Films', 
        'Detective fiction', 'Political satire', 'Law & Crime', 'Gangster', 
        'Buddy cop'
    ],
    'Documentary': [
        'Documentary', 'Docudrama', 'Historical Documentaries', 
        'Political Documetary', 'Nature', 'True Crime', 'Music Documentary', 
        'Science Documentary', 'Sports Documentary', 'Cultural Documentary', 
        'Education', 'Anthropology', 'History', 'Environmental Science', 'Animals',
        'Archives and records', 'Culture & Society', 'Educational', 'Graphic & Applied Arts', 
        'Health & Fitness', 'Inventions & Innovations', 'Language & Literature', 'Journalism', 
        'Libraries and librarians', 'Linguistics', 'Media Studies', 'Political cinema', 'Rockumentary',
        'Archaeology', 'Travel', 'World History'
    ],
    'Family/Animation': [
        'Animation', 'Animated cartoon', 'Anime', 'Computer Animation', 
        'Clay animation', 'Silhouette animation', 'Animated Musical', 
        'Stop motion', 'Supermarionation', 'Live action', 'Children\'s', 'Children\'s Fantasy', 
        'Family Film', 'Children\'s Entertainment', "Children's Issues", "Children's/Family",
        'Christmas movie'
    ],
    'Musical': [
        'Musical', 'Musical Drama', 'Musical comedy', 'Jukebox musical', 
        'Film-Opera', 'Operetta', 'Backstage Musical', 'Concert film', 
        'Singing cowboy', 'Dance', 'Instrumental Music', 'Music'
    ],
    'War/Crime': [
        'War film', 'Combat Films', 'Military', 'Anti-war film', 'World War II', 
        'Gulf War', 'Crime', 'Crime Comedy', 'Crime Drama',
        'Heist', 'Police Procedural',  'Juvenile Delinquency Film', 'Cold War',
        'War effort', 'Revenge', 'Outlaw biker film', 'Outlaw', 'Patriotic film',
        'Private military company', 'Cavalry Film', 'British Empire Film'
    ],
    'Sports': [
        'Sports', 'Boxing', 'Baseball', 'Basketball', 'Football', 'Soccer', 
        'Auto racing', 'Horse racing', 'Extreme Sports', 'Parkour in popular culture', 
        'Breakdance'
    ],
    'LGBTQ+': [
        'LGBT', 'Gay', 'Gay Interest', 'Gay Themed', 'Homoeroticism', 
        'Queer Cinema', 'New Queer Cinema', 'Gay pornography', 'Pornographic movie', 
        'Pinku eiga', 'Pornography', 'Feminist Film'
    ],
    'Others': [
        'B-Movie', 'Mockumentary', 'Mondo film', 'Cult', 'Art film', 'Film Noir', 
        'New Wave', 'Czechoslovak New Wave', 'Kitchen sink realism', 'Neo-noir', 
        'Blaxploitation', 'Bollywood', 'World cinema', 'Z movie', 'Experimental film', 'Avant-garde',
        'Expressionism', 'Essay Film', 'Abstract', 'Structural Film', 'Americana', 'Coming of age', 'Biographical film',
        'Biography', 'Historical fiction', 'Period piece', 'Biker Film', 'B-Western', 'B-movie', 'C-Movie', 
        'Caper story', 'Feature film', 'Fictional film', 'Filipino', 'Filipino Movies', 'Film', 'Film adaptation',
        'Film noir', 'Film à clef', 'Filmed Play', 'Finance & Investing', 'Foreign legion', 'Gender Issues',
        'Giallo', 'Goat gland', 'Hardcore pornography', 'Heaven-Can-Wait Fantasies', 'Hip hop movies', 'Holiday Film',
        'Exploitation', 'Fan film', 'Female buddy film', 'Film & Television History', 'Film serial', 'Horror comedy',
        'Mumblecore', 'Neorealism', 'News', 'Ninja movie', 'Northern', 'Nuclear warfare', 'Point of view shot',
        'Pre-Code', 'Psycho-biddy', 'Race movie', 'Reboot', 'Remake', 'Road movie', 'Roadshow theatrical release',
        'School story', 'Short Film', 'Silent film', 'Slice of life story', 'Sponsored film', 'Star vehicle', 
        'Statutory rape', 'Stoner film', 'Superhero', 'Superhero movie', 'Sword and Sandal', 'Sword and sorcery',
        'Tamil cinema', 'Teen', 'Television movie', 'The Netherlands in World War II', 'Theremin music', 'Therimin music',
        'Acid western', 'Adult', 'Airplanes and airports', 'Albino bias', 'Animal Picture', 'Anthology', 'Anti-war',
        'Beach Film', 'Bengali Cinema', 'Biopic [feature]', 'Black-and-white', 'Bloopers & Candid Camera',
        'British New Wave', 'Bruceploitation', 'Business', 'Chinese Movies', 'Computers', 'Disaster', 'Dogme 95',
        'Early Black Cinema', 'Ensemble Film', 'Indie', 'Japanese Movies', 'Jungle Film', 'Kafkaesque', 'Latino',
        'Malayalam Cinema', 'Movie serial', 'New Hollywood', 'Roadshow/Carny', 'Tokusatsu', 'Tollywood',
        'Women in prison films', 'Wuxia', 'Natural disaster', 'Christian film', 'Religious Film', 'Hagiography', 'Propaganda film', 
        'Spirituality', 'Biblical Film'
    ]
}

COUNTRY_MAPPING = {
    'Asia': [
        'Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'China', 'Cyprus', 'Georgia', 
        'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Lebanon', 'Malaysia', 
        'Maldives', 'Mongolia', 'Myanmar (Burma)', 'Nepal', 'North Korea', 'Oman', 'Pakistan', 'Palestinian Territories', 'Philippines', 
        'Qatar', 'Republic of China (Taiwan)', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Tajikistan', 'Thailand', 
        'Timor-Leste', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen',
        "Burma", "Georgian SSR", "Hong Kong", "Iraqi Kurdistan", "Korea", "Macau", "Mandatory Palestine",
        "Palestinian territories", "Republic of China", "Taiwan", "Uzbek SSR", "Soviet Union", "Malayalam Language"
    ],
    'Africa': [
        'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 
        'Comoros', 'Congo', 'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 
        'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast (Côte d\'Ivoire)', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 
        'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'São Tomé and Príncipe', 
        'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
    ],
    'Europe': [
        'Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 
        'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kazakhstan', 
        'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 
        'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 
        'Ukraine', 'United Kingdom', "Crime", "Czechoslovakia", "England", "Federal Republic of Yugoslavia", "German Democratic Republic",
        "Isle of Man", "Kingdom of Great Britain", "Kingdom of Italy", "Northern Ireland",
        "Republic of Macedonia", "Scotland", "Serbia and Montenegro", "Slovak Republic",
        "Socialist Federal Republic of Yugoslavia", "Soviet occupation zone", "Ukrainian SSR", "Ukranian SSR",
        "Wales", "Weimar Republic", "West Germany", "Yugoslavia", "German Language", "Nazi Germany"
    ],
    'North America': [
        'Canada', 'Costa Rica', 'Cuba', 'Dominican Republic', 'El Salvador', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 
        'Nicaragua', 'Panama', 'United States of America', "Aruba", "Bahamas", "Puerto Rico"
    ],
    'South America': [
        'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela'
    ],
    'Oceania': [
        'Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 
        'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu'
    ]
}


"""
Helper function to recategorize genres
"""
def recategorize_genre(genre_mapping, genre_list):
    new_genre_set = set()
    for genre in genre_list:
        for category, genres in genre_mapping.items():
            if genre in genres:
                new_genre_set.add(category)
    return list(new_genre_set)

"""
Helper function to recategorize countries
"""
def recategorize_countries(country_list, country_mapping):
    new_continent = set()  
    for genre in country_list:
        for category, genres in country_mapping.items():
            if genre in genres:
                new_continent.add(category) 
    return list(new_continent)



"""
Function to recategorize countries
"""
def load_and_transform_genres():
    
    #load
    df_movie = pd.read_csv(DATA_PATH)
    df_genre = df_movie[["Wikipedia_movie_ID", "Movie_genres"]].copy()
    
    #separate genres
    df_genre["Movie_genres"] = df_movie["Movie_genres"].str.split(", ").to_frame()
    df_genre.dropna(inplace=True)

    #apply recategorization
    df_genre['category'] = df_genre['Movie_genres'].apply(lambda genres: recategorize_genre(GENRE_MAPPING, genres))
    return df_genre

"""
Function to recategorize countries
"""
def load_and_transform_countries():

    #load
    df_movie = pd.read_csv(DATA_PATH)
    df_countries = df_movie[["Wikipedia_movie_ID", "Movie_countries"]].copy()
    
    #separate
    df_countries["Movie_countries"] = df_movie["Movie_countries"].str.split(", ").to_frame()
    df_countries.dropna(inplace=True)

    #apply recategorization
    df_countries['continent'] = df_countries['Movie_countries'].apply(lambda c: recategorize_countries(c, COUNTRY_MAPPING)[0])
    return df_countries


"""
Function to apply all data transformations
"""
def all_data_transformations():

    #Add genre
    clean_df = pd.read_csv(DATA_PATH)
    df_genre = load_and_transform_genres()
    category_joined = clean_df.merge(df_genre[['Wikipedia_movie_ID', 'category']], on='Wikipedia_movie_ID', how='outer')

    #Add country
    df_countries = load_and_transform_countries()
    continent_joined = category_joined.merge(df_countries[['Wikipedia_movie_ID', 'continent']], on='Wikipedia_movie_ID', how='outer')

    return continent_joined