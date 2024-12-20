# Behind the Curtains: The Emotional Arcs of Cinema 

_LINK TO OUR DATASTORY:_ https://xavmarch13.github.io/dataffoneurs-story/

## Abstract

This project delves into the intricate relationship between emotions in films and their reception, using a multi-faceted analysis that combines sentiment classification, clustering, and exploratory data techniques. By leveraging movie summaries, we extract emotional arcs—capturing sentiments such as joy, sadness, and fear—and analyze their patterns across genres, time periods, and continents. Through techniques like PCA and clustering, we uncover distinct emotional dynamics and investigate how these relate to key performance indicators such as ratings.  

The motivation behind this work lies in understanding the emotional strategies that resonate with audiences and drive a film’s impact. We explore how societal changes, cinematic innovations, and cultural contexts shape emotional storytelling. This project aims to uncover the emotional fingerprints of successful films and offers insights into how films captivate, comfort, or challenge their audiences, blending art and analysis into a compelling narrative of cinema's emotional evolution.

## Research questions we aim to answer 

- *Emotional Patterns Across Time and Space:*  
   - How do emotional arcs in films evolve across time and continents?  
   - Are these patterns influenced by societal changes, technological innovations, or historical events?  

- *Genre-Specific Dynamics:*  
   - Do specific genres exhibit distinct emotional trajectories?  
   - What are the dominant emotions in each genre, and how do they shift over time and across cultures?  

- *Impact on Film Success:*  
   - How do emotional arcs influence ratings ?  
   - Do films with more varied emotional arcs perform better?  
   - Does the emotional tone at a film’s conclusion (e.g., joyful or melancholic) affect its reception?   
   - How does emotional complexity (the interplay of multiple emotions) contribute to a film's success? 


## Additional datasets
To address gaps in our dataset, we incorporated supplementary information from Wikipedia to enhance the CMU Movie Dataset's completeness. Approximately 40% of film summaries and all of the rating values were missing—critical data for our analysis of emotional arcs, ratings, and success metrics. To fill these gaps, we employed libraries like wikipedia-api and pywikibot and used requests and BeautifulSoup for web scraping in more complex cases.

We extracted summaries from the Plot/Synopsis/Summary sections. Special care was taken to handle variations in titles, adding "(film)" or resolving page redirections to ensure accurate matches. Summaries with fewer than 200 words were replaced with their Wikipedia counterparts.

Through this process, we recovered 36% of missing summaries and 93% of missing rating values, significantly enriching the dataset. By focusing on validity constraints, such as removing outliers and incomplete entries, we ensured the dataset's usability while enhancing its comprehensiveness.

## Methods

To estimate the sentiment of a movie, we apply sentiment classification models on the text of its summary. First, we clean each summary by removing any irrelevant or problematic content (e.g., html tags, weird citations). After cleaning, we calculate the word count of each summary to ensure it meets a minimum length requirement. For summaries that are too short, we retrieve additional information by scraping extended descriptions from the internet.

Once the summary is ready, we segment it—either by splitting into phrases or by splitting into segments (can be more than 1 sentence) based on semantic similarity. Each segment is then passed through a sentiment analysis model, specifically distiled version of RoBERTa. This model classifies each segment into one of seven emotions: anger, disgust, fear, joy, neutral, sadness, or surprise, providing a proportional score for each emotion.

To be able to compare emotion dynamics across movies, we use interpolation (or extrapolation when needed), to be able to have a common timeline for all movies. This allows us to approximate the emotions through the length of a movie with the emotions brought out by the summary.

In the final step, we fill in any missing data, such as summary and ratings.

Once the dataset was complete, we applied Principal Component Analysis (PCA) to uncover underlying patterns in the emotional profiles of movies. By reducing the seven-dimensional emotion vectors—anger, disgust, fear, joy, neutral, sadness, and surprise—into two or three principal components, PCA highlighted the most significant variations in the data while preserving interpretability. This approach allowed us to identify correlations between emotions, visualize differences across genres, and track temporal shifts in emotional dynamics.

Building on these insights, we implemented K-means clustering to group movies based on their emotional signatures. The number of clusters was determined using metrics like the Elbow Method and Silhouette Scores, ensuring an optimal balance between granularity and clarity. Each cluster revealed distinct emotional patterns, linking specific emotional profiles to genres, historical contexts, and measures of success such as ratings. Together, PCA and clustering provided a deeper understanding of how emotional strategies shape audience engagement and storytelling.

Details of the methods are presented in the results notebook.

## Contributions of all group member
- Mathieu : sentiment analysis, PCA, clustering, exploration of the data, wrote the intro part of the data story.
- Xavier : sentiment analysis and general emotion analysis, k-shape clustering, website animation and links, exploration of the data, wrote the sentiment analysis part of the data story and k-shape.
- Ines : scraping, analysis between emotions and ratings and wrote that part of the data story.
- Florian : data cleaning, analyzed the emotional arc from different perspectives, explored the emotional complexity in runtime,wrote the uncovering patterns in film sentiment part of the data story.
- Alix: data cleaning, historical analysis of the emotional arc, graphs of the sentimental analysis across continents, wrote the historical analysis part of the data story.
