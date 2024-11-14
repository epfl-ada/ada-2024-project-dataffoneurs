# From Laughs to Tears: How Emotional Journeys Win Audiences 

## Abstract

This project investigates the relationship between emotional arcs in films and their success, focusing on how emotions evolve within genres, across continents, and over time. The goal is to identify whether universal or genre-specific emotional patterns exist and how these patterns influence the reception of films, particularly in terms of box office performance, ratings, and awards. The project explores whether films with more varied emotional trajectories perform better and whether certain emotional dynamics optimize success. Additionally, it examines the role of historical context in shaping emotional arcs across different genres. By analyzing these emotional patterns, the research aims to uncover key insights into how emotions impact audience engagement and contribute to a film’s success. The motivation behind this work is to offer a deeper understanding of the emotional strategies that make films resonate with viewers globally, providing a better understanding of the factors that drive their enjoyment and connection to the story.

## Research questions we aim to answer

 -  Does a universal or genre-specific emotional arc exist across movies, and how does it change over time and across different eras?

 -  What emotions or sentiments dominate in each genre?

 - How do emotional arcs in films within the same genre vary across continents? Especially do differences emerge when North America is not taken into account ?

 -  How does the emotional arc in a specific genre (e.g., action, drama) vary across continents, and can we link this variation to historical contexts or events?

 - To what extent does the average emotional tone of a film influence its success (box office, ratings, awards)? Is emotion a leading factor? Does a more varied emotional arc contribute to the film’s success?

 - How do emotions differ between successful and unsuccessful films? What impact does a film’s ending have on its success—should it end on a high note or a more melancholic one? Does a joyful conclusion lead to greater success?

## Additional datasets
To address missing data in our dataset, we incorporate additional information from Wikipedia pages for films to complete the CMU dataset. Approximately 40% of film summaries and over 90% of box office values are missing, and these metrics are essential for our analysis. To fill these gaps, we use libraries like wikipedia-api and pywikibot, and in cases requiring more detail, we utilize requests and BeautifulSoup for web scraping. Our approach involves extracting summaries from the Plot/Synopsis/Summary sections and box office revenue from the InfoBox, ensuring we access the correct page by handling title variations, such as adding "(film), and handling page redirections.

This process enriches our data, specifically targeting films with incomplete summaries (replacing those under 200 words with the Wikipedia entry) and adding missing box office values.

## Methods

To estimate the sentiment of a movie, we apply sentiment classification models on the text of its summary. First, we clean each summary by removing any irrelevant or problematic content (e.g., html tags, weird citations). After cleaning, we calculate the word count of each summary to ensure it meets a minimum length requirement. For summaries that are too short, we retrieve additional information by scraping extended descriptions from the internet.

Once the summary is ready, we segment it—either by splitting into phrases or by splitting into segments (can be more than 1 sentence) based on semantic similarity. Each segment is then passed through a sentiment analysis model, specifically distiled version of RoBERTa. This model classifies each segment into one of seven emotions: anger, disgust, fear, joy, neutral, sadness, or surprise, providing a proportional score for each emotion.

To be able to compare emotion dynamics across movies, we use interpolation (or extrapolation when needed), to be able to have a common timeline for all movies. This allows us to approximate the emotions through the length of a movie with the emotions brought out by the summary.

In the final step, we fill in any missing data, such as box office revenue, release year, or genre, by scraping additional sources as needed and remove every row with NaN value. This thorough process results in a comprehensive dataset that captures not only the sentiment profile of each movie summary but also key contextual and financial data.

Details of the methods are presented in the results notebook.

## Proposed Timeline

| Week          | Description | Person in charge |
| -----------   | ----------- | ----------- |
| Week 10 | Finish scraping if needed, run sentiment analysis on all films once done | |
| Week 10 | Draw plots, answer general questions about general dynamics for all movies, across genres, time, and continent |  |
| Week 11 | Dig deeper into differences across genres | |
| Week 11 | Dig deeper into differences continent time | |
| Week 11 | Dig deeper into differences across continents |  |
| Week 12 | Investigate the box office - emotional relationship in detail |  |
| Week 12 | Investigate the reviews, awards - emotional relationship in detail |
| Week 13 | Get together week, analysis of results together |  |
| Week 14 |  Data story writing  ||





