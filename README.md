# Information Retrieval Project - Wikipedia Search Engine


## Description

This project has been done as part of the Information Retrieval course at BGU. <br/>
Using a preproccessed wikipedia dump as the source of data, we built indexes<br/> using a cluster on google cloud platform(GCP). 
The indexes were stored in a bucket, page rank and page view were also implemented.

## Tech/Framework used

* Google Cloud Platform
* Flask
* PySpark

## Project Modules
### search_frontend.py
* search - tokanization, stopword removal and stemming of query. using binary ranking of document title. in the comments is an example for a body-title composition ranking.
* search_body - tokanization, stopword removal of query. cosine similarity using tf-idf on the body of articles.
* search_title - tokenization, stopword removal of query. binary ranking of document titles.
* search_anchor - tokenization, stopword removal of query. binary ranking of the anchor texts of articles.
* get_pagerank - the function returns the pagerank scores of a given list of article id's.
* get_pageview - the function returns the pageview numbers of a given list of article id's.
### helper.py
#### containing the stopword declarations, and functions that search_frontend uses <br/>
#### For Example, handle_query_body, body_title_compositions, and different experiements we ran like, title_anchor_composition.
### best_inverted_index_ever.py
#### The invertex index class we used for indexing, and also use in helper.py.
### run_gcp.ipynb
#### the main code we used to built the indexes on the cluster.
### draw_graphs_map_40.ipynb
#### a simple code for calculation of map@40 and graph creation.
