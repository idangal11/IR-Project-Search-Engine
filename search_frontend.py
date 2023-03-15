from flask import Flask, request, jsonify
from best_inverted_index_ever import InvertedIndex

from nltk.stem.porter import *
import math
from google.cloud import storage
import os
import pickle
from helper import *


class MyFlaskApp(Flask):
    os.environ["GCLOUD_PROJECT"] = "irproject-374020"
    bucket_name = 'irproject-315786798-205814999-bucket'
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    inverted_body = None
    inverted_title = None
    inverted_body_stem = None
    inverted_title_stem = None
    inverted_anchor = None
    page_rank = None  # PandasFrame
    page_view = None  # a Counter

    def run(self, host=None, port=None, debug=None, **options):
        # get body index from bucket
        src = "index_body.pkl"
        blob = self.bucket.blob(f"indexes/{src}")
        self.inverted_body = pickle.loads(blob.download_as_string())

        # get title index from bucket
        src = "index_title.pkl"
        blob = self.bucket.blob(f"indexes/{src}")
        self.inverted_title = pickle.loads(blob.download_as_string())

        # get body index with stemming from bucket
        src = "index_body_stem.pkl"
        blob = self.bucket.blob(f"indexes/{src}")
        self.inverted_body_stem = pickle.loads(blob.download_as_string())

        # get title index with stemming from bucket
        src = "index_title_stem.pkl"
        blob = self.bucket.blob(f"indexes/{src}")
        self.inverted_title_stem = pickle.loads(blob.download_as_string())

        # get anchor index from bucket
        src = "index_anchor.pkl"
        blob = self.bucket.blob(f"indexes/{src}")
        self.inverted_anchor = pickle.loads(blob.download_as_string())

        # get page rank from bucket
        src = "pagerank.pkl"
        blob = self.bucket.blob(f"pagerank_and_pageview/{src}")
        self.page_rank = pickle.loads(blob.download_as_string())

        # get page view from bucket
        src = "pageview.pkl"
        blob = self.bucket.blob(f"pagerank_and_pageview/{src}")
        self.page_view = pickle.loads(blob.download_as_string())

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # # EXAMPLE FOR TITLE-BODY COMPOISITION

    # query_title = handle_query_title(query, use_stemming=True)
    # # test to determine composition by query length
    # if len(query_title) > 1:
    #     body_weight=0.2
    #     title_weight=0.8
    # else:
    #     body_weight=0
    #     title_weight=1
    # sorted_cosine_similarity_title = get_cossim_binary_title_dict(query_title, app, use_stemming=True)
    # query_body, normalized_query_body = handle_query_body(query, use_stemming=True)
    # sorted_cosine_similarity_body = get_cossim_tfidf_body_dict(query_body, normalized_query_body, app, use_stemming=True) 
    
    # merged_res = body_title_composition(body_weight, title_weight, sorted_cosine_similarity_body, sorted_cosine_similarity_title)
    # for tup in merged_res:
    #     res.append((tup[0], app.inverted_title_stem.doc_to_title[tup[0]]))


    # # BEGIN SOLUTION
    query = handle_query_title(query, use_stemming=True)
    sorted_cosine_similarity = get_cossim_binary_title_dict(query, app, use_stemming=True)
    # create result with tuple (wiki_id, title) structure
    for tup in sorted_cosine_similarity:
        res.append((tup[0], app.inverted_title_stem.doc_to_title[tup[0]]))
    # END SOLUTION
    return jsonify(res[:100])



@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query, normalized_query = handle_query_body(query, use_stemming=False)
    sort_sim = get_cossim_tfidf_body_dict(query, normalized_query, app, use_stemming=False)
    for tup in sort_sim:
        res.append((tup[0], app.inverted_title.doc_to_title[tup[0]]))
    # END SOLUTION
    return jsonify(res[:100])


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = handle_query_title(query, use_stemming=False)
    sorted_cosine_similarity = get_cossim_binary_title_dict(query, app, use_stemming=False)
    # create result with tuple (wiki_id, title) structure
    for tup in sorted_cosine_similarity:
        res.append((tup[0], app.inverted_title.doc_to_title[tup[0]]))

    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    #anchor and title query handling is the same
    query = handle_query_title(query, use_stemming=False)
    sorted_cosine_similarity = get_cossim_binary_anchor_dict(query, app)
    # create result with tuple (wiki_id, title) structure
    for tup in sorted_cosine_similarity:
        res.append((tup[0], app.inverted_title.doc_to_title.get(tup[0],0)))
    # END SOLUTION
    return jsonify(res)



@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = []
    for docid in wiki_ids:
        row = app.page_rank.loc[app.page_rank['id'] == docid]
        pagerank = row['pagerank'].iloc[0]
        res.append(float(pagerank))
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = []
    for docid in wiki_ids:
        res.append(int(app.page_view.get(int(docid), 0)))
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
