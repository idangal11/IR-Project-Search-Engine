from best_inverted_index_ever import InvertedIndex
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
import math
from google.cloud import storage
import os
import pickle

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", "may", "first", "see", "history",
                    "people", "one", "two", "part", "thumb", "including", "second", "following", "many", "however",
                    "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stemmer = PorterStemmer()


def anchor_title_composition(anchor_weight, title_weight, sorted_cosine_similarity_anchor, sorted_cosine_similarity_title):
    merged_dict = {}
    i = 0
    while i < 100:
        if i < len(sorted_cosine_similarity_anchor):
            # body part
            anchor_docid, anchor_score = sorted_cosine_similarity_anchor[i]
            if merged_dict.get(anchor_docid, 0) == 0:
                merged_dict[anchor_docid] = anchor_score
            else:
                merged_dict[anchor_docid] = merged_dict.get(anchor_docid) * title_weight + anchor_score * anchor_weight
        if i < len(sorted_cosine_similarity_title):
            # body part
            title_docid, title_score = sorted_cosine_similarity_title[i]
            if merged_dict.get(title_docid, 0) == 0:
                merged_dict[title_docid] = title_score
            else:
                merged_dict[title_docid] = merged_dict.get(title_docid) * anchor_weight + title_score * title_weight
        i += 1

    return sorted([(doc_id, score) for doc_id, score in merged_dict.items()], key=lambda x: x[1],
                      reverse=True)

def body_title_composition(body_weight, title_weight, sorted_cosine_similarity_body, sorted_cosine_similarity_title):
    merged_dict = {}
    i = 0
    while i < 100:
        if i < len(sorted_cosine_similarity_body):
            # body part
            body_docid, body_score = sorted_cosine_similarity_body[i]
            if merged_dict.get(body_docid, 0) == 0:
                merged_dict[body_docid] = body_score
            else:
                merged_dict[body_docid] = merged_dict.get(body_docid) * title_weight + body_score * body_weight
        if i < len(sorted_cosine_similarity_title):
            # body part
            title_docid, title_score = sorted_cosine_similarity_title[i]
            if merged_dict.get(title_docid, 0) == 0:
                merged_dict[title_docid] = title_score
            else:
                merged_dict[title_docid] = merged_dict.get(title_docid) * body_weight + title_score * title_weight
        i += 1

    return sorted([(doc_id, score) for doc_id, score in merged_dict.items()], key=lambda x: x[1],
                      reverse=True)


def handle_query_body(query, use_stemming):
    # tokenazing procces to remove unnecessary symbols
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    # remove stop words from query and create clean one
    query_ready = []
    for token in tokens:
        if token not in all_stopwords:
            query_ready.append(token)
    if use_stemming:
        query_ready = [stemmer.stem(word) for word in query_ready]  # stemming
    # create dictinary: key=term, value=number of performance (frequency) in query
    query_frequency = {}
    for term in query_ready:
        query_frequency[term] = query_frequency.get(term, 0) + 1
    # create query normal value
    sum_of_query_terms_powered = 0
    for term in query_frequency.keys():
        sum_of_query_terms_powered += query_frequency[term] * query_frequency[term]
    normalized_query = 1 / math.sqrt(sum_of_query_terms_powered)
    return query_ready, normalized_query


def handle_query_title(query, use_stemming):
    # tokenazing procces to remove unnecessary symbols
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    # remove stop words from query and create clean one
    # DISTINCT QUERY WORDS so append only if not exist in query
    query = []
    for token in tokens:
        if token not in all_stopwords and token not in query:
            query.append(token)
    if use_stemming:
        query = [stemmer.stem(word) for word in query]  # stemming
    return query

def get_cossim_binary_anchor_dict(query, app):
    # initialize cosine_similarity dictinary: key=docID, value=cosine score
    cosine_similarity = {}
    # ranking using Cosine Similarity. based on lecture3 algorithm
    # regardless of the number of times the term appeared in the title! (+1)
    for term in query:
        pls = app.inverted_anchor.read_posting_list(term, app.bucket_name, "anchor")
        for tup in pls:
            cosine_similarity[tup[0]] = cosine_similarity.get(tup[0], 0) + 1
    return sorted([(doc_id, score) for doc_id, score in cosine_similarity.items()], key=lambda x: x[1],
                      reverse=True)

def get_cossim_binary_title_dict(query, app, use_stemming):
    # initialize cosine_similarity dictinary: key=docID, value=cosine score
    cosine_similarity = {}
    # ranking using Cosine Similarity. based on lecture3 algorithm
    # regardless of the number of times the term appeared in the title! (+1)
    if use_stemming:
        for term in query:
            pls = app.inverted_title_stem.read_posting_list(term, app.bucket_name, "title_stem")
            for tup in pls:
                cosine_similarity[tup[0]] = cosine_similarity.get(tup[0], 0) + 1
    else:
        for term in query:
            pls = app.inverted_title.read_posting_list(term, app.bucket_name, "title")
            for tup in pls:
                cosine_similarity[tup[0]] = cosine_similarity.get(tup[0], 0) + 1

    return sorted([(doc_id, score) for doc_id, score in cosine_similarity.items()], key=lambda x: x[1],
                      reverse=True)

def get_cossim_tfidf_body_dict(query, normalized_query, app, use_stemming):
    # initialize cosine_similarity dictinary: key=docID, value=cosine score
    cosine_similarity = {}
    # ranking using Cosine Similarity. based on lecture3 algorithm
    if use_stemming:
        N = len(app.inverted_body_stem.DL)
        for term in query:
            pls = app.inverted_body_stem.read_posting_list(term, app.bucket_name, "body_stem")
            for tup in pls:
                tf = tup[1] / app.inverted_body_stem.DL[tup[0]]
                idf = math.log(N / app.inverted_body_stem.df[term], 2)
                tfidf = tf * idf
                cosine_similarity[tup[0]] = cosine_similarity.get(tup[0], 0) + tfidf
        # normalized tfidf scores same as cosine similarity formula
        for doc_id in cosine_similarity.keys():
            score = cosine_similarity[doc_id] * app.inverted_body_stem.NF[doc_id] * normalized_query
            cosine_similarity[doc_id] = score
    else:
        cosine_similarity = {}
        N = len(app.inverted_body.DL)
        # ranking using Cosine Similarity. based on lecture3 algorithm
        for term in query:
            pls = app.inverted_body.read_posting_list(term, app.bucket_name, "body")
            for tup in pls:
                tf = tup[1] / app.inverted_body.DL[tup[0]]
                idf = math.log(N / app.inverted_body.df[term], 2)
                tfidf = tf * idf
                cosine_similarity[tup[0]] = cosine_similarity.get(tup[0], 0) + tfidf
        # normalized tfidf scores same as cosine similarity formula
        for doc_id in cosine_similarity.keys():
            score = cosine_similarity[doc_id] * app.inverted_body.NF[doc_id] * normalized_query
            cosine_similarity[doc_id] = score

    return sorted([(doc_id, score) for doc_id, score in cosine_similarity.items()], key=lambda x: x[1],
                  reverse=True)
