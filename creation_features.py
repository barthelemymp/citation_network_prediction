''' This file computes the features for the training and testing set and save them in numpy arrays .npy in order not to compute them before each train'''

###############
# IMPORTATION #
###############

import random
import math
import csv
import re
import nltk
import igraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, log_loss

#############################################
# UTILS : functions that will be used later #
#############################################

nltk.download('punkt')  # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()
stemmerRegXP = nltk.stem.RegexpStemmer(r'\([^)]*\)')  # removes text inside parenthesis & parenthesis

def jaccard_coefficent(source, target, g):
    source_neighbors = set(g.neighbors(source))
    target_neighbors = set(g.neighbors(target))
    intersection = len(source_neighbors.intersection(target_neighbors))
    union = len(source_neighbors.union(target_neighbors))
    if union == 0:
        return 0.0
    else:
        return float(intersection / float(union))

def adamic_adar(source, target, g):
    source_neighbors = set(g.neighbors(source))
    target_neighbors = set(g.neighbors(target))
    coef = 0.0
    for i in source_neighbors.intersection(target_neighbors):
        if math.log(len(g.neighbors(i))) == 0:
            coef += 0.0
        else:
            coef += float(1 / math.log(len(g.neighbors(i))))
    return coef

def earliness_coef(target,g):
    score=0.
    index_target = IDs.index(target)
    for s in gAdjInList[index_target]:
        s_info = node_info[s]
        score+=1/(1+abs(float((target_info[1]))-float(s_info[1])))
    return score

#################################################
# LOADING TRAINING SET AND TESTING SET AS LISTS #
#################################################

''' 
The columns of the data frame below are:
(1) paper unique ID (integer)
(2) publication year (integer)
(3) paper title (string)
(4) authors (strings separated by ,)
(5) name of journal (optional) (string)
(6) abstract (string) - lowercased, free of punctuation except intra-word dashes 
'''

with open("data/training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set = list(reader)  # Make a list from the training_set.txt

training_set = [element[0].split(" ") for element in training_set]

with open("data/testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

with open("data/node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info = list(reader)  # Make a list from the node_information.csv

IDs = [element[0] for element in node_info]  # Paper IDs
corpusTitle = [element[2] for element in node_info] # Paper titles
corpusAuthor = [element[3] for element in node_info] # Paper authors
corpusJournal = [element[4] for element in node_info] # Name of the journal
corpus = [element[5] for element in node_info]  # Paper abstracts

###############################
# PREPROCESS TEXTUAL FEATURES #
###############################

''' Preprocess textual features in order to  vectorize them in a TFIDF vector'''

# vectorizer initializes the TfidfVectorizer()
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words="english")

# Each textual features is transfromed in a TFIDF vector

features_TFIDF_Abstract = vectorizer.fit_transform(corpus)
features_TFIDF_Title = vectorizer.fit_transform(corpusTitle)
features_TFIDF_Author = vectorizer.fit_transform(corpusAuthor)
features_TFIDF_Journal = vectorizer.fit_transform(corpusJournal)

######################
# Graph construction #
######################

''' Construct the graph architecture and prepare useful information for later features'''

edges = [(element[0], element[1]) for element in training_set if element[2] == "1"]
nodes = IDs

# create empty directed graph
g = igraph.Graph(directed=True)

# Construction of the graph
g.add_vertices(nodes)
g.add_edges(edges)

# We keep in a list the adjency elements of each nodes - undirected and directed
gAdjList = [set(x) for x in g.get_adjlist(mode="ALL")]
gAdjInList= [set(x) for x in g.get_adjlist(mode="IN")]

# We keep the degree of each node
degrees = g.degree()

# We calculate the page rank indice for each node
page_rank = []
page_rank = g.pagerank()

#################################################
# CREATION OF THE FEATURES FOR THE TRAINING SET #
#################################################

'''
The list of features is :
- Baseline features -
(0) Number of overlapping words in the title 
(1) Years between the publication of the source paper and the target paper
(2) Number of common authors
- Our features -
(3) number of common words in the journal title
(4) Number of common words in abstracts
(5) Cosine similarity of the two abstracts
(6) Cosine similarity of the authors
(7) Cosine similarity of the titles
(8) Cosine similarity of the journal titles
(9) Number of common neighbours in the graph
(10) Preferential attachment
(11) Jaccard similarity coefficient
(12) Ademic Adar similarity coefficient
(13) Earliness coefficient (see the report for more explanations)
(14) Page rank indice of the source article
(15) Page rank indice of the target article
'''

#Features initialization 

overlap_title = []
temp_diff = []
comm_auth = []
comm_journ = []
comm_abstr = []
cos_sim_abstract = []
cos_sim_author = []
cos_sim_title = []
cos_sim_journal = []
com_neigh = []
pref_attach = []
jac_sim = []
adam_adar = []
earliness = []
page_rank_list_target = []
page_rank_list_source = []

# Features creation for the training set

print("Start computing the features for the training set")

counter = 0

# For each row of the training_set calculate the features
for i in range(len(training_set)):

    # 1: Source ID and target ID
    source = training_set[i][0]
    target = training_set[i][1]

    # 2: Corresponding index in the list of nodes
    index_source = IDs.index(source)
    index_target = IDs.index(target)

    # 3: Extract the information for the source and target
    source_info = [element for element in node_info if element[0] == source][0]
    target_info = [element for element in node_info if element[0] == target][0]

    # 4/a: Textual preprocessing
    source_title = source_info[2].lower().split(" ")  # convert to lowercase and tokenize
    source_title = [token for token in source_title if token not in stpwds]  # remove stopwords
    source_title = [stemmerRegXP.stem(token) for token in source_title]  # perform stemming for parenthesis
    source_title = [stemmer.stem(token) for token in source_title]  # perform stemming

    target_title = target_info[2].lower().split(" ")  # convert to lowercase and tokenize
    target_title = [token for token in target_title if token not in stpwds]  # remove stopwords
    target_title = [stemmerRegXP.stem(token) for token in target_title]  # perform stemming for parenthesis
    target_title = [stemmer.stem(token) for token in target_title]  # perform stemming


    source_abstr = source_info[5].lower().split(" ")  # convert to lowercase and tokenize
    source_abstr = [token for token in source_abstr if token not in stpwds]  # remove stopwords
    source_abstr = [stemmer.stem(token) for token in source_abstr]  # perform stemming

    target_abstr = target_info[5].lower().split(" ")  # convert to lowercase and tokenize
    target_abstr = [token for token in target_abstr if token not in stpwds]  # remove stopwords
    target_abstr = [stemmer.stem(token) for token in target_abstr]  # perform stemming


    source_auth = source_info[3]
    source_auth = re.sub(r'\([^)]*\)', '', source_auth)  # remove parenthesis and content inside them
    source_auth = source_auth.split(",")
    source_auth = [stemmerRegXP.stem(token) for token in source_auth]  # perform stemming for parenthesis
    source_auth[:] = [val for val in source_auth if not val == " " or val == ""]  # remove empty entries in our list

    # iterate through our author list end call strip() to remove starting and trailing spaces
    for sa, val in enumerate(source_auth):
        source_auth[sa] = source_auth[sa].strip()

    target_auth = target_info[3]
    target_auth = re.sub(r'\([^)]*\)', '', target_auth)
    target_auth = target_auth.split(",")
    target_auth = [stemmerRegXP.stem(token) for token in target_auth]
    target_auth[:] = [val for val in target_auth if not val == " " or val == ""]
    for ta, val in enumerate(target_auth):
        target_auth[ta] = target_auth[ta].strip()

    source_journal = source_info[4]
    target_journal = target_info[4]

    # Compute the features

    #0
    overlap_title.append(len(set(source_title).intersection(set(target_title))))
    #1
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    #2
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
    #3
    comm_journ.append(len(set(source_journal).intersection(set(target_journal))))
    #4
    comm_abstr.append(len(set(source_abstr).intersection(set(target_abstr))))
    #5
    cos_sim_abstract.append(cosine_similarity(features_TFIDF_Abstract[index_source], features_TFIDF_Abstract[index_target]))
    #6
    cos_sim_title.append(cosine_similarity(features_TFIDF_Title[index_source], features_TFIDF_Title[index_target]))
    #7
    cos_sim_author.append(cosine_similarity(features_TFIDF_Author[index_source], features_TFIDF_Author[index_target]))
    #8
    cos_sim_journal.append(cosine_similarity(features_TFIDF_Journal[index_source], features_TFIDF_Journal[index_target]))
    #9
    com_neigh.append(len(gAdjList[index_source].intersection(gAdjList[index_target])))
    #10
    pref_attach.append(int(degrees[index_source] * degrees[index_target]))
    #11
    jac_sim.append(jaccard_coefficent(index_source, index_target, g))
    #12
    adam_adar.append(adamic_adar(index_source, index_target, g))
    #13
    earliness.append(earliness_coef(target,g))
    #14
    page_rank_list_target.append(page_rank[index_target])
    page_rank_list_source.append(page_rank[index_source])


    counter += 1
    if counter % 10000==0:
        print("step : "+str(counter))

# Convert the list as a numpy array
training_features = np.array(
    [overlap_title, temp_diff, comm_auth, comm_journ, comm_abstr, cos_sim_abstract, cos_sim_author,
     cos_sim_journal, cos_sim_title, com_neigh, pref_attach, jac_sim, adam_adar,earliness, page_rank_list_source,
     page_rank_list_target]).astype(np.float64).T

# And save it
np.save("data_npy/training_features",training_features)

#Rescale the array to better train
training_features_scaled = preprocessing.scale(training_features, copy=False)

# And also save the scaled array
np.save("data_npy/training_features_scaled",training_features_scaled)

print("End computing the features for the training set")

#################################################
# CREATION OF THE LABELS FOR THE TRAINING SET #
#################################################

labels = [int(element[2]) for element in training_set]
# Convert the labels in a numpy array
training_labels = np.array(labels)
# And save it
np.save("data_npy/training_labels",training_labels)


#################################################
# CREATION OF THE FEATURES FOR THE TESTING SET #
#################################################

'''We compute the same features for the testing set'''

#Features initialization 

overlap_title_test = []
temp_diff_test = []
comm_auth_test = []
comm_journ_test = []
comm_abstr_test = []
cos_sim_abstract_test = []
cos_sim_author_test = []
cos_sim_journal_test = []
cos_sim_title_test = []
com_neigh_test = []
pref_attach_test = []
jac_sim_test = []
adam_adar_test = []
earliness_test = []
page_rank_list_source_test = []
page_rank_list_target_test = []


print("Start computing the features for the testing set")

counter = 0

for i in range(len(testing_set)):

    # 1: Source ID and target ID
    source = testing_set[i][0]
    target = testing_set[i][1]

    # 2: Corresponding index in the list of nodes
    index_source = IDs.index(source)
    index_target = IDs.index(target)

    # 3: Extract the information for the source and target
    source_info = [element for element in node_info if element[0] == source][0]
    target_info = [element for element in node_info if element[0] == target][0]

    # 4: Manipulate source & target title
    # convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
    # remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    # perform stemming
    source_title = [stemmer.stem(token) for token in source_title]
    # 4/a: Textual preprocessing
    source_title = source_info[2].lower().split(" ")  # convert to lowercase and tokenize
    source_title = [token for token in source_title if token not in stpwds]  # remove stopwords
    source_title = [stemmerRegXP.stem(token) for token in source_title]  # perform stemming
    source_title = [stemmer.stem(token) for token in source_title]  # perform stemming

    target_title = target_info[2].lower().split(" ")  # convert to lowercase and tokenize
    target_title = [token for token in target_title if token not in stpwds]  # remove stopwords
    target_title = [stemmerRegXP.stem(token) for token in target_title]  # perform stemming
    target_title = [stemmer.stem(token) for token in target_title]  # perform stemming

    source_abstr = source_info[5].lower().split(" ")  # convert to lowercase and tokenize
    source_abstr = [token for token in source_abstr if token not in stpwds]  # remove stopwords
    source_abstr = [stemmer.stem(token) for token in source_abstr]  # perform stemming

    target_abstr = target_info[5].lower().split(" ")  # convert to lowercase and tokenize
    target_abstr = [token for token in target_abstr if token not in stpwds]  # remove stopwords
    target_abstr = [stemmer.stem(token) for token in target_abstr]  # perform stemming

    source_auth = target_info[3]
    source_auth = re.sub(r'\([^)]*\)', '', source_auth)
    source_auth = source_auth.split(",")
    source_auth = [stemmerRegXP.stem(token) for token in source_auth]
    source_auth[:] = [val for val in source_auth if not val == " " or val == ""]
    for ta, val in enumerate(source_auth):
        source_auth[ta] = source_auth[ta].strip()

    target_auth = target_info[3]
    target_auth = re.sub(r'\([^)]*\)', '', target_auth)
    target_auth = target_auth.split(",")
    target_auth = [stemmerRegXP.stem(token) for token in target_auth]
    target_auth[:] = [val for val in target_auth if not val == " " or val == ""]
    for ta, val in enumerate(target_auth):
        target_auth[ta] = target_auth[ta].strip()

    source_journal = source_info[4]
    target_journal = target_info[4]


    # Compute the features

    #0
    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
    #1
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
    #2
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))
    #3
    comm_journ_test.append(len(set(source_journal).intersection(set(target_journal))))
    #4
    comm_abstr_test.append(len(set(source_abstr).intersection(set(target_abstr))))
    #5
    cos_sim_abstract_test.append(cosine_similarity(features_TFIDF_Abstract[index_source], features_TFIDF_Abstract[index_target]))
    #6
    cos_sim_title_test.append(cosine_similarity(features_TFIDF_Title[index_source], features_TFIDF_Title[index_target]))
    #7
    cos_sim_author_test.append(cosine_similarity(features_TFIDF_Author[index_source], features_TFIDF_Author[index_target]))
    #8
    cos_sim_journal_test.append(cosine_similarity(features_TFIDF_Journal[index_source], features_TFIDF_Journal[index_target]))
    #9
    com_neigh_test.append(len(gAdjList[index_source].intersection(gAdjList[index_target])))
    #10
    pref_attach_test.append(int(degrees[index_source] * degrees[index_target]))
    #11
    jac_sim_test.append(jaccard_coefficent(index_source, index_target, g))
    #12
    adam_adar_test.append(adamic_adar(index_source, index_target, g))
    #13
    earliness_test.append(earliness_coef(target,g))
    #14
    page_rank_list_target_test.append(page_rank[index_target])
    #15
    page_rank_list_source_test.append(page_rank[index_source])

    counter += 1
    if counter % 10000==0:
        print("step: "+ str(counter))

# Convert the list into an array
testing_features = np.array(
    [overlap_title_test, temp_diff_test, comm_auth_test, comm_journ_test, comm_abstr_test, cos_sim_abstract_test,
     cos_sim_author_test, cos_sim_journal_test, cos_sim_title_test, com_neigh_test, pref_attach_test,
     jac_sim_test, adam_adar_test, earliness_test, page_rank_list_source_test, page_rank_list_target_test]).astype(
    np.float64).T

# Scaled features
testing_features_scaled = preprocessing.scale(testing_features, copy=False)

# And both the two array
np.save('data_npy/testing_features',testing_features)
np.save('data_npy/testing_features_scaled',testing_features_scaled)

print("End computing the features for the testing set")
