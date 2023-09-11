import math
import os
import pandas as pd
import string
from stemming.porter2 import stem
from coll import parse_rcv_coll
from numpy import dot
from numpy.linalg import norm


def query_process(topic_file):
    query = dict()
    start_end = False

    for line in open(topic_file):
        line = line.strip()

        if start_end == False:
            line_list = []
            long_text = ""
            if line.startswith("<num>"):
                id = line.split()[2][1:]
                start_end = True

        elif start_end == True:
            if line.startswith("<narr>"):
                start_end = False
            else:
                if line.startswith("<title>"):
                    line = line.translate(str.maketrans('', '', string.punctuation))
                    line_list.append(line.replace("<title>", "").strip())
                else:
                    line = line.translate(str.maketrans('', '', string.punctuation))
                    line_list.append(line)
            long_text = "".join(line_list).replace("<desc> Description:", " ")
            query[id] = long_text

    return query


def tfidf(doc, df, ndocs):
    weight_dict = dict()
    term_dict = doc.terms
    # calculate bottom part of equation
    bottom = 0
    for k, v in term_dict.items():
        fi = v  # term_frequency
        ni = df[k]  # document frequency
        bottom += ((math.log(fi, 10)+1)*math.log(ndocs/ni, 10))**2
    bottom = math.sqrt(bottom)
    # calculate top part of equation
    for term, freq in term_dict.items():
        weight = ((math.log(freq, 10)+1) * math.log(ndocs/df[term]))/bottom
        weight_dict[term] = weight
    return weight_dict


def parse_query(q):
    terms = []
    query_terms = q.split()
    for t in query_terms:
        term = stem(t.lower())
        terms.append(term)
    return terms


def create_hist(term_list):
    hist = dict()
    for t in term_list:
        if t in hist:
            hist[t] += 1
        else:
            hist[t] = 1
    return hist


def cosine_similarity(df):
    similarity_scores = dict()
    for i in range(df.shape[0]-1):
        doc_vec = df.iloc[i, :]
        q_vec = df.iloc[-1, :]
        cosine_sim = dot(doc_vec, q_vec) / (norm(doc_vec) * norm(q_vec))
        similarity_scores[df.index[i]] = cosine_sim
        similarity_scores_sort = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    return similarity_scores_sort


# Execution start
cwd = os.getcwd()

# directory for output
if not os.path.exists("{}/Result".format(cwd)):
    os.makedirs("{}/Result".format(cwd))

query_dict_terms = dict()
for qId, query in query_process("{}/Topics.txt".format(cwd)).items():
    query_dict_terms[qId] = create_hist(parse_query(query))

stopwords_f = open('{}/common-english-words.txt'.format(cwd), 'r')
stop_words = stopwords_f.read().split(',')
stopwords_f.close()

data_id = list(range(101, 151))
all_documents = dict()
for ids in data_id:
    all_documents[ids] = parse_rcv_coll("{}/DataCollection/Dataset{}".format(cwd, ids), stop_words)

# get materials for our dataframe
# set up all_terms containing all term frequency dictionary
# set up all index containing indexes (which is our document
all_terms = []
all_index = []

# create vectors for all documents in each dataset
for datasetid, dataset in all_documents.items():
    temp = []
    index_temp = []
    for docId, bowdoc in dataset.get_docs().items():
        temp.append(bowdoc.terms)
        index_temp.append(docId)
    temp.append(query_dict_terms[str(datasetid)])
    index_temp.append("Query")
    all_terms.append(temp)
    all_index.append(index_temp)

docs_similarity = []
for i, terms in enumerate(all_terms):
    df = pd.DataFrame(terms)
    df.index = all_index[i]
    df = df.fillna(0)
    sim = cosine_similarity(df)
    docs_similarity.append(sim)


fp = open("{}/Result/Cosine_Similarity_Values.txt".format(cwd), "w")
# fp_10 = open("{}/Result/Cosine_Similarity_Values_top10.txt".format(cwd), "w")
for i in range(len(all_terms)):
    fp.write("Document ID {}: \n". format(101+i))
    # fp_10.write("Document ID {}: \n".format(101 + i))
    print("Document ID {}: \n". format(101+i))
    for pairs in docs_similarity[i]:
        fp.write("{}: {}\n".format(pairs[0], pairs[1]))
        print("{}: {}".format(pairs[0], pairs[1]))
    # for pairs in docs_similarity[i][:10]:
        # fp_10.write("{}: {}\n".format(pairs[0], pairs[1]))
fp.close()
# fp_10.close()



