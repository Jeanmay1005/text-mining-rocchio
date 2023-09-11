import math
import string
from stemming.porter2 import stem
from coll import parse_rcv_coll
import os


def calc_df(coll):
    """Calculate DF of each term in vocab and return as term:df dictionary."""
    df_ = {}
    for id, doc in coll.get_docs().items():
        for term in doc.get_term_list():
            try:
                df_[term] += 1
            except KeyError:
                df_[term] = 1
    return df_


def create_hist(term_list):
    hist = dict()
    for t in term_list:
        if t in hist:
            hist[t] += 1
        else:
            hist[t] = 1
    return hist


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
                    line_list.append(line.replace("title", " ").strip())
                    print(line_list)
                else:
                    line = line.translate(str.maketrans('', '', string.punctuation))
                    line_list.append(line)
            long_text = "".join(line_list).replace("desc Description", " ")
            query[id] = long_text

    return query


def avg_doc_len(coll):
    tot_dl = 0
    for id, doc in coll.get_docs().items():
        tot_dl = tot_dl + doc.get_doc_len()
    return tot_dl/coll.get_num_docs()


def bm25(coll, q, df): # dataset's document frequency dictionary
    bm25s = {}
    avg_dl = avg_doc_len(coll)
    no_docs = coll.get_num_docs()
    for id, doc in coll.get_docs().items():
        query_terms = q.split()
        qfs = {}
        for t in query_terms:
            term = stem(t.lower())
            try:
                qfs[term] += 1
            except KeyError:
                qfs[term] = 1
        k = 1.2 * ((1 - 0.75) + 0.75 * (doc.get_doc_len() / float(avg_dl)))
        bm25_ = 0.0
        for qt in qfs.keys():
            n = 0
            if qt in df.keys():
                n = df[qt]
                f = doc.get_term_count(qt)
                qf = qfs[qt]
                bm = math.log(1.0 / ((n + 0.5) / (no_docs - n + 0.5)), 2) * (((1.2 + 1) * f) / (k + f)) * ( ((100 + 1) * qf) / float(100 + qf))
                # bm valuse may be negative if no_docs < 2n+1, so we may use 3*no_docs to solve this problem.
                bm25_ += bm
        bm25s[doc.get_docid()] = bm25_
    return bm25s


# Execution start
cwd = os.getcwd()

# directory for output
if not os.path.exists("{}/Result".format(cwd)):
    os.makedirs("{}/Result".format(cwd))

stopwords_f = open('{}/common-english-words.txt'.format(cwd), 'r')
stop_words = stopwords_f.read().split(',')
stopwords_f.close()

data_id = list(range(101, 151))
all_documents = dict()
for ids in data_id:
    all_documents[ids] = parse_rcv_coll("{}/DataCollection/Dataset{}".format(cwd, ids), stop_words)

q = query_process("{}/Topics.txt".format(cwd))

# fp_10 = open("{}/Result/BM25_weights_top10.txt".format(cwd), "w")
fp = open("{}/Result/BM25_weights.txt".format(cwd), "w")

for datasetid, datasetcoll in all_documents.items():
    bm25_query = []
    df = calc_df(datasetcoll)
    fp.write("BM25 score for DatasetID {}.\n".format(datasetid))
    # fp_10.write("BM25 score for DatasetID {}.\n".format(datasetid))
    print("BM25 score for DatasetID {}.\n".format(datasetid))
    fp.write("Query: {}.\n".format(q[str(datasetid)]))
    # fp_10.write("Query: {}.\n".format(q[str(datasetid)]))
    print("Query: {}.\n".format(q[str(datasetid)]))
    bm_score = bm25(datasetcoll, q[str(datasetid)], df)
    bm_sorted_list = sorted(bm_score.items(), key=lambda x: x[1], reverse=True)
    for a in bm_sorted_list:
        print(str(a[0])+" : " + str(a[1]) + "\n")
        fp.write(str(a[0])+" : " + str(a[1]) + "\n")
    # for a in bm_sorted_list[:10]:
    #     fp_10.write(str(a[0]) + " : " + str(a[1]) + "\n")
fp.close()
# fp_10.close()









