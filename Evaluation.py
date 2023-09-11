import os
from sklearn.metrics import classification_report
from scipy.stats import ttest_ind

cwd = os.getcwd()

ground_truth = dict()
for i in range(101, 151):
    text = open("{}/RelevanceFeedback/Dataset{}.txt".format(cwd, i))
    for line in text:
        id, weight = line.split(" ")[1], line.split(" ")[2]
        id = id.strip()
        weight = weight.strip()
        ground_truth[str(i)+id] = weight
text.close()

bm25_text = open("{}/Result/BM25_weights.txt".format(cwd))
bm_rel = dict()
for line in bm25_text:
    if line[0] == "B":
        datasetid = line[-5:-2]
    if line[:2].isdigit():
        id, weight = line.split(":")[0], line.split(":")[1]
        id = id.strip()
        weight = weight.replace("\n", "").strip()
        if float(weight) > 3.75:
            weight = 1
        else:
            weight = 0
        bm_rel[datasetid+id] = weight
text.close()


cosine_sim_text = open("{}/Result/Cosine_Similarity_Values.txt".format(cwd))
cosine_sim_rel = dict()
for line in cosine_sim_text:
    if line[0] == "D":
        datasetid = line[-6:-3]
    if line[:2].isdigit():
        id, weight = line.split(":")[0], line.split(":")[1]
        id = id.strip()
        weight = weight.strip()
        weight = weight.replace("\n", "").strip()
        if float(weight) > 0.2:
            weight = 1
        else:
            weight = 0
        cosine_sim_rel[datasetid+id] = weight
text.close()

rocc_cosine_sim_text = open("{}/Result/Rocchio_Cosine_Similarity_Values.txt".format(cwd))
rocc_cosine_sim_rel = dict()
for line in rocc_cosine_sim_text:
    if line[0] == "D":
        datasetid = line[-6:-3]
    if line[:2].isdigit():
        id, weight = line.split(":")[0], line.split(":")[1]
        id = id.strip()
        weight = weight.strip()
        weight = weight.replace("\n", "").strip()
        if float(weight) > 0.4:
            weight = 1
        else:
            weight = 0
        rocc_cosine_sim_rel[datasetid+id] = weight
text.close()

ground_truth_arr = []
bm_rel_arr = []
cos_sim_arr = []
rocc_cos_sim_arr = []

for id, rel in ground_truth.items():
    ground_truth_arr.append(int(rel))
    bm_rel_arr.append(bm_rel[id])
    cos_sim_arr.append(cosine_sim_rel[id])
    rocc_cos_sim_arr.append(rocc_cosine_sim_rel[id])

print(classification_report(ground_truth_arr, bm_rel_arr, digits=4))
print(classification_report(ground_truth_arr, cos_sim_arr, digits=4))
print(classification_report(ground_truth_arr, rocc_cos_sim_arr, digits=4))

# t-test between models
# seed the random number generator
# compare samples
stat, p = ttest_ind(cos_sim_arr, bm_rel_arr)
print('t=%.3f, p=%.3f' % (stat, p))
stat, p = ttest_ind(cos_sim_arr, rocc_cos_sim_arr)
print('t=%.3f, p=%.3f' % (stat, p))
stat, p = ttest_ind(bm_rel_arr, rocc_cos_sim_arr)
print('t=%.3f, p=%.3f' % (stat, p))