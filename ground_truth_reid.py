import os


queries_list = os.listdir("reid_queries/")
reid_list = os.listdir("reid_validation/")


ground_truth = {}

for i in queries_list:
    id = i[:4]
    list=[]
    for j in reid_list:
        if j.startswith(id):
            ground_truth[i]=list
            ground_truth[i].append(j)

print(ground_truth)