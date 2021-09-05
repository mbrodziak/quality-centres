#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import random
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from scipy.spatial import distance

print('''Wybierz zbiór danych:
    1 - zbiór Cars
    2 - zbiór CC General
    3 - zbiór Sales Transaction Dataset Weekly''')
dataset = input()

print("Podaj liczbę skupień: ")
k = input()
k = int(k)

print('''Wybierz miarę podobieństwa:
    1 - odległość Euklidesowa
    2 - odległość Manhattan
    3 - odległość Czebyszewa''')
kind_of_distance = input()

print('''Wybierz indeks miary:
    1 - indeks Dunna
    2 - indeks Daviesa-Bouldina''')
kind_of_index = input()

print()

if dataset == "1":
    data = pd.read_csv('./data/cars_p.csv', delimiter=";")
    print("Wybrano zbiór Cars")
elif dataset == "2":
    data = pd.read_csv('./data/CC GENERAL_p.csv', delimiter=";")
    print("Wybrano zbiór CC General")
elif dataset == "3":
    data = pd.read_csv('./data/Sales_Transactions_Dataset_Weekly_p.csv', delimiter=";")
    print("Wybrano zbiór Sales Transaction Dataset Weekly")
else:
    raise ValueError("Pod tym numerem nie ma zbioru danych!")
    
data = data.fillna(data.mean())

print("Podana liczba skupień: ", k)

if kind_of_distance == "1":
    print("Wybrano odległość Euklidesową")
elif kind_of_distance == "2":
    print("Wybrano odległość Manhattan")
elif kind_of_distance == "3":
    print("Wybrano odległość Czebyszewa")
    
if kind_of_index == "1":
    print("Wybrano indeks Dunna")
elif kind_of_index == "2":
    print("Wybrano indeks Daviesa-Bouldina")
    
print()

centroid = []
centroids = []
distances = []
p_distances = []
clusters = []
p_clusters = []
column_clusters = []
part_clusters = []
all_clusters = []
min_val = 0
min_in = 0
c = 0
mean = 0
means = []
is_migrate = True
dist_outside = []
dist_inside = []
che_dist_outside = []
che_dist_inside = []
min_value = 0
max_value = 0
index_dunna = 0
sum_inside = 0
p = 0
p_all = []
p_all_sum = []
p_all_sum_c = []
dist_cen = []
dist_centroids = []
r = []
r_max = 0
db_index = 0


def calculate_distances():
    global p_distances
    if kind_of_distance == "1":
        for i in range(k):
            distances.append(euclidean_distances(data, [centroids[i]]))
    elif kind_of_distance == "2":
        for i in range(k):
            distances.append(manhattan_distances(data, [centroids[i]]))
    elif kind_of_distance == "3":
        for i in range(k):
            for j in range(len(data)):
                p_distances.append(distance.chebyshev([data.iloc[j]], [centroids[i]]))
            distances.append(p_distances)
            p_distances = []
    else:
        raise ValueError("Pod tym numerem nie ma miary podobieństwa!")

        
def calculate_dunn_index():
    global max_value, che_dist_outside, che_dist_inside
    if kind_of_distance == "1":
        for i in range(len(all_clusters)):
            for j in range(i + 1, len(all_clusters)):
                if len(all_clusters[i]) != 0 and len(all_clusters[j]) != 0:
                    dist_outside.append(euclidean_distances(all_clusters[i], all_clusters[j]))

        for i in range(len(all_clusters)):
            if len(all_clusters[i]) != 0:
                dist_inside.append(euclidean_distances(all_clusters[i], all_clusters[i]))
    elif kind_of_distance == "2":
        for i in range(len(all_clusters)):
            for j in range(i + 1, len(all_clusters)):
                if len(all_clusters[i]) != 0 and len(all_clusters[j]) != 0:                
                    dist_outside.append(manhattan_distances(all_clusters[i], all_clusters[j]))

        for i in range(len(all_clusters)):
            if len(all_clusters[i]) != 0:
                dist_inside.append(manhattan_distances(all_clusters[i], all_clusters[i]))
    elif kind_of_distance == "3":
        for i in range(len(all_clusters)):
            for j in range(i + 1, len(all_clusters)):
                for l in range(len(all_clusters[i])):
                    for m in range(len(all_clusters[j])):
                        che_dist_outside.append(distance.chebyshev(all_clusters[i][l], all_clusters[j][m]))    
                dist_outside.append([che_dist_outside])
                che_dist_outside = []

        for i in range(len(all_clusters)):
            for j in range(len(all_clusters[i]) - 1):
                for l in range(j + 1, len(all_clusters[i])):
                    che_dist_inside.append(distance.chebyshev(all_clusters[i][j], all_clusters[i][l]))
            dist_inside.append([che_dist_inside])
            che_dist_inside = []

    if len(dist_outside) == 0 or (kind_of_distance == "3" and dist_outside == [[[]]]):
        min_value = 0
    else:
        if kind_of_distance == "3":
            if dist_outside[0] == [[]]:
                min_value = 1000000000
            else: 
                min_value = dist_outside[0][0][0]
        else:    
            min_value = dist_outside[0][0][0]
        for i in range(len(dist_outside)):
            for j in range(len(dist_outside[i])):
                for l in range(len(dist_outside[i][j])):
                    if dist_outside[i][j][l] < min_value:
                        min_value = dist_outside[i][j][l]

    for i in range(len(dist_inside)):
        for j in range(len(dist_inside[i])):
            for l in range(len(dist_inside[i][j])):
                if dist_inside[i][j][l] > max_value:
                    max_value = dist_inside[i][j][l]

    index_dunna = min_value / max_value
    print("Index Dunna:")
    print(index_dunna)


def calculate_davies_bouldin_index():
    global sum_inside, sum_all, r_max, dist_cen, che_dist_inside, p_all_sum, db_index
    if k == 1:
        db_index = 0
    else: 
        if kind_of_distance == "1":
            for i in range(len(all_clusters)):
                if len(all_clusters[i]) != 0:
                    dist_inside.append(euclidean_distances(all_clusters[i], [centroids[i]]))

            for i in range(len(centroids)):
                for j in range(len(centroids)):
                    if i != j:
                        dist_cen.append(euclidean_distances([centroids[i]], [centroids[j]]))
                dist_centroids.append(dist_cen)
                dist_cen = []
        elif kind_of_distance == "2":
            for i in range(len(all_clusters)):
                if len(all_clusters[i]) != 0:
                    dist_inside.append(manhattan_distances(all_clusters[i], [centroids[i]]))

            for i in range(len(centroids)):
                for j in range(len(centroids)):
                    if i != j:
                        dist_cen.append(manhattan_distances([centroids[i]], [centroids[j]]))
                dist_centroids.append(dist_cen)
                dist_cen = []
        elif kind_of_distance == "3": 
            for i in range(len(all_clusters)):
                if len(all_clusters[i]) != 0:
                    for j in range(len(all_clusters[i])):
                        che_dist_inside.append(distance.chebyshev(all_clusters[i][j], [centroids[i]]))
                dist_inside.append(che_dist_inside)
                che_dist_inside = []

            for i in range(len(centroids)):
                for j in range(len(centroids)):
                    if i != j:
                        dist_cen.append(distance.chebyshev([centroids[i]], [centroids[j]]))
                dist_centroids.append(dist_cen)
                dist_cen = []

        for i in range(len(dist_inside)):
            sum_inside = 0
            p = 0
            for j in range(len(dist_inside[i])):
                sum_inside += dist_inside[i][j]
            if kind_of_distance == "3":
                try:
                    p = sum_inside * (1 / len(dist_inside[i]))
                except ZeroDivisionError:
                    p = 0
            else:
                p = sum_inside * (1 / len(dist_inside[i]))
            p_all.append(p)

        if kind_of_distance == "1" or kind_of_distance == "2":
            if len(p_all) < len(centroids):
                tab = [0]
                cen = []
                for j in range(len(data.columns)):
                    cen.append(0)

                for i in range(len(centroids)):
                    if centroids[i] == cen:
                        p_all.insert(i, tab)


        for i in range(len(p_all)):
            for j in range(len(p_all)):
                if i != j:
                    if p_all[i] == [0] and p_all[j] == [0]:
                        p_all_sum.append([0])
                    else:
                        p_all_sum.append(p_all[i] + p_all[j]) 
            p_all_sum_c.append(p_all_sum)
            p_all_sum = []

        for i in range(len(p_all_sum_c)):
            r = []
            for j in range(len(p_all_sum_c[i])):
                if dist_centroids[i][j] == 0:
                    r.append([0])
                else:
                    r.append(p_all_sum_c[i][j] / dist_centroids[i][j])
            r_max += max(r)


        db_index = (1 / k) * r_max
    print("Index Daviesa-Bouldina: ")
    print(db_index)

if __name__ == '__main__':
    for i in range(k):
        rand = random.randint(0, len(data) - 1)
        for j in range(len(data.columns)):
            centroid.append(data.iloc[rand][j])
        centroids.append(centroid)
        centroid = []

    calculate_distances()

    while is_migrate:
        p_clusters = clusters
        clusters = []

        for i in range(len(data)):
            min_val = distances[0][i]
            min_in = 0
            for j in range(1, len(distances)):
                if distances[j][i] < min_val:
                    min_val = distances[j][i]
                    min_in = j
            clusters.append(min_in)

        if clusters == p_clusters:
            is_migrate = False
        else:
            centroids = []
            all_clusters = []

            for i in range(k):
                part_clusters = []
                for j in range(len(clusters)):
                    if clusters[j] == i:
                        for l in range(len(data.columns)):
                            column_clusters.append(data.at[j, data.columns[l]])
                        part_clusters.append(column_clusters)
                        column_clusters = []
                all_clusters.append(part_clusters)

            for i in range(len(all_clusters)):
                for l in range(len(data.columns)):
                    c = 0
                    mean = 0
                    for j in range (len(all_clusters[i])):
                        c = c + all_clusters[i][j][l]
                    if len(all_clusters[i]) > 0:
                        mean = c / len(all_clusters[i])
                    means.append(mean)
                centroids.append(means)
                means = []

            distances = []
            calculate_distances()

    if kind_of_index == "1":
        calculate_dunn_index()
    elif kind_of_index == "2":
        calculate_davies_bouldin_index()
    else:
        raise ValueError("Pod tym numerem nie ma indeksu miary!")


# In[ ]:




