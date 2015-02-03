#! /usr/local/bin/ipython
import os
import re
import itertools


import random as rd
import numpy as np







def toy_data(num_clusters, cluster_size,  num_per_cluster, overlap=0, bg_prob = 0):
    #generates toy data with num_cluster clusters of size cluster_size+overlap, and overlap of overlap. bg_prob adds background noise.
    links =[]
    clusters = []
    num_users = num_clusters*cluster_size+overlap
    for i in range(num_clusters):
        for n in range(num_per_cluster):
            clusters.append(i)
            if np.random.rand()<bg_prob:
                sr = rd.sample(xrange(0,num_users),2)
            else:
                sr = rd.sample(xrange(i*cluster_size, (i+1)*cluster_size+overlap),2)
            links.append(sr)
    
    Z = np.zeros((num_users,num_users))
    for link in links:
        Z[link[0],link[1]]+=1

    return links, clusters, Z
