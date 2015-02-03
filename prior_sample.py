#! /usr/local/bin/ipython
from os.path import isfile
import re
import itertools
from urllib import urlopen
from math import log, ceil, lgamma
import random as rd
import pickle
#import numpypy
import numpy as np
import pdb
from scipy.special import gammaln, betaln
from scipy.misc import logsumexp
from pylab import *
from copy import copy, deepcopy
import networkx as nx
import csv
import operator
import time
import warnings

def full_asym(N, gamma, tau, alpha):
    # samples N pairs, with parameters gamma, tau and alpha. Distribution is different for senders and receivers. Returns:
    #links: raw list of link pairs 
    #clusters: ground truth cluster allocation for each pair 
    #props: num_users x num_clusters matrix, ith row is the cluster proportions for the ith customer 
    #Z: links in matrix form
    #Zreordered: Z rearranged to show block structure
    #
    # increasing alpha increases the number of clusters, increasing gamma increases the number of customers, increasing tau increases the amount of overlap between clusters (and also somewhat increases the number of clusters), and also the amount of overlap between sender distributions and receiver distributions


    #set up empty structures
    customer_mk_s = {}
    table_to_dish_s = {}
    customer_mk_r = {}
    table_to_dish_r = {}
    table_mk = {}
    table_mk[-1] = gamma
    link_mk = {}
    link_mk[-1] = alpha
    links = []
    clusters = []

    for n in range(N):
        #sample link cluster
        try:
            lc = np.random.choice(link_mk.keys(),1,p=link_mk.values()/sum(link_mk.values()))[0]
        except ValueError:
            pdb.set_trace()
        if lc==-1:
            #add new cluster
            lc = max(link_mk.keys())+1
            customer_mk_s[lc] = {}
            customer_mk_s[lc][-1] = tau
            customer_mk_r[lc] = {}
            customer_mk_r[lc][-1] = tau
            table_to_dish_s[lc] = []
            table_to_dish_r[lc] = []
        link_mk[lc] = link_mk.get(lc,0)+1
        #sample sender
        try:
            sender_table = np.random.choice(customer_mk_s[lc].keys(), 1, p=customer_mk_s[lc].values()/sum(customer_mk_s[lc].values()))[0]
        except TypeError:
            pdb.set_trace()
        if sender_table==-1:
            #add new sender
            sender_table = max(customer_mk_s[lc].keys())+1
            dish = np.random.choice(table_mk.keys(), 1, p=table_mk.values()/sum(table_mk.values()))[0]
            if dish==-1:
                #add new person
                dish = max(table_mk.keys())+1
            table_to_dish_s[lc].append(dish)
            try:
                table_mk[dish] = table_mk.get(dish,0)+1
            except TypeError:
                pdb.set_trace()
        customer_mk_s[lc][sender_table] = customer_mk_s[lc].get(sender_table,0)+1
        sender = table_to_dish_s[lc][sender_table]

        #sample receiver
        receiver_table = np.random.choice(customer_mk_r[lc].keys(), 1, p=customer_mk_r[lc].values()/sum(customer_mk_r[lc].values()))[0]
        if receiver_table==-1:
            #add new receiver
            receiver_table = max(customer_mk_r[lc].keys())+1
            dish = np.random.choice(table_mk.keys(), 1, p=table_mk.values()/sum(table_mk.values()))[0]
            if dish==-1:
                #add new person
                dish = max(table_mk.keys())+1
            table_to_dish_r[lc].append(dish)
            try:
                table_mk[dish] = table_mk.get(dish,0)+1
            except TypeError:
                pdb.set_trace()
        customer_mk_r[lc][receiver_table] = customer_mk_r[lc].get(receiver_table,0)+1
        receiver = table_to_dish_r[lc][receiver_table]
        
        links.append((sender, receiver))
        clusters.append(lc)
    

    num_clusters = max(link_mk.keys())+1
    
    num_users = max(table_mk.keys())+1
    customer_cluster_counts = np.zeros((num_users,num_clusters))
    #count how many times each person belongs to each cluster
    for i in range(N):
        sender = links[i][0]
        receiver = links[i][1]
        lc = clusters[i]
        customer_cluster_counts[sender,lc]+=1
        customer_cluster_counts[receiver,lc]+=1

    #reorder customers to create a "pretty" block-structured Z matrix
    customer_max_cluster = np.argmax(customer_cluster_counts,1)

    Z = np.zeros((num_users, num_users))
    Zreordered = np.zeros((num_users,num_users))
    customer_switch = {}
    ll = 0
    for lc in range(num_clusters):
        for i in range(num_users):
            if customer_max_cluster[i]==lc:
                customer_switch[i] = ll
                ll+=1
    reordered_links = [(customer_switch[a], customer_switch[b]) for (a,b) in links]
    for link in links:
        sender = link[0]
        receiver = link[1]
        Z[sender, receiver]+=1

    for link in reordered_links:
        sender = link[0]
        receiver = link[1]
        Zreordered[sender, receiver]+=1

    props = (customer_cluster_counts.transpose()/np.sum(customer_cluster_counts,axis=1)).transpose()

    return links, clusters, props, Z, Zreordered
                 
            


def full_sym(N, gamma, tau, alpha):
    # samples N pairs, with parameters gamma, tau and alpha. Distribution is the same for senders and receivers, within a group. Returns:
    #links: raw list of link pairs 
    #clusters: ground truth cluster allocation for each pair 
    #props: num_users x num_clusters matrix, ith row is the cluster proportions for the ith customer 
    #Z: links in matrix form
    #Zreordered: Z rearranged to show block structure
    #
    # increasing alpha increases the number of clusters, increasing gamma increases the number of customers, increasing tau increases the amount of overlap between clusters (and also somewhat increases the number of clusters).


    customer_mk = {}
    table_to_dish = {}
    table_mk = {}
    table_mk[-1] = gamma
    link_mk = {}
    link_mk[-1] = alpha
    links = []
    clusters = []

    for n in range(N):
        #sample link cluster
        try:
            ptemp = np.array(link_mk.values())+0.
            lc = np.random.choice(link_mk.keys(),1,p=ptemp/sum(ptemp))[0]
        except ValueError:
            pdb.set_trace()
        if lc==-1:
            #add cluster
            lc = max(link_mk.keys())+1
            customer_mk[lc] = {}
            customer_mk[lc][-1] = tau
            table_to_dish[lc] = []
        link_mk[lc] = link_mk.get(lc,0)+1
        #sample sender
        try:
            ptemp = np.array(customer_mk[lc].values())+0.
            sender_table = np.random.choice(customer_mk[lc].keys(), 1, p = ptemp/sum(ptemp))[0]
        except TypeError:
            pdb.set_trace()
        if sender_table==-1:
            #add table
            sender_table = max(customer_mk[lc].keys())+1
            ptemp = np.array(table_mk.values())+0.
            dish = np.random.choice(table_mk.keys(), 1, p=ptemp/sum(ptemp))[0]
            if dish==-1:
                #add customer
                dish = max(table_mk.keys())+1
            table_to_dish[lc].append(dish)
            try:
                table_mk[dish] = table_mk.get(dish,0)+1
            except TypeError:
                pdb.set_trace()
        customer_mk[lc][sender_table] = customer_mk[lc].get(sender_table,0)+1
        sender = table_to_dish[lc][sender_table]

        #sample receiver
        #print customer_mk[lc].values()
        #print sum(customer_mk[lc].values())
        #print customer_mk[lc].values()/sum(customer_mk[lc].values())
        ptemp = np.array(customer_mk[lc].values())+0.
        receiver_table = np.random.choice(customer_mk[lc].keys(), 1, p = ptemp/sum(ptemp) )[0]
        if receiver_table==-1:
            #add table
            receiver_table = max(customer_mk[lc].keys())+1
            ptemp = np.array(table_mk.values())+0.
            dish = np.random.choice(table_mk.keys(), 1, p=ptemp/sum(ptemp))[0]
            if dish==-1:
                #add customer
                dish = max(table_mk.keys())+1
            table_to_dish[lc].append(dish)
            try:
                table_mk[dish] = table_mk.get(dish,0)+1
            except TypeError:
                pdb.set_trace()
        customer_mk[lc][receiver_table] = customer_mk[lc].get(receiver_table,0)+1
        receiver = table_to_dish[lc][receiver_table]
        
        links.append((sender, receiver))
        clusters.append(lc)
    

    num_clusters = max(link_mk.keys())+1
    
    num_users = max(table_mk.keys())+1
    customer_cluster_counts = np.zeros((num_users,num_clusters))
    for i in range(N):
        sender = links[i][0]
        receiver = links[i][1]
        lc = clusters[i]
        customer_cluster_counts[sender,lc]+=1
        customer_cluster_counts[receiver,lc]+=1

    customer_max_cluster = np.argmax(customer_cluster_counts,1)

    Z = np.zeros((num_users, num_users))
    Zreordered = np.zeros((num_users, num_users))
    customer_switch = {}
    ll = 0
    for lc in range(num_clusters):
        for i in range(num_users):
            if customer_max_cluster[i]==lc:
                customer_switch[i] = ll
                ll+=1

    reordered_links = [(customer_switch[a], customer_switch[b]) for (a,b) in links]
    
    for link in reordered_links:
        sender = link[0]
        receiver = link[1]
        Zreordered[sender, receiver]+=1

    for link in links:
        sender = link[0]
        receiver = link[1]
        Z[sender,receiver]+=1
    props = (customer_cluster_counts.transpose()/np.sum(customer_cluster_counts,axis=1)).transpose()

    return links, clusters, props, Z, Zreordered

                 
            


def simple_asym(N, gamma, tau):
    customer_mk_s = {}
    customer_mk_s[-1] = tau
    table_to_dish_s = []
    customer_mk_r = {}
    customer_mk_r[-1] = tau
    table_to_dish_r = []
    table_mk = {}
    table_mk[-1] = gamma
    links = []

    for n in range(N):
        
        #sample sender
        try:
            sender_table = np.random.choice(customer_mk_s.keys(), 1, p=customer_mk_s.values()/sum(customer_mk_s.values()))[0]
        except TypeError:
            pdb.set_trace()
        if sender_table==-1:
            sender_table = max(customer_mk_s.keys())+1
            dish = np.random.choice(table_mk.keys(), 1, p=table_mk.values()/sum(table_mk.values()))[0]
            if dish==-1:
                dish = max(table_mk.keys())+1
            table_to_dish_s.append(dish)
            try:
                table_mk[dish] = table_mk.get(dish,0)+1
            except TypeError:
                pdb.set_trace()
        customer_mk_s[sender_table] = customer_mk_s.get(sender_table,0)+1
        sender = table_to_dish_s[sender_table]

        #sample receiver
        receiver_table = np.random.choice(customer_mk_r.keys(), 1, p=customer_mk_r.values()/sum(customer_mk_r.values()))[0]
        if receiver_table==-1:
            receiver_table = max(customer_mk_r.keys())+1
            dish = np.random.choice(table_mk.keys(), 1, p=table_mk.values()/sum(table_mk.values()))[0]
            if dish==-1:
                dish = max(table_mk.keys())+1
            table_to_dish_r.append(dish)
            try:
                table_mk[dish] = table_mk.get(dish,0)+1
            except TypeError:
                pdb.set_trace()
        customer_mk_r[receiver_table] = customer_mk_r.get(receiver_table,0)+1
        receiver = table_to_dish_r[receiver_table]
        
        links.append((sender, receiver))
        
    
    num_users = max(table_mk.keys())+1

    Z = np.zeros((num_users, num_users))

    
    for link in links:
        sender = link[0]
        receiver = link[1]
        try:
            Z[sender, receiver]+=1
        except IndexError:
            pdb.set_trace()
    return links,  Z      










 
def simple_sym(N, gamma):
    customer_mk = {}
    customer_mk[-1] = gamma
    links = []

    for n in range(N):
        
        #sample sender
        try:
            sender = np.random.choice(customer_mk.keys(), 1, p=customer_mk.values()/sum(customer_mk.values()))[0]
        except TypeError:
            pdb.set_trace()
        if sender==-1:
            sender = max(customer_mk.keys())+1

        customer_mk[sender] = customer_mk.get(sender,0)+1
        
        #sample receiver
        receiver = np.random.choice(customer_mk.keys(), 1, p=customer_mk.values()/sum(customer_mk.values()))[0]
        if receiver==-1:
            receiver = max(customer_mk.keys())+1

        customer_mk[receiver] = customer_mk.get(receiver,0)+1
        
        links.append((sender, receiver))
        
    
    num_users = max(customer_mk.keys())+1

    Z = np.zeros((num_users, num_users))

    
    for link in links:
        sender = link[0]
        receiver = link[1]
        try:
            Z[sender, receiver]+=1
        except IndexError:
            pdb.set_trace()
    return links,  Z



