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
from stirling import *
from scipy.special import gammaln, betaln
from scipy.misc import logsumexp
from pylab import *
from copy import copy, deepcopy
import networkx as nx
import csv
import operator
import time
import warnings

def sample_from_prior_full_asym(N, gamma, tau, alpha):
    customer_mk_s = {}
    table_to_dish_s = {}
    customer_mk_r = {}
    table_to_dish_r = {}
    table_mk = {}
    table_mk[-1] = tau
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
            lc = max(link_mk.keys())+1
            customer_mk_s[lc] = {}
            customer_mk_s[lc][-1] = gamma
            customer_mk_r[lc] = {}
            customer_mk_r[lc][-1] = gamma
            table_to_dish_s[lc] = []
            table_to_dish_r[lc] = []
        link_mk[lc] = link_mk.get(lc,0)+1
        #sample sender
        try:
            sender_table = np.random.choice(customer_mk_s[lc].keys(), 1, p=customer_mk_s[lc].values()/sum(customer_mk_s[lc].values()))[0]
        except TypeError:
            pdb.set_trace()
        if sender_table==-1:
            sender_table = max(customer_mk_s[lc].keys())+1
            dish = np.random.choice(table_mk.keys(), 1, p=table_mk.values()/sum(table_mk.values()))[0]
            if dish==-1:
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
            receiver_table = max(customer_mk_r[lc].keys())+1
            dish = np.random.choice(table_mk.keys(), 1, p=table_mk.values()/sum(table_mk.values()))[0]
            if dish==-1:
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
    for i in range(N):
        sender = links[i][0]
        receiver = links[i][1]
        lc = clusters[i]
        customer_cluster_counts[sender,lc]+=1
        customer_cluster_counts[receiver,lc]+=1

    customer_max_cluster = np.argmax(customer_cluster_counts,1)

    Z = np.zeros((num_users, num_users))
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
        try:
            Z[sender, receiver]+=1
        except IndexError:
            pdb.set_trace()
    return links, reordered_links, Z




def sample_from_prior_full_sym(N, gamma, tau, alpha):
    customer_mk = {}
    table_to_dish = {}
    table_mk = {}
    table_mk[-1] = tau
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
            lc = max(link_mk.keys())+1
            customer_mk[lc] = {}
            customer_mk[lc][-1] = gamma
            table_to_dish[lc] = []
        link_mk[lc] = link_mk.get(lc,0)+1
        #sample sender
        try:
            sender_table = np.random.choice(customer_mk[lc].keys(), 1, p=customer_mk[lc].values()/sum(customer_mk[lc].values()))[0]
        except TypeError:
            pdb.set_trace()
        if sender_table==-1:
            sender_table = max(customer_mk[lc].keys())+1
            dish = np.random.choice(table_mk.keys(), 1, p=table_mk.values()/sum(table_mk.values()))[0]
            if dish==-1:
                dish = max(table_mk.keys())+1
            table_to_dish[lc].append(dish)
            try:
                table_mk[dish] = table_mk.get(dish,0)+1
            except TypeError:
                pdb.set_trace()
        customer_mk[lc][sender_table] = customer_mk[lc].get(sender_table,0)+1
        sender = table_to_dish[lc][sender_table]

        #sample receiver
        receiver_table = np.random.choice(customer_mk[lc].keys(), 1, p=customer_mk[lc].values()/sum(customer_mk[lc].values()))[0]
        if receiver_table==-1:
            receiver_table = max(customer_mk[lc].keys())+1
            dish = np.random.choice(table_mk.keys(), 1, p=table_mk.values()/sum(table_mk.values()))[0]
            if dish==-1:
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
        try:
            Z[sender, receiver]+=1
        except IndexError:
            pdb.set_trace()
    return links, reordered_links, Z















def sample_from_prior_simple_asym(N, gamma, tau):
    customer_mk_s = {}
    customer_mk_s[-1] = gamma
    table_to_dish_s = []
    customer_mk_r = {}
    customer_mk_r[-1] = gamma
    table_to_dish_r = []
    table_mk = {}
    table_mk[-1] = tau
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











def sample_from_prior_simple_sym(N, gamma):
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


def network_wl_asymmetric_restart(Ztrain, init_file, picklename, sample_params=1, maxit=1000,  init_K = -1, true_clusters = None,Ztest = None, fix_K = False):
    #np.random.seed(0)
    num_users = np.max(Ztrain)+1
    num_links = np.shape(Ztrain)[0]

    L=2*num_users
    if isfile(picklename):
        last_sample = pickle.load(open(picklename,"rb"))
        alpha = copy(last_sample['alpha'])
        gamma = copy(last_sample['gamma'])
        tau = copy(last_sample['tau'])
        beta = last_sample['beta'].copy()
        pies = deepcopy(last_sample['pies'])
        pier = deepcopy(last_sample['pier'])
        c = last_sample['c'].copy()
        mk = last_sample['mk'].copy()
        Kc = max(mk.keys())
        cluster_sender_counts = deepcopy(last_sample['cluster_sender_counts'])
        cluster_receiver_counts = deepcopy(last_sample['cluster_receiver_counts'])
        last_iter = copy(last_sample['iter'])
    if sample_params:
        alpha_params = init_file[0]
        gamma_params = init_file[1]
        tau_params = init_file[2]
    else:

        if sample_params:
            alpha_params = init_file[0]
            alpha = alpha_params[0]/alpha_params[1]
            gamma_params = init_file[1]
            gamma = gamma_params[0]/gamma_params[1]
            tau_params = init_file[2]
            tau = tau_params[0]/tau_params[1]
        else:
            alpha = init_file[0]
            gamma = init_file[1]
            tau = init_file[2]


        print repr(num_users) + " users; " + repr(num_links) + " emails"
        c = {}

        Kc = 0
        mk = {}
        mk[-1] = alpha
        cluster_sender_counts={}
        cluster_receiver_counts = {}
        beta = np.array([1.]*L)
        beta/=sum(beta)
        pies = {}
        pier = {}
        n=0


        if init_K >0:
            for link in Ztrain:
                if true_clusters is None:
                    c[n] = np.random.randint(init_K) #np.random.choice(init_K,1)[0]
                else:
                    c[n] = true_clusters[n]
                mk[c[n]] = mk.get(c[n],0.)+1
                if mk[c[n]] == 1:
                    cluster_sender_counts[c[n]] = {}
                    cluster_receiver_counts[c[n]] = {}
                cluster_sender_counts[c[n]][link[0]] = cluster_sender_counts[c[n]].get(link[0],0.)+1
                cluster_receiver_counts[c[n]][link[1]] = cluster_receiver_counts[c[n]].get(link[1],0.)+1
                n+=1

            Kc = init_K
        else:
            for link in Ztrain:
                link_cluster_pointer = np.searchsorted(np.cumsum(mk.values()), sum(mk.values())*np.random.random())
                link_cluster = mk.keys()[link_cluster_pointer]
                #pdb.set_trace()
                #link_cluster = np.random.choice(mk.keys(),1,p=mk.values()/sum(mk.values()))[0]
                if link_cluster==-1:
                    link_cluster = Kc
                    cluster_sender_counts[Kc] = {}
                    cluster_receiver_counts[Kc] = {}
                    Kc+=1

                c[n]=link_cluster
                mk[link_cluster] = mk.get(link_cluster,0.)+1
                n+=1

                cluster_sender_counts[link_cluster][link[0]] = cluster_sender_counts.get(link_cluster,{}).get(link[0],0.)+1
                cluster_receiver_counts[link_cluster][link[1]] = cluster_receiver_counts.get(link_cluster,{}).get(link[1],0.)+1

        gamrho = np.array([gamma/L]*L)

        for link_cluster in cluster_sender_counts.keys():
            for user in cluster_sender_counts[link_cluster].keys():
                if (cluster_sender_counts[link_cluster][user] == 1):
                    gamrho[user]+=1
                else:
                    num_times = int(cluster_sender_counts[link_cluster][user])
                    try:
                        gamrho[user] = gamrho[user] + partitionCRP(beta[link_cluster]*tau, num_times)
                    except TypeError:
                        pdb.set_trace()

        for link_cluster in cluster_receiver_counts.keys():
            for user in cluster_receiver_counts[link_cluster].keys():
                if (cluster_receiver_counts[link_cluster][user] == 1):
                    gamrho[user]+=1
                else:
                    num_times = int(cluster_receiver_counts[link_cluster][user])
                    gamrho[user] = gamrho[user] + partitionCRP(beta[link_cluster]*tau, num_times)

        beta = np.random.dirichlet(gamrho)
        while any((tau*beta)==0):
            beta = beta+np.spacing(1)
            beta/=sum(beta)

        for link_cluster in cluster_sender_counts.keys():
            tbns = tau*beta
            tbnr = tau*beta
            for sender in cluster_sender_counts[link_cluster].keys():
                tbns[sender] = tbns[sender]+cluster_sender_counts[link_cluster][sender]
            for receiver in cluster_receiver_counts[link_cluster].keys():
                tbnr[receiver] = tbnr[receiver]+cluster_receiver_counts[link_cluster][receiver]
            pies[link_cluster] = np.random.dirichlet(tbns)
            pier[link_cluster] = np.random.dirichlet(tbnr)
            while any(pies[link_cluster]==0):
                pies[link_cluster] = pies[link_cluster]+np.spacing(1)
                pies[link_cluster]/=sum(pies[link_cluster])
            while any(pier[link_cluster]==0):
                pier[link_cluster] = pier[link_cluster]+np.spacing(1)
                pier[link_cluster]/=sum(pier[link_cluster])
        if Ztest is not None:
            test_ll = predict_here(Ztest, mk,beta,pies,pier,alpha, L, num_users)

        if fix_K:
            mk[-1] = 0

        last_iter = 0

    for iter in range(last_iter, maxit):
        n=0
        if iter%10==0:
            print "iter " + repr(iter) +" of " + repr(maxit)

        for link in Ztrain:
            sender = link[0]
            receiver = link[1]
            link_cluster = c[n]
            #remove link
            mk[link_cluster] = mk.get(link_cluster,0.)-1
            if mk[link_cluster]<=0:
                del mk[link_cluster]
                del pies[link_cluster]
                del pier[link_cluster]
                del cluster_sender_counts[link_cluster]
                del cluster_receiver_counts[link_cluster]
            else:
                cluster_receiver_counts[link_cluster][receiver] = cluster_receiver_counts[link_cluster].get(receiver,0.)-1
                if cluster_receiver_counts[link_cluster][receiver]<1:
                    del cluster_receiver_counts[link_cluster][receiver]
                cluster_sender_counts[link_cluster][sender] = cluster_sender_counts[link_cluster].get(sender,0.)-1
                if cluster_sender_counts[link_cluster][sender]<1:
                    del cluster_sender_counts[link_cluster][sender]

            #get cluster probability for links
            lpc={}

            for link_cluster in mk:

                if link_cluster == -1:
                    try:
                        if beta[sender]==0:
                            pdb.set_trace()
                        if beta[receiver]==0:
                            pdb.set_trace()
                        lpc[link_cluster] = np.log(mk[link_cluster])+np.log(beta[sender]) + np.log(beta[receiver])
                    except RuntimeWarning:
                        pdb.set_trace()

                else:
                    try:
                        if mk[link_cluster]==0:
                            pdb.set_trace()
                        if pies[link_cluster][sender]==0:
                            pdb.set_trace()
                        if pier[link_cluster][receiver]==0:
                            pdb.set_trace()
                        lpc[link_cluster] = np.log(mk[link_cluster])+np.log(pies[link_cluster][sender])+np.log(pier[link_cluster][receiver])
                    except RuntimeWarning:
                        pdb.set_trace()
                if np.isnan(lpc[link_cluster]):
                    print 'nan! in' +link_cluster
                    pdb.set_trace()


            #lpc[-1] = np.log(alpha)+np.log(beta[sender]) + np.log(beta[receiver])
            lpctmp = lpc.copy()
            #minlpc = min(lpc.values())
            #lpc.update((x,y-minlpc) for x,y in lpc.items())
            lse = logsumexp(lpc.values())
            lpc.update((x,y-lse) for x,y in lpc.items())

            #sample new link cluster
            try:
                link_cluster_pointer = np.searchsorted(np.cumsum(np.exp(lpc.values())), np.random.random())
                link_cluster = lpc.keys()[link_cluster_pointer]
                #pdb.set_trace()
                #link_cluster = np.random.choice(lpc.keys(), 1, p=np.exp(lpc.values()))[0]
            except ValueError:
                pdb.set_trace()
            if link_cluster==-1:

                link_cluster = Kc
                cluster_sender_counts[Kc] = {}
                cluster_receiver_counts[Kc] = {}
                Kc+=1
                tbns = tau*beta
                tbnr = tau*beta
                tbns[sender] +=1
                tbnr[receiver]+=1
                pies[link_cluster] = np.random.dirichlet(tbns)
                pier[link_cluster] = np.random.dirichlet(tbnr)
                while any(pies[link_cluster]==0):
                    pies[link_cluster] = pies[link_cluster]+np.spacing(1)
                    pies[link_cluster]/=sum(pies[link_cluster])
                while any(pier[link_cluster]==0):
                    pier[link_cluster] = pier[link_cluster]+np.spacing(1)
                    pier[link_cluster]/=sum(pier[link_cluster])

            c[n]=link_cluster
            try:
                mk[link_cluster] = mk.get(link_cluster,0.)+1
            except TypeError:
                pdb.set_trace()
            if Ztest is not None:
                new_test_ll = predict_here(Ztest, mk,beta,pies,pier,alpha, L, num_users)

                pdb.set_trace()
                test_ll = new_test_ll
            n+=1
            #if n%10000==0:
            #    time_elapsed = time.time() - tlast
            #    tlast = time.time()
            #    print repr(n) + " of " + repr(num_links) +"; " + repr(time_elapsed)
            #    print "number of clusters: " +repr(len(mk))
            cluster_sender_counts[link_cluster][link[0]] = cluster_sender_counts[link_cluster].get(link[0],0.)+1
            cluster_receiver_counts[link_cluster][link[1]] = cluster_receiver_counts[link_cluster].get(link[1],0.)+1

        #sample beta
        gamrho = np.array([gamma/L]*L)
        for link_cluster in cluster_sender_counts.keys():
            for user in cluster_sender_counts[link_cluster].keys():
                num_times = int(cluster_sender_counts[link_cluster][user])
                if num_times == 1:
                    gamrho[user]+=1
                else:
                    gamrho[user] = gamrho[user] + partitionCRP(beta[user]*tau, num_times)

        for link_cluster in cluster_receiver_counts.keys():
            for user in cluster_receiver_counts[link_cluster].keys():
                num_times = int(cluster_receiver_counts[link_cluster][user])
                if num_times==1:
                    gamrho[user]+=1
                else:
                    gamrho[user] = gamrho[user] + partitionCRP(beta[user]*tau, num_times)

        beta = np.random.dirichlet(gamrho)
        while any((tau*beta)==0):
            beta = beta+np.spacing(1)
            beta/=sum(beta)

        #sample pie
        for link_cluster in cluster_sender_counts.keys():
            tbn = tau*beta
            for user in cluster_sender_counts[link_cluster].keys():
                tbn[user] = tbn[user]+cluster_sender_counts[link_cluster][user]
            pies[link_cluster] = np.random.dirichlet(tbn)
            while any(pies[link_cluster]==0):
                pies[link_cluster] = pies[link_cluster]+np.spacing(1)
                pies[link_cluster]/=sum(pies[link_cluster])
        for link_cluster in cluster_receiver_counts.keys():
            tbn = tau*beta
            for user in cluster_receiver_counts[link_cluster].keys():
                tbn[user] = tbn[user]+cluster_receiver_counts[link_cluster][user]
            pier[link_cluster] = np.random.dirichlet(tbn)
            while any(pier[link_cluster]==0):
                pier[link_cluster] = pier[link_cluster]+np.spacing(1)
                pier[link_cluster]/=sum(pier[link_cluster])
        if Ztest is not None:
            new_test_ll = predict_here(Ztest, mk,beta,pies,pier,alpha, L, num_users)

            pdb.set_trace()
            test_ll = new_test_ll
        if sample_params:
            stepsize = 0.1
            slice = True
            if slice == True:
                cc = 0
                w=1.
                u = np.random.rand()
                lp_old = (gamma_params[0]-1)*np.log(gamma) - gamma_params[1]*gamma
                ll_old = gammaln(gamma) - L*gammaln(gamma/L) + (gamma/L - 1) * sum(np.log(beta))
                l_old = lp_old+ll_old+np.log(u)

                rr = np.random.rand()
                s = [gamma-rr*w,gamma+(1-rr)*w]
                l0 = (gamma_params[0]-1)*np.log(s[0]) - gamma_params[1]*s[0] + gammaln(s[0]) - L*gammaln(s[0]/L) + (s[0]/L - 1) * sum(np.log(beta))
                l1 = (gamma_params[0]-1)*np.log(s[1]) - gamma_params[1]*s[1] + gammaln(s[1]) - L*gammaln(s[1]/L) + (s[1]/L - 1) * sum(np.log(beta))
                while (l0>l_old)&(l1>l_old):
                    s[0]-=w
                    s[1]+=w
                    l0 = (gamma_params[0]-1)*np.log(s[0]) - gamma_params[1]*s[0] + gammaln(s[0]) - L*gammaln(s[0]/L) + (s[0]/L - 1) * sum(np.log(beta))
                    l1 = (gamma_params[0]-1)*np.log(s[1]) - gamma_params[1]*s[1] + gammaln(s[1]) - L*gammaln(s[1]/L) + (s[1]/L - 1) * sum(np.log(beta))

                keepgoing = True
                while keepgoing:
                    gamma_prop = np.random.uniform(low=s[0], high=s[1])
                    l_new = (gamma_params[0]-1)*np.log(gamma_prop) - gamma_params[1]*gamma_prop + gammaln(gamma_prop) - L*gammaln(gamma_prop/L) + (gamma_prop/L - 1) * sum(np.log(beta))
                    if l_new>l_old:
                        gamma=gamma_prop
                        keepgoing = False
                    else:
                        cc+=1
                        if cc==50:
                            keepgoing = False
                        if gamma_prop>gamma:
                            s[1] = gamma_prop
                        else:
                            s[0] = gamma_prop

            else:

                gamma_prop = gamma + np.random.normal(loc=0,scale = stepsize,size=1)[0]
                if gamma_prop >0:
                    lp_old = (gamma_params[0]-1)*np.log(gamma) - gamma_params[1]*gamma
                    lp_new = (gamma_params[0]-1)*np.log(gamma_prop) - gamma_params[1]*gamma_prop

                    ll_old = gammaln(gamma) - L*gammaln(gamma/L) + (gamma/L - 1) * sum(np.log(beta))
                    ll_new = gammaln(gamma_prop) - L*gammaln(gamma_prop/L) + (gamma_prop/L-1)*sum(np.log(beta))

                    l_accept = lp_new + ll_new - lp_old - ll_old
                    r = np.random.rand()
                    if np.log(r)<l_accept:
                        gamma = gamma_prop + 0.


            if slice == True:
                w=1.
                cc=0
                u = np.random.rand()
                lp_old = (tau_params[0]-1)*np.log(tau) - tau_params[1]*tau
        print 'tau:'
        print tau
        ll_old = (len(mk)-1)*(gammaln(tau) - sum(gammaln(tau*beta)))
        print 'll_old:'
        print ll_old
        pdb.set_trace()
        for cluster in pies.keys():
            ll_old += sum((tau*beta-1)*(np.log(pies[cluster])+np.log(pier[cluster])))

            l_old = lp_old+ll_old+np.log(u)

            rr = np.random.rand()
            s = [tau-rr*w,tau+(1-rr)*w]
            l0 = (tau_params[0]-1)*np.log(s[0]) - tau_params[1]*s[0]+(len(mk)-1)*(gammaln(s[0])-sum(gammaln(s[0]*beta)))
            l1 = (tau_params[0]-1)*np.log(s[1]) - tau_params[1]*s[1] +(len(mk)-1)*(gammaln(s[1])-sum(gammaln(s[1]*beta)))
            for cluster in pies.keys():
                tmp = np.log(pies[cluster])+np.log(pier[cluster])
                l0 += sum((s[0]*beta-1)*tmp)
                l1 += sum((s[1]*beta-1)*tmp)
            while (l0>l_old)&(l1>l_old):
                s[0]-=w
                s[1]+=w
                l0 = (tau_params[0]-1)*np.log(s[0]) - tau_params[1]*s[0]+(len(mk)-1)*(gammaln(s[0])-sum(gammaln(s[0]*beta)))
                l1 = (tau_params[0]-1)*np.log(s[1]) - tau_params[1]*s[1] +(len(mk)-1)*(gammaln(s[1])-sum(gammaln(s[1]*beta)))
                for cluster in pies.keys():
                    tmp = np.log(pies[cluster])+np.log(pier[cluster])
                    l0 += sum((s[0]*beta-1)*tmp)
                    l1 += sum((s[1]*beta-1)*tmp)

            keepgoing = True
            while keepgoing:
                tau_prop = np.random.uniform(low=s[0], high=s[1])

                l_new = (tau_params[0]-1)*np.log(tau_prop) - tau_params[1]*tau_prop +(len(mk)-1)*(gammaln(tau_prop)-sum(gammaln(tau_prop*beta)))
                for cluster in pies.keys():
                    tmp = np.log(pies[cluster])+np.log(pier[cluster])
                    l_new += sum((tau_prop*beta-1)*tmp)

                if l_new>l_old:
                    tau=tau_prop
                    if any((tau*beta)==0):
                        keepgoing = True
                    else:
                        keepgoing = False
                else:
                    cc+=1
                    if cc==50:
                        keepgoing = False
                    if tau_prop>tau:
                        s[1] = tau_prop
                    else:
                        s[0] = tau_prop

            else:

                tau_prop = tau + np.random.normal(loc=0,scale=stepsize,size=1)[0]
                if all((tau_prop*beta) > 0):
                    lp_old = (tau_params[0]-1)*np.log(tau) - tau_params[1]*tau
                    lp_new = (tau_params[0]-1)*np.log(tau_prop) - tau_params[1]*tau_prop

                    ll_old = (len(mk)-1)*(gammaln(tau) - sum(gammaln(tau*beta)))
                    ll_new = (len(mk)-1)*(gammaln(tau_prop) - sum(gammaln(tau_prop*beta)))
                    for cluster in pies.keys():
                        ll_old += sum((tau*beta-1)*(np.log(pies[cluster])+np.log(pier[cluster])))
                        ll_new += sum((tau*beta-1)*(np.log(pies[cluster])+np.log(pier[cluster])))
                    l_accept = lp_new + ll_new - lp_old - ll_old
                    r = np.random.rand()

                    if np.log(r)<l_accept:
                        tau = tau_prop + 0.

            aeta = np.random.beta(alpha + 1, num_links)
            ap = (alpha_params[0] + len(mk) - 2)/(num_links*(alpha_params[1]-np.log(aeta)))


            ap = ap/(1+ap)
            r = np.random.rand()
            if r < ap:
                alpha = np.random.gamma(alpha_params[0]+len(mk)-1,1/(alpha_params[1]-log(aeta)))
            else:
                alpha = np.random.gamma(alpha_params[0]+len(mk)-2,1/(alpha_params[1]-log(aeta)))

            if fix_K:
                mk[-1] = 0
            else:
                mk[-1] = alpha

            if Ztest is not None:
                new_test_ll = predict_here(Ztest, mk,beta,pies,pier,alpha, L, num_users)

                pdb.set_trace()
                test_ll = new_test_ll
        save_every = 100
        if iter%save_every==0:
            last_sample={}
            last_sample['alpha'] = copy(alpha)
            last_sample['gamma'] = copy(gamma)
            last_sample['tau'] = copy(tau)
            last_sample['beta'] = beta.copy()
            last_sample['pies'] = deepcopy(pies)
            last_sample['pier'] = deepcopy(pier)
            last_sample['c'] = c.copy()
            last_sample['mk'] = mk.copy()
            last_sample['cluster_sender_counts'] = deepcopy(cluster_sender_counts)
            last_sample['cluster_receiver_counts'] = deepcopy(cluster_receiver_counts)
            last_sample['iter'] = copy(iter)
            print iter
            pickle.dump(last_sample,open(picklename,'wb'))
    return last_sample


def network_wl_asymmetric(Ztrain, init_file, sample_params=1, maxit=1000,init_K = -1, true_clusters = None,Ztest = None, fix_K = False):
    #np.random.seed(0)

    samples = {}
    if sample_params:
        alpha_params = init_file[0]
        alpha = alpha_params[0]/alpha_params[1]
        gamma_params = init_file[1]
        gamma = gamma_params[0]/gamma_params[1]
        tau_params = init_file[2]
        tau = tau_params[0]/tau_params[1]
    else:
        alpha = init_file[0]
        gamma = init_file[1]
        tau = init_file[2]

    num_users = np.max(Ztrain)+1
    num_links = np.shape(Ztrain)[0]

    L=2*num_users
    print repr(num_users) + " users; " + repr(num_links) + " emails"
    c = {}

    Kc = 0
    mk = {}
    mk[-1] = alpha
    cluster_sender_counts={}
    cluster_receiver_counts = {}
    beta = np.array([1.]*L)
    beta/=sum(beta)
    pies = {}
    pier = {}
    n=0

    stirling = {}
    if init_K >0:
        for link in Ztrain:
            if true_clusters is None:
                c[n] = np.random.randint(init_K) #np.random.choice(init_K,1)[0]
            else:
                c[n] = true_clusters[n]
            mk[c[n]] = mk.get(c[n],0.)+1
            if mk[c[n]] == 1:
                cluster_sender_counts[c[n]] = {}
                cluster_receiver_counts[c[n]] = {}
            cluster_sender_counts[c[n]][link[0]] = cluster_sender_counts[c[n]].get(link[0],0.)+1
            cluster_receiver_counts[c[n]][link[1]] = cluster_receiver_counts[c[n]].get(link[1],0.)+1
            n+=1

        Kc = init_K
    else:
        for link in Ztrain:
            link_cluster_pointer = np.searchsorted(np.cumsum(mk.values()), sum(mk.values())*np.random.random())
            link_cluster = mk.keys()[link_cluster_pointer]
            #pdb.set_trace()
            #link_cluster = np.random.choice(mk.keys(),1,p=mk.values()/sum(mk.values()))[0]
            if link_cluster==-1:
                link_cluster = Kc
                cluster_sender_counts[Kc] = {}
                cluster_receiver_counts[Kc] = {}
                Kc+=1

            c[n]=link_cluster
            mk[link_cluster] = mk.get(link_cluster,0.)+1
            n+=1

            cluster_sender_counts[link_cluster][link[0]] = cluster_sender_counts.get(link_cluster,{}).get(link[0],0.)+1
            cluster_receiver_counts[link_cluster][link[1]] = cluster_receiver_counts.get(link_cluster,{}).get(link[1],0.)+1

    gamrho = np.array([gamma/L]*L)

    for link_cluster in cluster_sender_counts.keys():
        print "."
        for user in cluster_sender_counts[link_cluster].keys():
            if (cluster_sender_counts[link_cluster][user] == 1):
                gamrho[user]+=1
            else:
                num_times = int(cluster_sender_counts[link_cluster][user])
                try:
                    gamrho[user] = gamrho[user] + partitionCRP(beta[link_cluster]*tau, num_times)
                except TypeError:
                    pdb.set_trace()

    for link_cluster in cluster_receiver_counts.keys():
        print "."
        for user in cluster_receiver_counts[link_cluster].keys():
            if (cluster_receiver_counts[link_cluster][user] == 1):
                gamrho[user]+=1
            else:
                num_times = int(cluster_receiver_counts[link_cluster][user])
                gamrho[user] = gamrho[user] + partitionCRP(beta[link_cluster]*tau, num_times)

    beta = np.random.dirichlet(gamrho)
    while any((tau*beta)==0):
        beta = beta+np.spacing(1)
        beta/=sum(beta)

    for link_cluster in cluster_sender_counts.keys():
        tbns = tau*beta
        tbnr = tau*beta
        for sender in cluster_sender_counts[link_cluster].keys():
            tbns[sender] = tbns[sender]+cluster_sender_counts[link_cluster][sender]
        for receiver in cluster_receiver_counts[link_cluster].keys():
            tbnr[receiver] = tbnr[receiver]+cluster_receiver_counts[link_cluster][receiver]
        pies[link_cluster] = np.random.dirichlet(tbns)
        pier[link_cluster] = np.random.dirichlet(tbnr)
        while any(pies[link_cluster]==0):
            pies[link_cluster] = pies[link_cluster]+np.spacing(1)
            pies[link_cluster]/=sum(pies[link_cluster])
        while any(pier[link_cluster]==0):
            pier[link_cluster] = pier[link_cluster]+np.spacing(1)
            pier[link_cluster]/=sum(pier[link_cluster])
    if Ztest is not None:
        test_ll = predict_here(Ztest, mk,beta,pies,pier,alpha, L, num_users)

    if fix_K:
        mk[-1] = 0
    for iter in range(maxit):
        n=0
        if iter%100==0:
            print "iter " + repr(iter) +" of " + repr(maxit)

        for link in Ztrain:
            sender = link[0]
            receiver = link[1]
            link_cluster = c[n]
            #remove link
            mk[link_cluster] = mk.get(link_cluster,0.)-1
            if mk[link_cluster]<=0:
                del mk[link_cluster]
                del pies[link_cluster]
                del pier[link_cluster]
                del cluster_sender_counts[link_cluster]
                del cluster_receiver_counts[link_cluster]
            else:
                cluster_receiver_counts[link_cluster][receiver] = cluster_receiver_counts[link_cluster].get(receiver,0.)-1
                if cluster_receiver_counts[link_cluster][receiver]<1:
                    del cluster_receiver_counts[link_cluster][receiver]
                cluster_sender_counts[link_cluster][sender] = cluster_sender_counts[link_cluster].get(sender,0.)-1
                if cluster_sender_counts[link_cluster][sender]<1:
                    del cluster_sender_counts[link_cluster][sender]

            #get cluster probability for links
            lpc={}

            for link_cluster in mk:

                if link_cluster == -1:
                    try:
                        if beta[sender]==0:
                            pdb.set_trace()
                        if beta[receiver]==0:
                            pdb.set_trace()
                        lpc[link_cluster] = np.log(mk[link_cluster])+np.log(beta[sender]) + np.log(beta[receiver])
                    except RuntimeWarning:
                        pdb.set_trace()

                else:
                    try:
                        if mk[link_cluster]==0:
                            pdb.set_trace()
                        if pies[link_cluster][sender]==0:
                            pdb.set_trace()
                        if pier[link_cluster][receiver]==0:
                            pdb.set_trace()
                        lpc[link_cluster] = np.log(mk[link_cluster])+np.log(pies[link_cluster][sender])+np.log(pier[link_cluster][receiver])
                    except RuntimeWarning:
                        pdb.set_trace()
                if np.isnan(lpc[link_cluster]):
                    print 'nan! in' +link_cluster
                    pdb.set_trace()


            #lpc[-1] = np.log(alpha)+np.log(beta[sender]) + np.log(beta[receiver])
            lpctmp = lpc.copy()
            #minlpc = min(lpc.values())
            #lpc.update((x,y-minlpc) for x,y in lpc.items())
            lse = logsumexp(lpc.values())
            lpc.update((x,y-lse) for x,y in lpc.items())

            #sample new link cluster
            try:
                link_cluster_pointer = np.searchsorted(np.cumsum(np.exp(lpc.values())), np.random.random())
                link_cluster = lpc.keys()[link_cluster_pointer]
                #pdb.set_trace()
                #link_cluster = np.random.choice(lpc.keys(), 1, p=np.exp(lpc.values()))[0]
            except ValueError:
                pdb.set_trace()
            if link_cluster==-1:

                link_cluster = Kc
                cluster_sender_counts[Kc] = {}
                cluster_receiver_counts[Kc] = {}
                Kc+=1
                tbns = tau*beta
                tbnr = tau*beta
                tbns[sender] +=1
                tbnr[receiver]+=1
                pies[link_cluster] = np.random.dirichlet(tbns)
                pier[link_cluster] = np.random.dirichlet(tbnr)
                while any(pies[link_cluster]==0):
                    pies[link_cluster] = pies[link_cluster]+np.spacing(1)
                    pies[link_cluster]/=sum(pies[link_cluster])
                while any(pier[link_cluster]==0):
                    pier[link_cluster] = pier[link_cluster]+np.spacing(1)
                    pier[link_cluster]/=sum(pier[link_cluster])

            c[n]=link_cluster
            try:
                mk[link_cluster] = mk.get(link_cluster,0.)+1
            except TypeError:
                pdb.set_trace()
            if Ztest is not None:
                new_test_ll = predict_here(Ztest, mk,beta,pies,pier,alpha, L, num_users)

                pdb.set_trace()
                test_ll = new_test_ll
            n+=1
            #if n%10000==0:
            #    time_elapsed = time.time() - tlast
            #    tlast = time.time()
            #    print repr(n) + " of " + repr(num_links) +"; " + repr(time_elapsed)
            #    print "number of clusters: " +repr(len(mk))
            cluster_sender_counts[link_cluster][link[0]] = cluster_sender_counts[link_cluster].get(link[0],0.)+1
            cluster_receiver_counts[link_cluster][link[1]] = cluster_receiver_counts[link_cluster].get(link[1],0.)+1

        #sample beta
        gamrho = np.array([gamma/L]*L)
        for link_cluster in cluster_sender_counts.keys():
            for user in cluster_sender_counts[link_cluster].keys():
                num_times = int(cluster_sender_counts[link_cluster][user])
                if num_times == 1:
                    gamrho[user]+=1
                else:
                    gamrho[user] = gamrho[user] + partitionCRP(beta[user]*tau, num_times)

        for link_cluster in cluster_receiver_counts.keys():
            for user in cluster_receiver_counts[link_cluster].keys():
                num_times = int(cluster_receiver_counts[link_cluster][user])
                if num_times==1:
                    gamrho[user]+=1
                else:
                    gamrho[user] = gamrho[user] + partitionCRP(beta[user]*tau, num_times)

        beta = np.random.dirichlet(gamrho)
        while any((tau*beta)==0):
            beta = beta+np.spacing(1)
            beta/=sum(beta)

        #sample pie
        for link_cluster in cluster_sender_counts.keys():
            tbn = tau*beta
            for user in cluster_sender_counts[link_cluster].keys():
                tbn[user] = tbn[user]+cluster_sender_counts[link_cluster][user]
            pies[link_cluster] = np.random.dirichlet(tbn)
            while any(pies[link_cluster]==0):
                pies[link_cluster] = pies[link_cluster]+np.spacing(1)
                pies[link_cluster]/=sum(pies[link_cluster])
        for link_cluster in cluster_receiver_counts.keys():
            tbn = tau*beta
            for user in cluster_receiver_counts[link_cluster].keys():
                tbn[user] = tbn[user]+cluster_receiver_counts[link_cluster][user]
            pier[link_cluster] = np.random.dirichlet(tbn)
            while any(pier[link_cluster]==0):
                pier[link_cluster] = pier[link_cluster]+np.spacing(1)
                pier[link_cluster]/=sum(pier[link_cluster])
        if Ztest is not None:
            new_test_ll = predict_here(Ztest, mk,beta,pies,pier,alpha, L, num_users)

            pdb.set_trace()
            test_ll = new_test_ll
        if sample_params:
            stepsize = 0.1
            slice = True
            if slice == True:
                cc = 0
                w=1.
                u = np.random.rand()
                lp_old = (gamma_params[0]-1)*np.log(gamma) - gamma_params[1]*gamma
                ll_old = gammaln(gamma) - L*gammaln(gamma/L) + (gamma/L - 1) * sum(np.log(beta))
                l_old = lp_old+ll_old+np.log(u)

                rr = np.random.rand()
                s = [gamma-rr*w,gamma+(1-rr)*w]
                l0 = (gamma_params[0]-1)*np.log(s[0]) - gamma_params[1]*s[0] + gammaln(s[0]) - L*gammaln(s[0]/L) + (s[0]/L - 1) * sum(np.log(beta))
                l1 = (gamma_params[0]-1)*np.log(s[1]) - gamma_params[1]*s[1] + gammaln(s[1]) - L*gammaln(s[1]/L) + (s[1]/L - 1) * sum(np.log(beta))
                while (l0>l_old)&(l1>l_old):
                    s[0]-=w
                    s[1]+=w
                    l0 = (gamma_params[0]-1)*np.log(s[0]) - gamma_params[1]*s[0] + gammaln(s[0]) - L*gammaln(s[0]/L) + (s[0]/L - 1) * sum(np.log(beta))
                    l1 = (gamma_params[0]-1)*np.log(s[1]) - gamma_params[1]*s[1] + gammaln(s[1]) - L*gammaln(s[1]/L) + (s[1]/L - 1) * sum(np.log(beta))

                keepgoing = True
                while keepgoing:
                    gamma_prop = np.random.uniform(low=s[0], high=s[1])
                    l_new = (gamma_params[0]-1)*np.log(gamma_prop) - gamma_params[1]*gamma_prop + gammaln(gamma_prop) - L*gammaln(gamma_prop/L) + (gamma_prop/L - 1) * sum(np.log(beta))
                    if l_new>l_old:
                        gamma=gamma_prop
                        keepgoing = False
                    else:
                        cc+=1
                        if cc==50:
                            keepgoing = False
                        if gamma_prop>gamma:
                            s[1] = gamma_prop
                        else:
                            s[0] = gamma_prop

            else:

                gamma_prop = gamma + np.random.normal(loc=0,scale = stepsize,size=1)[0]
                if gamma_prop >0:
                    lp_old = (gamma_params[0]-1)*np.log(gamma) - gamma_params[1]*gamma
                    lp_new = (gamma_params[0]-1)*np.log(gamma_prop) - gamma_params[1]*gamma_prop

                    ll_old = gammaln(gamma) - L*gammaln(gamma/L) + (gamma/L - 1) * sum(np.log(beta))
                    ll_new = gammaln(gamma_prop) - L*gammaln(gamma_prop/L) + (gamma_prop/L-1)*sum(np.log(beta))

                    l_accept = lp_new + ll_new - lp_old - ll_old
                    r = np.random.rand()
                    if np.log(r)<l_accept:
                        gamma = gamma_prop + 0.


            if slice == True:
                w=1.
                cc=0
                u = np.random.rand()
                lp_old = (tau_params[0]-1)*np.log(tau) - tau_params[1]*tau

                ll_old = (len(mk)-1)*(gammaln(tau) - sum(gammaln(tau*beta)))

                for cluster in pies.keys():
                    ll_old += sum((tau*beta-1)*(np.log(pies[cluster])+np.log(pier[cluster])))

                l_old = lp_old+ll_old+np.log(u)

                rr = np.random.rand()
                s = [tau-rr*w,tau+(1-rr)*w]
                l0 = (tau_params[0]-1)*np.log(s[0]) - tau_params[1]*s[0]+(len(mk)-1)*(gammaln(s[0])-sum(gammaln(s[0]*beta)))
                l1 = (tau_params[0]-1)*np.log(s[1]) - tau_params[1]*s[1] +(len(mk)-1)*(gammaln(s[1])-sum(gammaln(s[1]*beta)))
                for cluster in pies.keys():
                    tmp = np.log(pies[cluster])+np.log(pier[cluster])
                    l0 += sum((s[0]*beta-1)*tmp)
                    l1 += sum((s[1]*beta-1)*tmp)
                while (l0>l_old)&(l1>l_old):
                    s[0]-=w
                    s[1]+=w
                    l0 = (tau_params[0]-1)*np.log(s[0]) - tau_params[1]*s[0]+(len(mk)-1)*(gammaln(s[0])-sum(gammaln(s[0]*beta)))
                    l1 = (tau_params[0]-1)*np.log(s[1]) - tau_params[1]*s[1] +(len(mk)-1)*(gammaln(s[1])-sum(gammaln(s[1]*beta)))
                    for cluster in pies.keys():
                        tmp = np.log(pies[cluster])+np.log(pier[cluster])
                        l0 += sum((s[0]*beta-1)*tmp)
                        l1 += sum((s[1]*beta-1)*tmp)

                keepgoing = True
                while keepgoing:
                    tau_prop = np.random.uniform(low=s[0], high=s[1])

                    l_new = (tau_params[0]-1)*np.log(tau_prop) - tau_params[1]*tau_prop +(len(mk)-1)*(gammaln(tau_prop)-sum(gammaln(tau_prop*beta)))
                    for cluster in pies.keys():
                        tmp = np.log(pies[cluster])+np.log(pier[cluster])
                        l_new += sum((tau_prop*beta-1)*tmp)

                    if l_new>l_old:
                        tau=tau_prop
                        keepgoing = False
                    else:
                        cc+=1
                        if cc==50:
                            keepgoing = False
                        if tau_prop>tau:
                            s[1] = tau_prop
                        else:
                            s[0] = tau_prop

            else:

                tau_prop = tau + np.random.normal(loc=0,scale=stepsize,size=1)[0]
                if tau_prop > 0:
                    lp_old = (tau_params[0]-1)*np.log(tau) - tau_params[1]*tau
                    lp_new = (tau_params[0]-1)*np.log(tau_prop) - tau_params[1]*tau_prop

                    ll_old = (len(mk)-1)*(gammaln(tau) - sum(gammaln(tau*beta)))
                    ll_new = (len(mk)-1)*(gammaln(tau_prop) - sum(gammaln(tau_prop*beta)))
                    for cluster in pies.keys():
                        ll_old += sum((tau*beta-1)*(np.log(pies[cluster])+np.log(pier[cluster])))
                        ll_new += sum((tau*beta-1)*(np.log(pies[cluster])+np.log(pier[cluster])))
                    l_accept = lp_new + ll_new - lp_old - ll_old
                    r = np.random.rand()

                    if np.log(r)<l_accept:
                        tau = tau_prop + 0.

            aeta = np.random.beta(alpha + 1, num_links)
            ap = (alpha_params[0] + len(mk) - 2)/(num_links*(alpha_params[1]-np.log(aeta)))


            ap = ap/(1+ap)
            r = np.random.rand()
            if r < ap:
                alpha = np.random.gamma(alpha_params[0]+len(mk)-1,1/(alpha_params[1]-log(aeta)))
            else:
                alpha = np.random.gamma(alpha_params[0]+len(mk)-2,1/(alpha_params[1]-log(aeta)))

            if fix_K:
                mk[-1] = 0
            else:
                mk[-1] = alpha

            if Ztest is not None:
                new_test_ll = predict_here(Ztest, mk,beta,pies,pier,alpha, L, num_users)

                pdb.set_trace()
                test_ll = new_test_ll
        sample_every = 10
        if iter%sample_every==0:
            samples[iter/sample_every] = {}
            samples[iter/sample_every]['alpha'] = copy(alpha)
            samples[iter/sample_every]['gamma'] = copy(gamma)
            samples[iter/sample_every]['tau'] = copy(tau)
            samples[iter/sample_every]['beta'] = beta.copy()
            samples[iter/sample_every]['pies'] = deepcopy(pies)
            samples[iter/sample_every]['pier'] = deepcopy(pier)
            samples[iter/sample_every]['c'] = c.copy()
            samples[iter/sample_every]['mk'] = mk.copy()
            samples[iter/sample_every]['cluster_sender_counts'] = deepcopy(cluster_sender_counts)
            samples[iter/sample_every]['cluster_receiver_counts'] = deepcopy(cluster_receiver_counts)
            print iter

    return samples


def partitionCRP(alpha, N):
    mk = [alpha,0]
    mk[-1] = alpha
    K=0
    for iter in range(N):
        r = np.random.rand()
        p = alpha/(alpha+iter)
        if r<p:
            K+=1

    return K


if __name__ == '__main__':
    num_clusters = 5
    cluster_size = 20
    num_per_cluster = 40
    Z, clusters = toy_data(num_clusters, cluster_size,num_per_cluster)
    init_file = [1., 1., 1.]
    samples = network_wl_symmetric(Z,init_file)
    print 'done'




