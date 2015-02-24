import numpy as np
import networkVI
#import prior_sample
import toy_data
def list2binarymat(links):
    linkmat = np.array(links)
    uni = np.unique(linkmat,return_index=False)
    N = linkmat.shape[0]
    K = uni.shape[0]
    sender = np.zeros((N,K))
    receiver = sender
    for n in range(N):
        for i in range(K):
            sender[n,i] = int(linkmat[n,0]==uni[i])
            receiver[n,i] = int(linkmat[n,1]==uni[i])
    return sender, receiver

if __name__=='__main__':
    # N = 100
    # gamma0 = 5
    # tau0 = 10
    # alpha0 = 5
    D = 5
    print 'begin sampling'
    links, clusters, Z = toy_data.toy_data(D, 50,  10, overlap=0, bg_prob = 0)
#    links, clusters, props, Z, Zreordered = prior_sample.full_sym(N, gamma0, tau0, alpha0)
    # print links
    sender, receiver = list2binarymat(links)

    print 'begin inference'
    (theta, eta, phi, tau, loglik_old) = networkVI.network_sym_VI(sender, receiver, D, thres=0.01)

    print loglik_old
    print 'Done. Param:'
    print 'theta'
    print theta
    print 'phi'
    print phi
    print 'tau'
    print tau
    print 'eta'
    print eta
