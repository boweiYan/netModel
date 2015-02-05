import numpy as np
import networkVI
import prior_sample

def list2binarymat(links):
    linkmat = np.array(links)
    uni = np.unique(linkmat,return_index=False)
    K = uni.shape[0]
    sender = np.zeros((N,K))
    receiver = sender
    for n in range(N):
        for i in range(K):
            sender[n,i] = int(linkmat[n,0]==uni[i])
            receiver[n,i] = int(linkmat[n,1]==uni[i])
    return sender, receiver

if __name__=='__main__':
    N = 100
    gamma0 = 5
    tau0 = 10
    alpha0 = 5
    print 'begin sampling'
    links, clusters, props, Z, Zreordered = prior_sample.full_sym(N, gamma0, tau0, alpha0)
    sender, receiver = list2binarymat(links)

    print 'begin inference'
    #TODO: Is it safe to set a D for prior sampling from non-parametric model?
    (alpha, gamma, eta, phi, tau, loglik_old) = networkVI.network_sym_VI(sender, receiver, D=10, thres=0.001)

    print (alpha, gamma, tau, loglik_old)