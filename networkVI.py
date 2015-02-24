import numpy as np
import numpy.random.mtrand
import scipy.special
import scipy.optimize
import math
import sys
import pdb
import time
#import prior_sample

eps = 1e-50
def trigamma(x) :
    return scipy.special.polygamma(1,x)

def newton_raphson(alpha,gamma,M,stepsize,thres):
    '''
    :param alpha: K-vector
    :param gamma: should be a K*M matrix, each column corresponds to an item in alpha
    :param M:
    :param stepsize:
    :param thres:
    :return:
    '''
    K = alpha.shape[0]
    g = np.zeros(K)
    h = np.zeros(K)
    alpha_new = alpha
    diff = 1e10
    while diff > thres:
        alpha_old = alpha_new
        z = M*trigamma(np.sum(alpha_old))
        for i in range(K):
            h[i] = -M*trigamma(alpha_old[i])
            g[i] = M*scipy.special.digamma(np.sum(alpha_old))-M*scipy.special.digamma(alpha_old[i])
            for d in range(M):
                g[i] += scipy.special.digamma(gamma[i, d])-scipy.special.digamma(np.sum(gamma[:, d]))

        c = np.sum(g/h)/(1/z+np.sum(1/h))
        # print (g,h,c)
        for i in range(K):
            alpha_new[i] = alpha_old[i]-stepsize*(g[i]-c)/h[i]

        diff = np.linalg.norm(alpha_new-alpha_old)

    return alpha_new

def getloglik(theta, tau, eta, phi, sender, receiver):
    D = theta.shape[0]
    N = phi.shape[0]
    K = tau.shape[0]

    loglik = 0
    for n in range(N):
        for d in range(D):
            loglik += phi[n,d]*np.log(theta[d])
    for d in range(D):
        for n in range(N):
            for j in range(K):
                for l in range(K):
                    loglik += phi[n,d]*(sender[n,j]*(scipy.special.digamma(eta[d,j]+eps)-scipy.special.digamma(np.sum(eta[d,:])+eps))+receiver[n,l]*(scipy.special.digamma(eta[d,l]+eps)-scipy.special.digamma(np.sum(eta[d,:])+eps)))
        loglik += scipy.special.gammaln(np.sum(tau)+eps)
        for i in range(K):
            loglik -= scipy.special.gammaln(tau[i]+eps)
            loglik += (tau[i]-1)*(scipy.special.digamma(eta[d,i]+eps)-scipy.special.digamma(np.sum(eta[d,:])))
            
    for d in range(D):
        loglik -= scipy.special.gammaln(np.sum(eta[d,:])+eps)
        for i in range(K):
            loglik += scipy.special.gammaln(eta[d,i]+eps)
            loglik -= (eta[d,i]-1)*(scipy.special.digamma(eta[d,i])-scipy.special.digamma(np.sum(eta[d,:])))
        for n in range(N):
            loglik -= phi[n,d]*np.log(phi[n,d])

    return loglik


def network_sym_VI(sender, receiver, D, thres):
    '''
    parameter estimate for symmetric link prediction model with variational inference approach
    :param sender: N*K binary
    :param receiver: N*K binary
    :param D: number of clusters
    :param thres: convergence threshold
    :return: alpha[D], gamma[D], eta[D*K], phi[N*D], tau[K]
    '''

    K = sender.shape[1]
    N = sender.shape[0]

    # Initialization
    theta = 1./D*np.ones(D)
    eta = 1./K*np.ones((D,K))
    
    #phi = 1./D * np.ones((N,D))
    phi = np.random.rand(N,D)
    for n in range(N):
 	phi[n,:] = phi[n,:]/np.sum(phi[n,:])
    tau = 1./K * np.ones(K)

    loglik_old = [0]
    loglik = 1
    iter=1
    '''
    # Set eta to truth
    print K
    for d in range(D):
        eta[d,:] = np.zeros(K)
        eta[d,d*(K/D):(d+1)*(K/D)] = np.ones(K/D)
    '''
    # Iteration until convergence
    while abs(loglik-loglik_old[-1]) > thres:
        loglik_old.append(loglik)
        iter+=1
        print(time.strftime('%H:%M:%S',time.localtime()))
        print "\n Iteration: "+str(iter)+" loglik "+str(loglik)

        # Update global parameters(tau, alpha)
        print 'Updating tau:'
        tau = newton_raphson(tau, np.transpose(eta), D, .1, thres)

        # For each document, estimate local latent variables
        print 'updating phi and eta'
        for n in range(N):
	    tmp = np.zeros(D)
            for d in range(D):
                phi[n,d] = theta[d]
                for j in range(K):
		    tmp[d] += sender[n,j]*(scipy.special.digamma(eta[d,j])-scipy.special.digamma(np.sum(eta[d,:])))
                    phi[n,d] *= np.exp(sender[n,j]*(scipy.special.digamma(eta[d,j])-scipy.special.digamma(np.sum(eta[d,:]))))
                    phi[n,d] *= np.exp(receiver[n,j]*(scipy.special.digamma(eta[d,j])-scipy.special.digamma(np.sum(eta[d,:]))))
                # Normalize phi
            #if n==0:
	    #	pdb.set_trace()
	    phi[n,:] /= np.sum(phi[n,:])
        # Update eta
        
        for d in range(D):
            for i in range(K):
                eta[d,i] = tau[i]+np.dot(phi[:,d],sender[:,i]+receiver[:,i])
        

        # check convergence (likelihood)
        loglik = getloglik(theta, tau, eta, phi, sender, receiver)
        sys.stdout.flush()

    return (theta, eta, phi, tau, loglik_old)

if __name__=='__main__':
    print "nothing done"
