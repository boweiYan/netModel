import numpy as np
import numpy.random.mtrand
import scipy.special
import scipy.optimize
import math
#import prior_sample

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
                g[i]+=scipy.special.digamma(gamma[i,d])-scipy.special.digamma(np.sum(gamma[:,d]))

        c = np.sum(g/h)/(1/z+np.sum(1/h))
        for i in range(K):
            alpha_new[i] = alpha_old[i]-stepsize*(g[i]-c)/h[i]

        diff = np.linalg.norm(alpha_new-alpha_old)

    return alpha_new

def getloglik(alpha, tau, gamma, eta, phi, sender, receiver):
    D = alpha.shape[0]
    N = phi.shape[0]
    K = tau.shape[0]
    eps = 1e-50
    loglik = np.log(scipy.special.gamma(np.sum(alpha))+eps)-np.sum(np.log(scipy.special.gamma(alpha)))
    for d in range(D):
        loglik += (alpha[d]-1)*(scipy.special.digamma(gamma[d]+eps)-scipy.special.digamma(np.sum(gamma)))
        for n in range(N):
            loglik += phi[n,d]*(scipy.special.digamma(gamma[d]+eps)-scipy.special.digamma(np.sum(gamma)))
            for j in range(K):
                for l in range(K):
                    loglik += phi[n,d]*(sender[n,j]*(scipy.special.digamma(eta[d,j]+eps)-scipy.special.digamma(np.sum(eta[d,:])+eps))+receiver[n,l]*(scipy.special.digamma(eta[d,l]+eps)-scipy.special.digamma(np.sum(eta[d,:])+eps)))
        loglik += np.log(scipy.special.gamma(np.sum(tau)+eps))
        for i in range(K):
            loglik -= np.log(scipy.special.gamma(tau[i]+eps))
            loglik += (tau[i]-1)*(scipy.special.digamma(eta[d,i]+eps)-scipy.special.digamma(np.sum(eta[d,:])))

    loglik -= np.log(scipy.special.gamma(np.sum(gamma)+eps))
    for d in range(D):
        loglik += np.log(scipy.special.gamma(gamma[d]+eps))
        loglik -= (gamma[d]-1)*(scipy.special.digamma(gamma[d])-scipy.special.digamma(np.sum(gamma)))
        loglik -= np.log(scipy.special.gamma(np.sum(eta[d,:])+eps))
        for i in range(K):
            loglik += np.log(scipy.special.gamma(eta[d,i]+eps))
            loglik -= (eta[d,i]-1)*(scipy.special.digamma(eta[d,i])-scipy.special.digamma(np.sum(eta[d,:])))
        for n in range(N):
            loglik -= phi[n,d]*np.log(phi[n,d])

    return loglik

def functau(tau,D,eta):
    k = tau.shape[0]
    f = np.zeros(k)
    for i in range(k):
        f[i]= D*np.log(scipy.special.gamma(np.sum(tau)))-D*np.log(tau[i])
        for d in range(D):
            f[i] += (tau[i]-1)*(scipy.special.digamma(eta[d,i])-scipy.special.digamma(np.sum(eta[d,:])))
    return -np.sum(f)

def dertau(tau,D,eta):
    n = tau.shape[0]
    f = np.zeros(n)
    for i in range(n):
        f[i] = D*scipy.special.digamma(np.sum(tau))-D*scipy.special.digamma(tau[i])
        for d in range(D):
            f[i] += scipy.special.digamma(eta[d,i])-scipy.special.digamma(np.sum(eta[d,:]))
    return -f

def hesstau(tau,D,eta):
    n = tau.shape[0]
    hess = np.ones((n,n))*trigamma(np.sum(tau))*D
    for i in range(n):
        hess[i,i] -= D*trigamma(tau[i])
    return -hess

def funcalpha(alpha,gamma):
    k = alpha.shape[0]
    f = np.zeros(k)
    for i in range(k):
        f[i]= np.log(scipy.special.gamma(np.sum(alpha)))-np.log(alpha[i])+(alpha[i]-1)*(scipy.special.digamma(gamma[i])-scipy.special.digamma(np.sum(gamma[i])))
    return -np.sum(f)

def deralpha(alpha,gamma):
    k = alpha.shape[0]
    f = np.zeros(k)
    for i in range(k):
        f[i] = scipy.special.digamma(np.sum(alpha))-scipy.special.digamma(alpha[i])+scipy.special.digamma(gamma[i])-scipy.special.digamma(np.sum(gamma[i]))
    return -f

def hessalpha(alpha,gamma):
    k = alpha.shape[0]
    hess = np.ones((k,k))*trigamma(np.sum(alpha))
    for i in range(k):
        hess[i,i] -= trigamma(alpha[i])
    return -hess


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
    alpha = 1./D*np.ones(D)
    eta = 1./K*np.ones((D,K))
    gamma = alpha
    phi = 1./D * np.ones((N,D))
    tau = 1./K * np.ones(K)

    loglik_old = [0]
    loglik = 1
    iter=1
    # Iteration until convergence
    while abs(loglik-loglik_old[-1]) > thres:
        loglik_old.append(loglik)
        iter+=1
        print "\n Iteration: "+str(iter)+" loglik "+str(loglik)
        print tau

        # For each document, estimate local latent variables
        for n in range(N):
            for d in range(D):
                phi[n,d]=np.exp(scipy.special.digamma(gamma[d])-scipy.special.digamma(np.sum(gamma)))
                for j in range(K):
                    phi[n,d] *= np.exp(sender[n,j]*(scipy.special.digamma(eta[d,j])-scipy.special.digamma(np.sum(eta[d,:]))))
                    phi[n,d] *= np.exp(receiver[n,j]*(scipy.special.digamma(eta[d,j])-scipy.special.digamma(np.sum(eta[d,:]))))
                # Normalize phi
            phi[n,:] /= np.sum(phi[n,:])
        print 'phi'
        print phi
            # Update eta
        for d in range(D):
            for i in range(K):
                eta[d,i] = tau[i]+np.dot(phi[:,d],sender[:,i]+receiver[:,i])
            # Normalize eta
            eta[d,i] /= np.sum(eta[d,:])
        # update gamma
        for i in range(D):
            gamma[i]=alpha[i]+np.sum(phi[:,i])
        print "gamma"
        print gamma

        # Update global parameters(tau, alpha)
        print 'Updating tau:'
        #tau = newton_raphson(tau,np.transpose(eta),D,.1,thres)
        res = scipy.optimize.minimize(functau, tau, method="Newton-CG", jac=dertau,hess=hesstau, args=(D,eta))
        tau = res.x
        print tau
        print 'Updating alpha'
        #alpha = newton_raphson(alpha,np.reshape(gamma,(D,1)),1,1,thres)
        res = scipy.optimize.minimize(funcalpha, alpha, method="Newton-CG", jac=deralpha,hess=hessalpha, args=(gamma))
        alpha = res.x

        print alpha

        # check convergence (likelihood)
        loglik = getloglik(alpha, tau, gamma, eta, phi, sender, receiver)

    return (alpha, gamma, eta, phi, tau, loglik_old)

if __name__=='__main__':
    print "nothing done"