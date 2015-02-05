import numpy as np
import numpy.random.mtrand
import scipy.special
import math
import prior_sample

def trigamma(x) :
    return ctrigamma(x).real

def ctrigamma(z):
    z = complex(z)
    g = 607./128.
    c = (
        0.99999999999999709182,
        57.156235665862923517,
        -59.597960355475491248,
        14.136097974741747174,
        -0.49191381609762019978,
        .33994649984811888699e-4,
        .46523628927048575665e-4,
        -.98374475304879564677e-4,
        .15808870322491248884e-3,
        -.21026444172410488319e-3,
        .21743961811521264320e-3,
        -.16431810653676389022e-3,
        .84418223983852743293e-4,
        -.26190838401581408670e-4,
        .36899182659531622704e-5)
    t1=0.
    t2=0.
    t3=0.
    for k in range(len(c)-1,0,-1):
        dz =1./(z+k);
        dd1 = c[k]* dz
        t1 += dd1
        dd2 = dd1 * dz
        t2 += dd2
        t3 += dd2 * dz

    t1 += c[0]
    c =  - (t2*t2)/(t1*t1)  +2*t3/t1

    result = 1./(z*z)
    gg = z + g + 0.5
    result += - (z+0.5)/ (gg*gg)
    result += 2./gg
    result += c

    return result

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
        print 'NR for tau:'
        tau = newton_raphson(tau,np.transpose(eta),D,.1,thres)
        print tau
        print 'NR for alpha'
        alpha = newton_raphson(alpha,np.reshape(gamma,(D,1)),1,1,thres)
        print alpha

        # check convergence (likelihood)
        loglik = getloglik(alpha, tau, gamma, eta, phi, sender, receiver)

    return (alpha, gamma, eta, phi, tau, loglik_old)

if __name__=='__main__':
    print "nothing done"