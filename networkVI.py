import numpy as np
import numpy.random.mtrand
import scipy.special
import math

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

def newton_raphson(alpha,gamma,w,K,stepsize,thres):
    M = w.shape[0]
    N = w.shape[1]
    g = np.zeros(K)
    h = np.zeros(K)
    alpha_old = alpha
    alpha_new = alpha
    diff = 10
    while diff > thres:
        alpha_old = alpha_new
        z = M*trigamma(np.sum(alpha))
        for i in range(K):
            h[i] = -M*trigamma(alpha[i])
            g[i] = M*scipy.special.digamma(np.sum(alpha))-M*scipy.special.digamma(alpha[i])
            for d in range(M):
                g[i]+=scipy.special.digamma(gamma[i,d])-scipy.special.digamma(np.sum(gamma[:,d]))

        c = np.sum(g/h)/(1/z+np.sum(1/h))
        for i in range(K):
            alpha_new[i] = alpha_old[i]-stepsize*(g[i]-c)/h[i]
        diff = np.linalg.norm(alpha_new-alpha_old)

    return alpha_new

def getloglik(alpha, beta, gamma, phi, w, V):
    M = w.shape[0]
    N = w.shape[1]
    K = alpha.shape[0]
    loglik = 0
    for d in range(M):
        loglik += math.log(scipy.special.gamma(np.sum(alpha)))
        for i in range(K):
            loglik += (alpha[i]-1)*(scipy.special.digamma(gamma[i,d])-scipy.special.digamma(np.sum(gamma[:,d])))-math.log(scipy.special.gamma(alpha[i]))
            for n in range(N):
                loglik += phi[n,i,d]*(scipy.special.digamma(gamma[i,d])-scipy.special.digamma(np.sum(gamma[:,d])))
                for j in range(V):
                    loglik += phi[n,i,d]*(w[d,n]==j)*beta[i,j]
        loglik += -math.log(scipy.special.gamma(np.sum(gamma[:,d])))
        for i in range(K):
            loglik -= (gamma[i,d]-1)*(scipy.special.digamma(gamma[i,d])-scipy.special.digamma(np.sum(gamma[:,d])))-math.log(scipy.special.gamma(gamma[i,d]))
            for n in range(N):
                loglik -= phi[n,i,d]*math.log(phi[n,i,d])

    return loglik

def lda(w,K,V,thres):
    # w: word data, numpy array with size M*N; M is No. of documents and N is No. of words in each documents
    # K: No. of topics
    # V: Vocabulary

    M = w.shape[0]
    N = w.shape[1]

    # Initialization for alpha, beta, gamma, phi
    alpha = 1./K*np.ones(K)
    beta = 1./V*np.ones((K,V))
    gamma = 1./K * np.ones((K,M))
    phi = 1./K * np.ones((N,K,M))

    loglik_old = [0]
    loglik = 1
    iter=1
    # Iteration until convergence
    while abs(loglik-loglik_old[-1]) > thres:
        loglik_old.append(loglik)
        iter+=1
        if iter%10==0:
            print "Iteration: "+str(iter)+"loglik"+str(loglik)

        # For each document, estimate local latent variables
        for d in range(M):
            for n in range(N):
                for i in range(K):
                    phi[n,i]=beta[i,w[d,n]]*np.exp(scipy.special.digamma(gamma[i]))
                # Normalize phi
                sum_phi=np.sum(phi[n,:])
                for i in range(K):
                    phi[n,i]=phi[n,i]/sum_phi
                gamma[i,d]=np.add(alpha[i],np.sum(phi[n,:,d]))
        # Update global parameters

        alpha = newton_raphson(alpha,gamma,w,K,1,thres)
        for i in range(K):
            for j in range(V):
                for d in range(M):
                    for n in range(N):
                        beta[i,j] += phi[n,i,d]*(w[d,n]==j)
            sum_beta = np.sum(beta[i,:])
            for j in range(V):
                beta[i,j] = beta[i,j]/sum_beta
        # check convergence (likelihood)
        loglik = getloglik(alpha, beta, gamma, phi, w, V)

    return (alpha, beta, gamma, phi, loglik_old)

if __name__=='__main__':
    print "nothing done"