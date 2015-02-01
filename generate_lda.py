import numpy as np
import lda_bowei
import matplotlib.pyplot as plt
# Generate a synthetic dataset for LDA test

M=100
N=50
V=5
K=2
thres=0.1
alpha=np.random.rand(K)
beta=np.random.rand(K,V)
for i in range(K):
    sum_beta = np.sum(beta[i,:])
    for j in range(V):
        beta[i,j] = beta[i,j]/sum_beta

theta=np.zeros((M,K))
z=np.zeros((M,N))# takes values in {1,...,K}
w=np.zeros((M,N))

def getmultisamp(pval):
    K=pval.shape[0]
    r=np.random.multinomial(1,pval)
    for i in range(K):
        if r[i]!=0:
            return i


for d in range(M):
    theta[d,:]=np.random.dirichlet(alpha)
    for n in range(N):
        z[d,n] = getmultisamp(theta[d,:])
        w[d,n] = getmultisamp(beta[z[d,n],:])

if __name__=='__main__':
    print "params: alpha beta theta z"
    print alpha
    print beta
    print theta
    print z

    print "words"
    print w

    (alpha, beta, gamma, phi,loglik)=lda_bowei.lda(w,K,V,thres)

    niter=len(loglik)
    plt.plot(np.arange(niter),loglik)
    plt.show()
    print "estimations: alpha beta gamma phi"
    print alpha
    print beta
    print gamma
    print phi