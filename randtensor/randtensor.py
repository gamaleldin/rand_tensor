from itertools import chain
import numpy as np
from numpy import random as nprnd
from scipy import linalg as sciLA



class randtensor:
    '''
    tensor class that takes an input, which is the size of different tensor modes
    
    '''
    def __init__(self, sizes):
        self.sizes = sizes;
        self.marginalCov = [];
        self.nmodes = len(sizes);
        self.margCovs = [];                            # marginal Covariances of the distribution
        self.margEigVectors = [];                        # eigenvectors of the marginal Covariances of the distribution
        self.margEigValues = [];
        for i in range(self.nmodes):
            self.margCovs += [np.eye(sizes[i])/sizes[i]];
            self.margEigVectors += [np.eye(sizes[i])];
            self.margEigValues += [np.eye(sizes[i])];
        self.vecEigValues = np.ones(np.prod(self.sizes)); # eigenvectors of the vectorized distribution
    
    ###### function to check if a matrix is proper covariance matrix (i.e., symmetric posistive semidef)
    def isproperCov(self, Sigma):  ### gamal
        isCov = True
        SigmaSPD= (Sigma+Sigma.T)/2  # make sure it is symmetric pos def. matrix (i.e., proper covariance matrix)
        if not isCov:
            print ('The covariance matrix is not correct, approximating to the nearest possible covariance')
        return SigmaSPD;
    
    ###### function to compute the eigenvalues and eigenvectors of the marginal covariances
    def margEig(self, margCovs):
        margEigValues = [];
        margEigVectors = [];
        for i in range(len(margCovs)):
            Sigma_i = margCovs[i];
            Sigma_i = self.isproperCov(Sigma_i);
            Q, s, _ = sciLA.svd(Sigma_i)
            ix = np.argsort(s)[::-1]
            s = s[ix]
            Q = Q[:, ix]
            margEigValues += [s];
            margEigVectors += [Q];
        return margEigValues, margEigVectors
    '''
    class function that fits maximum entropy distribution with covariances specified in margCovs list
    '''
    def fitMaxEntropy(self, margCovs):
        margEigValues, margEigVectors = self.margEig(margCovs);
        maxIter = 1000;
        figFlg = False;
        Lagrangians, f = optMaxEntropy(margEigValues, maxIter, figFlg);
        if f>1e-10:
            print ("algorithm did not completely converged. error = %.10f \n results may be inaccurate" %f)
        self.vecEigValues = 1/diagKronSum(Lagrangians);
        self.margCovs = margCovs;
        self.margEigValues = margEigValues;
        self.margEigVectors = margEigVectors;
    
    '''
    class function that samples tensors (number of tensors is specified as an input) 
    with the marginal covariance constraints.
    '''
    def sampleTensors(self, nsamples):
        vecTensors = np.zeros([np.prod(self.sizes), nsamples]);
        for i in range(nsamples):
            vecTensors[:, i] = nprnd.randn(np.prod(self.sizes))
            vecTensors[:, i] = vecTensors[:, i]*np.sqrt(self.vecEigValues)
        vecTensors = np.real(kron_mvprod(self.margEigVectors, vecTensors))
        tensors = []
        for i in range(nsamples):
            tensors += [np.reshape(vecTensors[:,i], tuple(self.sizes), order = 'F')]
        
        _ = self.samplingError(tensors)
        return tensors
    
    '''
    Compute the marginal covariances empirically a list of tensors specified as an input
    '''
    def empiricalMargCovs(self, tensors):
        nsamples = len(tensors)
        M =  self.empiricalMean(tensors)
        sizes = np.array(tensors[0].shape)
        nmodes = len(sizes)
        allDim = range(nmodes)
        estMargCov = []
        for i in allDim:
            estMargCov.append(np.zeros([sizes[i], sizes[i]]))

        for j in range(nsamples):
            tensor = tensors[j]-M
            marginalCovs = []
            for i in allDim:
                niSet = list(set(allDim) - set([i]))
                z = np.reshape(np.transpose(tensor, list(chain.from_iterable([[i], niSet]))), tuple([sizes[i], np.prod(sizes[niSet])]), order = 'F')
                estMargCov[i] = (estMargCov[i]+(np.dot(z,z.T)/(nsamples-1)))
        for i in allDim:
            estMargCov[i] = (estMargCov[i])
        return estMargCov
    

    '''
    Compute the mean tensor from a list of tensors specified as an input
    '''
    def empiricalMean(self, tensors):
        nsamples = len(tensors);
        M = tensors[0]/nsamples
        for i in range(nsamples-1):
            M = M+tensors[i+1]/nsamples
        return M
    
    '''
    Compute the sampling error, defined as the square difference between the 
    class marginal covariances and the empirical estimated from a list of tensors 
    specified as an input
    '''
    def samplingError(self, tensors):
        nsamples = len(tensors);
        estMargCov = self.empiricalMargCovs(tensors);
        error = [];
        for i in range(self.nmodes):
            error += [(sciLA.norm(estMargCov[i]-self.margCovs[i], 'fro')/sciLA.norm(self.margCovs[i], 'fro'))**2*100];
            print "Error in estimated marginal covariance of mode %d, empirically estimated from samples, is %.2f %%" \
            %(i, error[i])
