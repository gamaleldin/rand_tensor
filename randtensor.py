import matplotlib
import numpy as np
from numpy import random as rng
from scipy import linalg as sciLA
from utils import *


class randtensor(object):
  '''
  tensor class that takes an input, which is the size of different tensor modes

  '''
  def __init__(self, sizes):
      self.sizes = sizes;
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
      Lagrangians, f = self.optMaxEntropy(margEigValues, maxIter, figFlg);
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
          vecTensors[:, i] = rng.randn(np.prod(self.sizes))
          vecTensors[:, i] = vecTensors[:, i]*np.sqrt(self.vecEigValues)
      vecTensors = np.real(kron_mvprod(self.margEigVectors, vecTensors))
      tensors = []
      for i in range(nsamples):
          tensors += [np.reshape(vecTensors[:,i], tuple(self.sizes), order = 'F')]
      return tensors

  '''
  Compute the sampling error, defined as the square difference between the
  class marginal covariances and the empirical estimated from a list of tensors
  specified as an input
  '''
  def samplingError(self, tensors):
      nsamples = len(tensors);
      estMargCov = empiricalMargCovs(tensors);
      error = [];
      for i in range(self.nmodes):
          error += [(sciLA.norm(estMargCov[i]-self.margCovs[i], 'fro')/sciLA.norm(self.margCovs[i], 'fro'))**2*100];
          print "Error in estimated marginal covariance of mode %d, empirically estimated from samples, is %.2f %%" \
          %(i, error[i])
      return error



  def logObjectiveMaxEntropyTensor(self, logL, params):
      Lagrangians = []
      for x in params.tensorIxs:
          Lagrangians.append(np.exp(logL[np.add(sum(params.sizes[0:x]),range(int(params.sizes[x])))]))
      ############ building blocks of the gradient ###########
      LsTensor = diagKronSum(Lagrangians)
      LsTensor = np.reshape(LsTensor, tuple(params.sizes), order='F')                                 # kronecker sum of the lagrangian matrices eigenvalues
      invLsTensor = 1.0/LsTensor                                            # elementwise inverse of the above
      invSquareLsTensor = 1.0/LsTensor**2                                     # elementwise inverse square of the above
      Er = []
      logSums = []
      fx = []                                             # the log transformed cost decomposed to different tensor dimensions
      gradf_logL = np.zeros(len(logL))
      for x in params.tensorIxs:
          nxSet = list(set(params.tensorIxs).difference(set([x])))
          logSums.append(np.log(sumTensor(invLsTensor, nxSet)))                    # elementwise log of invLsTensor)
          Er.append(np.reshape(params.logEigValues[x], logSums[x].shape, order='F')- logSums[x])     # error with respect to each marginal covariance eigenvalue


          fx.append(np.reshape(Er[x], tuple([params.sizes[x], 1]), order='F')**2)
      f = sum(np.vstack(fx))/params.normalizeTerm

      ############ build the gradient from the blocks ###########
      for x in params.tensorIxs:
          nxSet = list(set(params.tensorIxs).difference(set([x])))
          gradfx_logLx = np.reshape(((2*Er[x]/sumTensor(invLsTensor, nxSet))*sumTensor(invSquareLsTensor, nxSet)),tuple([params.sizes[x],1]), order='F')*np.reshape(Lagrangians[x],     tuple([params.sizes[x],1]), order='F')
          gradf_logLx = gradfx_logLx

          for y in nxSet:
              nySet = list(set(params.tensorIxs).difference(set([y])))
              nxySet = list(set(params.tensorIxs).difference(set([x, y])))
              gradfy_logLx = np.reshape(sumTensor((2*Er[y]/sumTensor(invLsTensor, nySet))*sumTensor(invSquareLsTensor, nxySet), [y]), tuple([params.sizes[x] ,1]), order='F')*np.reshape(Lagrangians[x], tuple([params.sizes[x] ,1]), order='F')
              gradf_logLx = gradf_logLx+gradfy_logLx



          gradf_logL[np.add(sum(params.sizes[0:x]),range(int(params.sizes[x])))] = np.squeeze(gradf_logLx)

      gradf_logL = gradf_logL/params.normalizeTerm
      ############ return ###########
      return f, gradf_logL




  def optMaxEntropy(self, eigValues, maxIter, figFlg):
  #  if the marginal covariances are low rank then the number of variables
  #  that we solve for are less. If full rank the number of variables that we
  #  solve for are equal to the sum of the tensor dimensions.
  ############# define optimization params class #####################
      class optParams:
          logEigValues = []
          nmodes = []
          tensorIxs = []
          sizes = []
          normalizeTerm = []
          logL = []
  ########################## initializations #########################
      params = optParams()
  #     figFlg = True          # display summary figure flag
      params.nmodes = len(eigValues);              # tensor size; i.e. the number of different dimensions of the tensor
      sizes = np.zeros(params.nmodes)              # tensor dimensions
      params.tensorIxs = range(params.nmodes)
      threshold = -10                                     # if an eigenvalue is below this threshold it is considered 0.
      for x in params.tensorIxs:
          sizes[x] = len(eigValues[x])

  # instead of solving for the largrangians directly we optimize latent variables that is equal to the log of the lagrangians

      preScale = sum(eigValues[0])/np.mean(sizes)         # prescale the eigenvalues for numerical stability
      params.logEigValues = []                            # the log of the eigenvalues
      params.sizes =  []                                  # true number of variables that we solve for, which is equal to the sum of the ranks of the marginal covariances
      for x in params.tensorIxs:
          params.logEigValues.append(np.array([i for i in np.log(eigValues[x]/preScale) if i>threshold])) # eigenvalues should be order apriori
          params.sizes.append(len(params.logEigValues[x]))
      params.sizes = np.array(params.sizes)
      params.normalizeTerm = sum(np.hstack(params.logEigValues)**2)

  ####################### optimization step #############################
  # initialization of the optimization variables
      logL0 = []
      for x in params.tensorIxs:
          nxSet = map(int, set(params.tensorIxs).difference(set([x])))
          logL0.append(np.log(sum(params.sizes[nxSet]))-params.logEigValues[x])

  #     maxIter = 1000;        # maximum allowed iterations
      logL0 = np.array(np.hstack(logL0)).T
      logL, f, i = minimize(logL0 ,self.logObjectiveMaxEntropyTensor , maxIter, params); # this function performs all the optimzation heavy lifting
      L = np.exp(logL)
      params.L = L

  ##################### convert solution from the lagrangian variables to the optimal eigenvalues of the big covariance matrix
      Lagrangians = []                                      # save the lagrangians to the output
      for x in params.tensorIxs:
          Lagrangians.append(np.hstack([L[np.add(sum(params.sizes[0:x]),range(int(params.sizes[x]))).astype(int)]/preScale, float('inf')*np.ones(int(sizes[x]-params.sizes[x]))]))

      return Lagrangians, f[-1]



