import numpy as np

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% This function evaluates the sum of tensor at specific dimensions
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Inputs:
#%       - A: is the input n-dimesnional tensor
#%       - sumDim: the dimensions to sum over.
#% Outputs:
#%       - sumA: an n-dimensional tensor of the sum of tensor A at the
#%       specified dimensions. The dimensions specified by sumDim will be of
#%       size 1.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def kron_mvprod(As, b):
    x = b
    numDraws = b[0, :].size
    CTN = b[:, 0].size
    for d in range(len(As)):
        A = As[d]
        Gd = A[:,0].size
        X = np.reshape(x, tuple([Gd, CTN*numDraws/Gd]), order = 'F')
        Z = np.dot(A, X).T
        x = np.reshape(Z, tuple([CTN,numDraws]), order = 'F')
    x = np.reshape(x, tuple([CTN*numDraws, 1]), order = 'F')
    x = np.reshape(x, tuple([numDraws, CTN]), order = 'F').T
    return x


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% This function evaluates the sum of tensor at specific dimensions
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Inputs:
#%       - A: is the input n-dimesnional tensor
#%       - sumDim: the dimensions to sum over.
#% Outputs:
#%       - sumA: an n-dimensional tensor of the sum of tensor A at the
#%       specified dimensions. The dimensions specified by sumDim will be of
#%       size 1.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def diagKronSum(Ds):
    nmodes = len(Ds)
    kronSumDs = np.zeros(1)
    for i in range(nmodes):
        kronSumDs = np.add(np.outer(kronSumDs, np.ones(len(Ds[i]))),np.outer(np.ones(len(kronSumDs)),Ds[i]))
        kronSumDs = np.hstack(kronSumDs.T)
    return kronSumDs

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% This function evaluates the sum of tensor at specific dimensions
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Inputs:
#%       - A: is the input n-dimesnional tensor
#%       - sumDim: the dimensions to sum over.
#% Outputs:
#%       - sumA: an n-dimensional tensor of the sum of tensor A at the
#%       specified dimensions. The dimensions specified by sumDim will be of
#%       size 1.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def sumTensor(T, sumDim):
    sumT = T
    for i in range(len(sumDim)):
        sumT = np.expand_dims(np.sum(sumT, sumDim[i]), axis = sumDim[i])
    return sumT




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% This function evaluates the sum of tensor at specific dimensions
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Inputs:
#%       - A: is the input n-dimesnional tensor
#%       - sumDim: the dimensions to sum over.
#% Outputs:
#%       - sumA: an n-dimensional tensor of the sum of tensor A at the
#%       specified dimensions. The dimensions specified by sumDim will be of
#%       size 1.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def logObjectiveMaxEntropyTensor(logL, params):
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
        gradfx_logLx = np.reshape(((2*Er[x]/sumTensor(invLsTensor, nxSet))*sumTensor(invSquareLsTensor, nxSet)),tuple([params.sizes[x],1]), order='F')*np.reshape(Lagrangians[x], tuple([params.sizes[x],1]), order='F')
        gradf_logLx = gradfx_logLx
        for y in nxSet:
            nySet = list(set(params.tensorIxs).difference(set([y])))
            nxySet = list(set(params.tensorIxs).difference(set([x, y])))
            gradfy_logLx = np.reshape(sumTensor((2*Er[y]/sumTensor(invLsTensor, nySet))*sumTensor(invSquareLsTensor, nxySet), [y]), tuple([params.sizes[x] ,1]), order='F')*np.reshape(Lagrangians[x], tuple([params.sizes[x] ,1]), order='F')
            gradf_logLx = gradf_logLx+gradfy_logLx

        gradf_logL[np.add(sum(params.sizes[0:x]),range(int(params.sizes[x])))] = gradf_logLx
    
    gradf_logL = gradf_logL/params.normalizeTerm  
    ############ return ###########
    return f, gradf_logL

    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% This function evaluates the sum of tensor at specific dimensions
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Inputs:
#%       - A: is the input n-dimesnional tensor
#%       - sumDim: the dimensions to sum over.
#% Outputs:
#%       - sumA: an n-dimensional tensor of the sum of tensor A at the
#%       specified dimensions. The dimensions specified by sumDim will be of
#%       size 1.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
def optMaxEntropy(eigValues, maxIter, figFlg):
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
    logL, f, i = minimize(logL0 ,logObjectiveMaxEntropyTensor , maxIter, params); # this function performs all the optimzation heavy lifting
    L = np.exp(logL)
    params.L = L
    
    if figFlg:
        import matplotlib.pyplot as plt
        plt.plot(range(len(f)), f)
        plt.ylabel('objective fn')
        plt.show()
##################### convert solution from the lagrangian variables to the optimal eigenvalues of the big covariance matrix
    Lagrangians = []                                      # save the lagrangians to the output 
    for x in params.tensorIxs: 
        Lagrangians.append(np.hstack([L[np.add(sum(params.sizes[0:x]),range(int(params.sizes[x]))).astype(int)]/preScale, float('inf')*np.ones(sizes[x]-params.sizes[x])]))
    
    return Lagrangians, f[-1]
  

