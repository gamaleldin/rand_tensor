import numpy as np
from itertools import chain

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

def diagKronSum(Ds):
  nmodes = len(Ds)
  kronSumDs = np.zeros(1)
  for i in range(nmodes):
    kronSumDs = np.add(np.outer(kronSumDs, np.ones(len(Ds[i]))),np.outer(np.ones(len(kronSumDs)),Ds[i]))
    kronSumDs = np.hstack(kronSumDs.T)
  return kronSumDs

def sumTensor(T, sumDim):
  sumT = T
  for i in range(len(sumDim)):
    sumT = np.expand_dims(np.sum(sumT, sumDim[i]), axis = sumDim[i])
  return sumT

def first_moment(tensor):
  sizes = np.array(tensor.shape)
  nmodes = len(sizes)
  tensorIxs = range(nmodes)
  tensor0 = tensor
  for x in tensorIxs:
    nxSet = list(set(tensorIxs).difference(set([x])))
    mu = sumTensor(tensor0, nxSet)/np.prod(sizes[nxSet])
    tensor0 = tensor0 - mu
  return tensor-tensor0

def second_moments(tensor):
  sizes = np.array(tensor.shape)
  nmodes = len(sizes)
  tensorIxs = range(nmodes)
  sec_moments = []
  for x in tensorIxs:
    nxSet = list(set(tensorIxs) - set([x]))
    z = np.reshape(np.transpose(tensor, list(chain.from_iterable([[x], nxSet]))), tuple([sizes[x], np.prod(sizes[nxSet])]), order = 'F')
    sec_moments.append(np.dot(z,z.T))
  return sec_moments


def empiricalMean(tensors):
  """ Compute the mean tensor from a list of tensors specified as an input
  """
  nsamples = len(tensors);
  M = tensors[0]/nsamples
  for i in range(nsamples-1):
    M = M+tensors[i+1]/nsamples
  return M


def empiricalMargCovs(tensors):
  """Compute the marginal covariances empirically a list of tensors specified as an input
  the code gives 2nd order statistic if n_samples = 1
  """
  nsamples = len(tensors)
  M =  empiricalMean(tensors)
  sizes = np.array(tensors[0].shape)
  nmodes = len(sizes)
  allDim = range(nmodes)
  estMargCov = []
  for i in allDim:
    estMargCov.append(np.zeros([sizes[i], sizes[i]]))
  for j in range(nsamples):
    tensor = tensors[j]-M
    for i in allDim:
      niSet = list(set(allDim) - set([i]))
      z = np.reshape(np.transpose(tensor, list(chain.from_iterable([[i], niSet]))), tuple([sizes[i], np.prod(sizes[niSet])]), order = 'F')
      estMargCov[i] = (estMargCov[i]+(np.dot(z,z.T)/max(1., nsamples-1.)))
  for i in allDim:
    estMargCov[i] = (estMargCov[i])
  return estMargCov



def minimize(X, obj_Fn, length, params):

  supressOutPut = True;
  INT = 0.1;    # don't reevaluate within 0.1 of the limit of the current bracket
  EXT = 3.0;    # extrapolate maximum 3 times the current step-size
  MAX = 20;     # max 20 function evaluations per line search
  RATIO = 10.0;   # maximum allowed slope ratio
  SIG = 0.1;
  RHO = SIG/2.0; # SIG and RHO are the constants controlling the Wolfe-
  # Powell conditions. SIG is the maximum allowed absolute ratio between
  # previous and new slopes (derivatives in the search direction), thus setting
  # SIG to low (positive) values forces higher precision in the line-searches.
  # RHO is the minimum allowed fraction of the expected (from the slope at the
  # initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
  # Tuning of SIG (depending on the nature of the function to be optimized) may
  # speed up the minimization; it is probably not worth playing much with RHO.
  realmin = np.finfo(np.double).tiny

  def unwrap(s):
    l = s.shape
    v = np.reshape(s, np.prod(l), order = 'F')
    return v

  def rewrap(s, v):
    l = s.shape;
    v = np.reshape(v, l, order = 'F')
    return v


  S='Linesearch'


  i = 0                                            # zero the run length counter
  ls_failed = False                             # no previous line search has failed
  f0, df0 = obj_Fn(X, params)    # get function value and gradient
  Z = X;X = unwrap(X); df0 = unwrap(df0);
  if not supressOutPut:
      print "%s %6i;  Value %4.6e\r" %(S, i, f0)
  fX = []
  fX.append(f0);

  s = -df0;
  d0 = -np.dot(s.T,s)           # initial search direction (steepest) and slope
  x3 = 1.0/(1.0-d0)

  while i < length:              # while not finished
    i = i + 1                   # count iterations
    X0 = X; F0 = f0; dF0 = df0; # make a copy of current values
    M = MAX;


    while True:   # keep extrapolating as long as necessary
      x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0;
      success = False;

      while (not success) and (M > 0):
        try:
            M = M - 1.0; i = i + 1;          # count epochs
            f3, df3 = obj_Fn(rewrap(Z, X+x3*s), params);
            df3 = unwrap(df3);
            if np.isnan(f3) or np.isinf(f3) or any(np.isnan(df3)+np.isinf(df3)):
                print "error(' ')"
            success = True;
        except:         # catch any error which occured in f
            x3 = (x2+x3)/2;  # bisect and try again


      if f3 < F0:
        X0 = X+x3*s; F0 = f3; dF0 = df3;       # keep best values
      d3 = np.dot(df3.T,s);                     # new slope


      if (d3 > (SIG*d0)) or (f3 > (f0+x3*RHO*d0)) or (M == 0):  # are we done extrapolating?
        break;


      x1 = x2; f1 = f2; d1 = d2;                      # move point 2 to point 1
      x2 = x3; f2 = f3; d2 = d3;                      # move point 3 to point 2
      A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);                # make cubic extrapolation
      B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);

      x3 = x1-d1*(x2-x1)**2/(B+np.sqrt(abs(B*B-A*d1*(x2-x1)))); # num. error possible, ok!
      if (not np.isreal(x3)) or np.isnan(x3) or np.isinf(x3) or (x3 < 0.): # num prob | wrong sign?
        x3 = x2*EXT;                    # extrapolate maximum amount
      elif x3 > x2*EXT:                  # new point beyond extrapolation limit?
        x3 = x2*EXT;                    # extrapolate maximum amount
      elif x3 < (x2+INT*(x2-x1)):         # new point too close to previous point?
        x3 = x2+INT*(x2-x1);
                             ### end extrapolation ####

    while (abs(d3) > -SIG*d0 or (f3 > (f0+x3*RHO*d0))) and (M > 0):  # keep interpolating
      if d3 > 0 or f3 > f0+x3*RHO*d0:             # choose subinterval
        x4 = x3; f4 = f3; d4 = d3;               # move point 3 to point 4
      else:
        x2 = x3; f2 = f3; d2 = d3;               # move point 3 to point 2

      if f4 > f0:
        x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2)); # quadratic interpolation
      else:
        A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);          # cubic interpolation
        B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
        x3 = x2+(np.sqrt(B*B-A*d2*(x4-x2)**2)-B)/A;   # num. error possible, ok!

      if np.isnan(x3) or np.isinf(x3):
        x3 = (x2+x4)/2;       # if we had a numerical problem then bisect


      x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));  # don't accept too close
      f3, df3 = obj_Fn(rewrap(Z, X+x3*s), params);
      df3 = unwrap(df3);

      if f3 < F0:
        X0 = X+x3*s; F0 = f3; dF0 = df3;    # keep best values
      M = M - 1.0; i = i + 1;               # count epochs
      d3 = np.dot(df3.T,s);               # new slope
                         #### end interpolation ####

    if (abs(d3) < (-SIG*d0)) and (f3 < (f0+x3*RHO*d0)): # if line search succeeded
      X = X+x3*s; f0 = f3; fX.append(f0);              # update variables
      if not supressOutPut:
        print "%s %6i;  Value %4.6e\r" %(S, i, f0);
      s = np.dot((np.dot(df3.T, df3)-np.dot(df0.T,df3))/(np.dot(df0.T,df0)), s) - df3;  # Polack-Ribiere CG direction
      df0 = df3;                        # swap derivatives
      d3 = d0; d0 = np.dot(df0.T,s);
      if d0 > 0:                         # new slope must be negative
        s = -df0; d0 = -np.dot(s.T,s); # otherwise use steepest direction

      x3 = x3 * min(RATIO, d3/(d0-realmin));  # slope ratio but max RATIO
      ls_failed = False;                      # this line search did not fail
    else:
      X = X0; f0 = F0; df0 = dF0;            # restore best point so far
      if ls_failed or (i > abs(length)):      # line search failed twice in a row
        break;                             # or we ran out of time, so we give up

      s = -df0; d0 = -np.dot(s.T,s);           # try steepest
      x3 = 1.0/(1.0-d0);
      ls_failed = True;                        # this line search failed
  X = rewrap(Z,X);
  return X, fX, i



