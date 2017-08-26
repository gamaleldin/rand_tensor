from randtensor import *
import matplotlib.image as mpimg
import numpy.random as rng
from scipy import linalg as sciLA
import numpy as np
import matplotlib.pyplot as plt


def imag_stats(imag):
  sizes = imag.shape
  nmodes=len(sizes)
  # specify covariance constraints
  # genarate marginal covariance matrices of size specified in dim
  mean = first_moment(imag)
  covs = second_moments(imag-mean)
  return mean, covs



def max_entropy_noise(imag_path):
  imag = mpimg.imread(imag_path)
  sizes = imag.shape
  mean, covs = imag_stats(imag)
  t = randtensor(sizes)   # create tensor class
  t.fitMaxEntropy(covs) #fit maximum entropy distribution with the known marginal covariances
  return t.sampleTensors(1)[0]+mean

def main():
  plt.imshow(max_entropy_noise("imag1.png"))
  plt.savefig("max_entropy_noise")


if __name__ == "__main__":
  main()
