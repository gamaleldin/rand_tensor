from setuptools import setup

setup(name='randtensor',
      version='0.1',
      description='This code package generate random tensors with user-specified marginal means and covariances from one of two optional distributions. The first is the maximum-entropy-distribution with the specified marginal means and covariances. The second is the tensor-normal-distribution with the specified means and covariances.',
      url='https://bitbucket.org/gamaleldin/randtensor',
      author='Gamaleldin F. Elsayed',
      author_email='gamaleldin.elsayed@gmail.com',
      license='General Public License',
      packages=['randtensor'],
      install_requires=[
                        'numpy',
                        'scipy',
                        'itertools',
                        ],
      zip_safe=False)