# MALAX
Association testing of bisulfite sequencing methylation data via a Laplace approximation

MALAX (Mixed model Association via Laplace ApproXimation) is a Python package for association testing of bisulfite sequencing methylation data. It models the phenotype as an explanatory variable and each tested site as a reponse variable, using a binomial generalized linear mixed model (GLMM). The likelihood is approximated via a Laplace approximation.

MALAX was published in: [Association testing of bisulfite-sequencing methylation data via a Laplace approximation. Bioinformatics 33.14,  i325-i332 (2017)](https://academic.oup.com/bioinformatics/article/33/14/i325/3953963/Association-testing-of-bisulfite-sequencing).

Several parts of the code are loosely based on code translated from the [GPML toolbox](http://www.gaussianprocess.org/gpml/code/matlab/).


<br><br>
# Installation
MALAX is designed to work in Python 2.7, and depends on the following freely available Python packages:
* [numpy](http://www.numpy.org/) and [scipy](http://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [cython](http://cython.org/)

Typically, the packages can be installed with the command "pip install --user \<package_name\>".

MALAX is particularly easy to use with the [Anaconda Python distribution](https://store.continuum.io/cshop/anaconda). The [numerically optimized version](http://docs.continuum.io/mkl-optimizations/index) of Anaconda can speed MALAX significantly.
Alternatively (if numerically optimized Anaconda can't be installed), for very fast performance it is recommended to have an optimized version of Numpy/Scipy [installed on your system](http://www.scipy.org/scipylib/building), using optimized numerical libraries such as [OpenBLAS](http://www.openblas.net) or [Intel MKL](https://software.intel.com/en-us/intel-mkl) (see [Compilation instructions for scipy with Intel MKL)](https://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl).

Once all the prerequisite packages are installed, MALAX can be installed on a git-enabled machine by typing:
```
git clone https://github.com/omerwe/MALAX
```

The project can also be downloaded as a zip file from the Github website.

After downloading the code, please compile the code by going to the MALAX directory and typing:
```
python setup.py build_ext --inplace
```
To verify that the compilation succeeded, please verify that the file `laplace_cython.so` was created in the directory.

<br><br>
# Usage Overview

MALAX can be invoked via the script `run_laplace.py`.The list of available options can be seen by typing `run_laplace.py --help`.

## TL;DR
For an example, please run the following command (using the anaconda version of python if available):
```
python run_laplace.py --mcounts example/y.txt --counts example/r.txt --predictor example/pred.txt --kernel example/K.txt --kernel2 example/cell_types_K.txt --covar example/covars.txt --out example/malax_2K.txt
```
This will analyze the data in the `example` directory using two variance components and will print the results to the file `example/malax_2K.txt`.


<br><br>
# Detailed Instructions

MALAX takes as input a file with number of reads (`r.txt` in the example directory), a file with with number of methylated reads (`y.txt`), a file with a predictor (`pred.txt`), a file with covariates (`covars.txt`), and one or two covariance matrices associated with random effects (`K.txt` and `cell_types_K.txt`). The corresponding flags can be seen in the example above. The code will print a file with a p-value for every tested site.

Additionally, the code supports a fixed effects beta-binomial model, which can be invoked by adding the flag `--test bb` to the example command above. This code will ignore the `--kernel` and `--kernel2` commands. 

The format of the files can be seen in the example directory. It is the same format as used by [MACAU](http://www.xzlab.org/software.html).

For questions and comments, please contact Omer Weissbrod at oweissbrod[at]hsph.harvard.edu


