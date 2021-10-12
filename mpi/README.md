PHPC - CONJUGATE GRADIENT PROJECT


THE CODE
========

The code is parallelized solely by parallelizing the function cblas-dgemv,
as 99.8% of the serial runtime was due to this function, the rest being mostly matrix reading,
for initialization.

Each processor computes a portion of the cross-products
Then all processors send and receive their computations, using MPI-Allgatherv().



HOWTO COMPILE AND RUN
=====================

Requirements : 

- a recent compiler (like gcc or intel)
- a cblas library (like openblas or intel MKL)

compile on SCITAS clusters :

```
$ module load gcc openblas
$ make
```

You should see this output (timing is indicative) :

```
$ srun ./conjugategradient lap2D_5pt_n100.mtx 
size of matrix = 10000 x 10000
Call cgsolver() on matrix size (10000 x 10000)
	[STEP 488] residual = 1.103472E-10
Time for CG = 36.269389 [s]
```

The given example is a 5-points stencil for the 2D Laplace problem. The matrix is in sparse format.

The matrix format is [Matrix Market format (.mtx extension)](https://sparse.tamu.edu/). You can use other matrices there or create your own. 

