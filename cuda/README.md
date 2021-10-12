PHPC - CONJUGATE GRADIENT PROJECT


THE CODE
========

There is 2 ways to compute the matrix-vector multiplication:
	
	- Using one thread per cross-product
	- Using p threads per cross-product (p is denoted n-reduce in the code)

- One thread per cross-product:

Kernel name: dgemv
Usage dgemv<<< , >>>(int m, int n, double alpha, Matrix A, double * x, double * y)
Returns y = alpha * A * x

In this kernel, the first 4096 elements of x are stored in shared memory before computing the cross-products,
and store them in y

- p threads per cross-product:

Here we use two kernels, the first one compute the partial cross-products and the second one sums the up.

The kernels are:
 
- First kernel: dgemv2, usage dgemv2<<< , >>>(int m, int n, double alpha, Matrix A, double * x, double * y, int width)
Computes the partial cross-products and store them in y

Arguments:
	- width    (int): The length of the arrays for the partial cross-products (might be reduced for the last column)

- Second Kernel: reduce(int n, int n-reduce, double * y)
Sums the values of y in each row, and store them in the first row

Argument:
	- n-reduce (int): the number of elements to sum, that is to say, the number of partial cross-products


- There is a last kernel:

Kernel name: daxpy, usage daxpy<<< , >>>(int n, double alpha, double * x, double * beta, double * y)
Computes y = alpha * x + beta * y


Note that the choice of kernel can be done during the call to the program.
Also note that width can not be > 6000, due to shared memory limitations


HOWTO COMPILE AND RUN
=====================

Requirements : 

- a recent compiler (like gcc or intel)
- a cblas library (like openblas or intel MKL)
- a cuda library (we use cuda)

compile on SCITAS clusters :

```
$ module load gcc openblas cuda
$ make
```

HOW TO RUN
==========

use:
	$ srun ./cgsolver [matrix.mtx] <block_size> <width= #number matrix rows>

Arguments:
	-matrix.mtx (str): filename of a matrix in mtx format
	-block_size (int): block_size of the thread blocks (in the row-axis)
	-width 	    (int): when defined, determines the length of the partial cross-products for matrix-vector multiplication
	
	Note that if width+block_size-1 > 6000 then the program will use 1 thread per cross-product (as the shared memory would be to small)



---------------------------------------------------



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

