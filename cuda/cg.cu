#include "cg.hh"
#include "matrix.hh"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

#define SHARED 4096

const double NEARZERO = 1.0e-14;
const bool DEBUG = false;

/*
    cgsolver solves the linear equation A*x = b where A is
    of size m x n

Code based on MATLAB code (from wikipedia ;-)  ):

function x = conjgrad(A, b, x)
    r = b - A * x;
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end
*/
//////////////////////////////////////////////////////////////////////////////////////////////
/*
I implemented 4 kernels:
  -dgemv: makes matrix multiplication y = alpha*A*x with 1 thread per cross-product, 
          puts SHARED values of x in shared memory

  -dgemv2: makes matrix multiplication y = alpha*A*x with n_reduce threads per cross-product, 
           each value is stores in y before reduction with reduce kernel
           puts width values of x in shared memory
  
  -reduce: reduces the value of y to finish the computations of dgemv2
  -daxpy: performs y = alpha*x + beta*y
*/
//////////////////////////////////////////////////////////////////////////////////////////////

__global__ void dgemv(int m, int n, double alpha, Matrix A, double * x, double * y) {
    /// Computes y = alpha*A*x
  
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    // Instantiate an array in shared memory too store x
    __shared__ double x_shar[SHARED+1023]; // +1023 to prevent bank conflicts (I think that's how it works)
    
    // Each thread in a block will put some data of x into shared memory

    // First look at columns to be put in x_shar by this thread
    int chunk;
    if (blockDim.x > SHARED) {chunk = 1;}
    else {chunk = SHARED/blockDim.x + (SHARED % blockDim.x == 0 ? 0 : 1);};
    int first_col = chunk*threadIdx.x;
    int last_col = first_col + chunk;
    if (last_col > SHARED) {last_col=SHARED;};

    // Put values of x in shared memory
    for (int col = first_col; col < last_col; col++) {
      x_shar[col] = x[col];
    }

    __syncthreads(); // Make sure it is done for all threads*/

    // stores partial cross-products in y, taking x in global memory only if needed
    double sum = 0;
    if (row < n) {
        for (int j = 0; j < m; j++) {
            if (j < SHARED) {
              sum += alpha* A(j,row) * x_shar[j];
          } else {
              sum += alpha* A(j,row) * x[j]; // A is spd, we access it col-wise for coalesced memory
          }
        }
        y[row] = sum;
    }
}

__global__ void dgemv2(int m, int n, double alpha, Matrix A, double * x, double * y, int width) {
  /// Computes truncated cross-products alpha * A(i,j) * x(i), the final summation is done elsewhere
  
  // Get the indices of x to make cross_product
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int start_col = blockIdx.y * width;
  int end_col = start_col + width;
  if (end_col > n) {end_col = n;};
  
  // Instantiate an array in shared memory too store x
  extern __shared__ double x_shar[];
  
  // Each thread in a block will put some data of x into shared memory

  // First look at columns to be put in x_shar by this thread
  int chunk;
  if (blockDim.x > width) {chunk = 1;}
  else {chunk = width/blockDim.x + (width%blockDim.x==0 ? 0 : 1);};
  int first_col = chunk*threadIdx.x + start_col;
  int last_col = first_col + chunk;
  if (last_col > end_col) {last_col=end_col;};

  // Put values of x in shared memory
  for (int col = first_col; col < last_col; col++) {
    x_shar[col-start_col] = x[col];
  }

  __syncthreads(); // Make sure it is done for all threads
  
  // Compute the partial cross-products and store them in y
  if (row < m) {
    double sum = 0;
    for (int col = start_col; col < end_col; col++) {
      sum += alpha*A(col,row)*x_shar[col-start_col]; // A is symmetric
    }
    y[blockIdx.y*n + row] = sum;
  }
}

__global__ void reduce(int n, int n_reduce, double * y) {
  /// Makes the summation to achieve y = alpha*A*x

  int row = blockDim.x * blockIdx.x + threadIdx.x;

  // if row sums up the partial cross-products in a row, and save it in the first row of y
  if (row < n) {
    double sum = 0;
    for (int j = 0; j < n_reduce; j++) {
      sum += y[j*n + row];
    }
    y[row] = sum;
  }
}

__global__ void daxpy(int n, double alpha, double * x, double beta, double * y) {
    // Computes y = alpha * x + beta * y

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
      double tmp = beta * y[i];
      y[i] = alpha * x[i] + tmp;
    }

}


void CGSolver::solve(double * x, int block_size, int width) {

  // Compute the grid sizes
  int grid_size = m_m / block_size + (m_m % block_size == 0 ? 0 : 1); //grid_size along rows
  int n_reduce = m_n / width + (m_n % width == 0 ? 0 : 1); // grid_size along y (# of partial cross-products)
  dim3 grid = dim3(grid_size, n_reduce);

  size_t sm = (width+block_size-1)*sizeof(double); //size of shared memory 
                                                  // (+block_size-1 to prevent, I think it works) 

  // Allocate memory in host and device
  double * r;
  double * p;
  double * Ap;
  double * tmp;

  cudaMallocManaged(&r, m_n*sizeof(double));
  cudaMallocManaged(&p, m_n*sizeof(double));
  // If using dgemv Ap only needs  m_n floating points
  if (width == m_n){
    cudaMallocManaged(&Ap, m_n*sizeof(double));
  } else {
    cudaMallocManaged(&Ap, m_n*n_reduce*sizeof(double));
  }
  cudaMallocManaged(&tmp, m_n*sizeof(double));

  // r = b - A * x;
  if (width == m_n) {
    dgemv<<<grid_size, block_size>>>(m_m, m_n, 1., m_A, x, Ap);
  } else {
    dgemv2<<<grid, block_size, sm>>>(m_m, m_n, 1., m_A, x, Ap, width);
    cudaDeviceSynchronize();
    reduce<<<grid_size, block_size>>>(m_n, n_reduce, Ap);
  }
  cudaDeviceSynchronize();

  //r = m_b;
  //for (int i = 0; i < m_n; i++) {r[i] = m_b[i];};
  daxpy<<<grid_size, block_size>>>(m_n, 1., m_b, 0., r);
  cudaDeviceSynchronize();
  
  // r = r - Ap
  daxpy<<<grid_size, block_size>>>(m_n, -1., Ap, 1., r);
  cudaDeviceSynchronize();

  // p = r;
  //for (int i = 0; i < m_n; i++) {p[i] = r[i];};
  daxpy<<<grid_size, block_size>>>(m_n, 1., r, 0., p);
  cudaDeviceSynchronize();

  // rsold = r' * r;
  auto rsold = cblas_ddot(m_n, r, 1, p, 1);

  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k) {

    // Ap = A * p;
    if (width == m_n) {
      dgemv<<<grid_size, block_size>>>(m_m, m_n, 1., m_A, p, Ap);
    } else {
      dgemv2<<<grid, block_size, sm>>>(m_m, m_n, 1., m_A, p, Ap, width);
      cudaDeviceSynchronize();
      reduce<<<grid_size, block_size>>>(m_n, n_reduce, Ap);
    }
    cudaDeviceSynchronize();

    // alpha = rsold / (p' * Ap);
    auto alpha = rsold / std::max(cblas_ddot(m_n, p, 1, Ap, 1),
                                  rsold * NEARZERO);

    // x = x + alpha * p;
    daxpy<<<grid_size, block_size>>>(m_n, alpha, p , 1., x);

    // r = r - alpha * Ap;
    daxpy<<<grid_size, block_size>>>(m_n, -alpha, Ap, 1., r);
    cudaDeviceSynchronize();

    // rsnew = r' * r;
    auto rsnew = cblas_ddot(m_n, r, 1, r, 1);

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;
    daxpy<<<grid_size, block_size>>>(m_n, 1., r, beta, p);
    cudaDeviceSynchronize();

    // rsold = rsnew;
    rsold = rsnew;
    if (DEBUG) {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(rsold) << "\r" << std::flush;
    }
  }

  if (DEBUG) {
    std::fill_n(&r[0], m_n, 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
                x, 1, 0., r, 1);
    cblas_daxpy(m_n, -1., m_b, 1, r, 1);
    auto res = std::sqrt(cblas_ddot(m_n, r, 1, r, 1)) /
               std::sqrt(cblas_ddot(m_n, m_b, 1, m_b, 1));
    auto nx = std::sqrt(cblas_ddot(m_n, x, 1, x, 1));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific
              << std::sqrt(rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  }

  cudaFree(&r);
  cudaFree(&p);
  cudaFree(&Ap);
  cudaFree(&tmp);
}

void CGSolver::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

/*
Initialization of the source term b
*/
void Solver::init_source_term(double h) {
  
  //m_b.resize(m_n);
  cudaMallocManaged(&m_b, m_n*sizeof(double));

  for (int i = 0; i < m_n; i++) {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
  }
}
