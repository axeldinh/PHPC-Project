#include "cg.hh"
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>


using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

static void usage(const std::string & prog_name) {
  std::cerr << prog_name << "[martix-market-filename] <block_size> <width [default: 1]>" << std::endl;
  exit(0);
}

/*
Implementation of a simple CG solver using matrix in the mtx format (Matrix
market) Any matrix in that format can be used to test the code.

I implemented two kernels for dgemv:
  - One with one thread per cross product
  - One with n_reduce threads per cross- product (In fact two kernels are used here)
    Each thread executes around width computations of the cross-product

To call the file:

  ./cgsolver [mat.mtx] block_size width

  if width + block_size > 6000, the shared memory can't be used
  in that case we use the 1rst version of dgemv (which as fixed shared memory size of 4096)
*/
int main(int argc, char ** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " [martix-market-filename] <block_size> <width [default: 1]>"
              << std::endl;
    return 1;
  }

  CGSolver solver;
  solver.read_matrix(argv[1]);

  int n = solver.n();
  int m = solver.m();
  double h = 1. / n;

  solver.init_source_term(h);

  double * x_d;
  cudaMallocManaged(&x_d, n*sizeof(double));
  std::fill_n(&x_d[0], n, 0.);

  int block_size = std::stoi(argv[2]);
  int width = n;
  if (argc == 4) {
    try {
      width = std::stoi(argv[3]);
    } catch(std::invalid_argument &) {
      usage(argv[0]);
    }
  }

  if (width + block_size - 1 > 6000) {
    width = n;
  }

  std::cout << "Call CG dense on matrix size (" << m << " x " << n << ")"
            << " GridSize = (" << n / block_size + (n % block_size == 0 ? 0 : 1) << " x " << n / width + (n % width == 0 ? 0 : 1)  
            << ") BlockSize = " << block_size << " Width = " << width
            << std::endl;
  auto t1 = clk::now();
  solver.solve(x_d, block_size, width);
  second elapsed = clk::now() - t1;
  std::cout << "Time for CG (dense solver)  = " << elapsed.count() << " [s]\n";

  /*
  CGSolverSparse sparse_solver;
  sparse_solver.read_matrix(argv[1]);
  sparse_solver.init_source_term(h);

  std::vector<double> x_s(n);
  std::fill(x_s.begin(), x_s.end(), 0.);

  std::cout << "Call CG sparse on matrix size " << m << " x " << n << ")"
            << std::endl;
  t1 = clk::now();
  sparse_solver.solve(x_s);
  elapsed = clk::now() - t1;
  std::cout << "Time for CG (sparse solver)  = " << elapsed.count() << " [s]\n";
  */
  
  return 0;
}
