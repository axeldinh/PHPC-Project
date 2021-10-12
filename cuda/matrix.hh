#include <string>
#include <vector>
#include <cuda_runtime.h>

#ifndef __MATRIX_H_
#define __MATRIX_H_

class Matrix {
public:
  Matrix(int m = 0, int n = 0) : m_m(m), m_n(n) {} //, m_a(m * n) {}
  void resize(int m, int n) {
    m_m = m;
    m_n = n;
    cudaMallocManaged(&m_a, m*n*sizeof(double));
    //m_a.resize(m * n);
  }

  __host__ __device__ inline double & operator()(int i, int j) { return m_a[i * m_n + j]; }

  __host__ __device__ inline int m() const { return m_m; }
  __host__ __device__ inline int n() const { return m_n; }
  inline double * data() {return m_a;}
  //inline double * data() { return m_a.data(); }

  void read(const std::string & filename);

private:
  int m_m{0};
  int m_n{0};
  double * m_a;
  //std::vector<double> m_a;
};

#endif // __MATRIX_H_
