#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <random>
#include "utils.h"

float gsSeq(float* u, int N, int iters) {
  const float h = 1.0 / (float)(N+1.0);
  float res = 0;
  
  for (int counter = 1; counter < iters; counter++) {
    float holder = 0;
    float local = 0;
    res = 0;  
    
    for (int i = 1; i < N+1; i++) {
      for (int j = 1; j < N+1; j++)  {
        holder = u[(i+1)*N+j] + u[(i-1)*N+j] + u[i*N+(j+1)] + u[i*N+(j-1)];
        local = -1 + ((-1*holder + 4*u[i*N+j])/(h*h));
        u[i*N+j] = (holder + (h*h))/4;
        if (std::abs(local) > res) {res = std::abs(local);}
      }  
    }
  }
  return res;
}

float gsOmp(float* u, int N, int iters, float* residuals) {
  const float h = 1.0 / (float)(N+1.0);
  float resReturned = 0;
  
  #pragma omp shared(resReturned) parallel 
  {
  int t = omp_get_thread_num();
  int p = omp_get_num_threads();
  
  for (int counter = 1; counter < iters; counter++) {
    float holder = 0;
    float local = 0;
    for (int i = 0; i < 2*p; i++) {residuals[i] = 0;}
  
  #pragma omp for schedule(static)  
  for (int i = 1; i < N+1; i++) {
    for (int j = 1; j < N+1; j++) {
      if ((i+j) % 2 == 0) {
        holder = u[(i+1)*N+j] + u[(i-1)*N+j] + u[i*N+(j+1)] + u[i*N+(j-1)];
        local = std::abs(-1 + ((-1*holder + 4*u[i*N+j])/(h*h)));
        u[i*N + j] = (holder + (h*h))/4;
        if (local > residuals[2*t]) {residuals[2*t] = local;}
      }
    }
  }
  
  #pragma omp for schedule(static)
  for (int i = 1; i < N+1; i++) {
    for (int j = 1; j < N+1; j++) {
      if ((i+j) % 2 == 1) {
        holder = u[(i+1)*N+j] + u[(i-1)*N+j] + u[i*N+(j+1)] + u[i*N+(j-1)];
        local = std::abs(-1 + ((-1*holder + 4*u[i*N+j])/(h*h)));
        u[i*N + j] = (holder + (h*h))/4;
        if (local > residuals[2*t+1]) {residuals[2*t+1] = local;}
      }
    }
  }  
  }
  
  #pragma omp single
  {
  for (int i = 0; i < 2*p; i++) {
    if (residuals[i] > resReturned) {resReturned = residuals[i];}
  }
  }
  
  }
  return resReturned;
}

int main() {
  int N = 75;
  int iterations = 7000;
  float resSeq = 0;
  float resOmp = 0;
  
  float* Bseq = (float*) malloc((N+1)*(N+1)*sizeof(float));
  float* Bomp = (float*) malloc((N+1)*(N+1)*sizeof(float));
  float* pRs = (float*) malloc(N * sizeof(float)); // # of threads < N !
  for (int i = 0; i < (N+1)*(N+1); i++) {
    Bseq[i] = 0; pRs[i%N] = 0; Bomp[i] = 0;
  }

  Timer t1;
  t1.tic();
  resSeq = gsSeq(Bseq, N, iterations);
  float timeSeq = t1.toc();
  
  Timer t2;
  t2.tic();
  resOmp = gsOmp(Bomp, N, iterations, pRs);
  float timeOmp = t2.toc();
  
  printf("        Dimension   Time      Iterations      res         \n");
  printf("Seq:%10i %10f %10i, %10f\n", N, timeSeq, iterations, resSeq);
  printf("Omp:%10i %10f %10i, %10f\n", N, timeOmp, iterations, resOmp);

  return 0;
}