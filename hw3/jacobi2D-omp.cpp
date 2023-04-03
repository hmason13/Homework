#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <random>
#include "utils.h"

// use jacobi method to iterate a 2d array of size N

float jIterSeq(float* u, float* next, int N) {
  // take in matrix of u
  // spit out new matrix of u
  
  const float h = 1.0/((float) N+1);
  float residual = 0;
  float holder = 0;
  for (int i = 1; i < N+1; i++) {
    for (int j = 1; j < N+1; j++) {
      next[i*N+j]  = u[((i-1)*N)+j] + u[(i*N)+(j-1)];
      next[i*N+j] += u[((i+1)*N)+j] + u[(i*N)+(j+1)];
      holder = (-1*next[i*N+j] + 4*u[i*N+j])/(h*h);
      next[i*N+j] += h*h;
      next[i*N+j] /= 4;
      if (std::abs(holder-1) > residual) {
        residual = std::abs(holder-1); 
      }
    }
  }
  return residual;
}

float jFullOmp(float* u, float* next, int N, float* residuals, int num) {
  // take in matrix of u
  // spit out new matrix of u
  const float h = 1.0/((float) N+1);
  int counter = 0;

  #pragma omp parallel
  {
    int t = omp_get_thread_num();
    int p = omp_get_num_threads();
    float holder = 0;
        
    for (int counter = 0; counter < num; counter+=2) {
      
    for (int i = 0; i < N; i++) {residuals[i] = 0;}
    #pragma omp for schedule(static)
    for (int i = 1; i < N+1; i++) {
      for (int j = 1; j < N+1; j++) {
        next[i*N+j]  = u[((i-1)*N)+j] + u[(i*N)+(j-1)];
        next[i*N+j] += u[((i+1)*N)+j] + u[(i*N)+(j+1)];
        holder = (-1*next[i*N+j] + 4*u[i*N+j])/(h*h);
        next[i*N+j] += h*h;
        next[i*N+j] /= 4;
        if (std::abs(holder-1) > residuals[t]) {
          residuals[t] = std::abs(holder-1);
        }
      }
    }
    
    for (int i = 0; i < N; i++) {residuals[i] = 0;}
    #pragma omp for schedule(static)
    for (int i = 1; i < N+1; i++) {
      for (int j = 1; j < N+1; j++) {
        u[i*N+j]  = next[((i-1)*N)+j] + next[(i*N)+(j-1)];
        u[i*N+j] += next[((i+1)*N)+j] + next[(i*N)+(j+1)];
        holder = (-1*u[i*N+j] + 4*next[i*N+j])/(h*h);
        u[i*N+j] += h*h;
        u[i*N+j] /= 4;
        if (std::abs(holder-1) > residuals[t]) {
          residuals[t] = std::abs(holder-1);
        }
      }
    }    
    }
    }
  float res = 0;
  for (int i = 0; i < N; i++){
    res = std::max(res, residuals[i]);
  }
  return res;
}

int main() {
  
  int N = 80;
  float* A    = (float*) malloc((N+1)*(N+1)*sizeof(float));
  float* Bseq = (float*) malloc((N+1)*(N+1)*sizeof(float));
  float* Bomp = (float*) malloc((N+1)*(N+1)*sizeof(float));
  float* residuals = (float*) malloc(N*sizeof(float));
  
  for (int i = 0; i < (N+1)*(N+1); i++) {
    A[i] = 0;
    Bseq[i] = 0;
    Bomp[i] = 0;
  }
  
  for (int i=1; i < N+1; i++) {
    for (int j=1; j < N+1; j++) {
      A[i*N + j] = 2;
    }
  }
  
  int iterations1 = 0;
  float res = 1e7;
  Timer t1;
  t1.tic();
  while (res > 1e-2){
    res = jIterSeq(A, Bseq, N);
    iterations1 += 1;
    res = jIterSeq(Bseq, A, N);
    iterations1 += 1;
    if (iterations1 > 1e6) {break;}
  }
  float time1 = t1.toc();

  for (int i=1; i < N+1; i++) {
    residuals[i-1] = 0;
    for (int j=1; j < N+1; j++) {
      A[i*N + j] = 2;
    }
  }
  
  Timer t2;
  t2.tic();
  int iters = 20864;
  float residual= jFullOmp(A, Bomp, N, residuals, iters);
  float time2 = t2.toc();

  
  printf("   Dimension   Time      Iterations           \n");
  printf("Seq:%10i %10f %10i\n", N, time1, iterations1);
  printf("Omp:%10i %10f %10i", N, time2, iters);
  
  free(A);
  free(Bseq);
  free(Bomp);
  free(residuals);
  
  return 0;
}