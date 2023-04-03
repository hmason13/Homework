#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n+1; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n, long* holder) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  
  const int chunky = n/8;
  #pragma omp parallel
  {
    int p = omp_get_num_threads();
    int t = omp_get_thread_num();
  
    #pragma omp for schedule(static,chunky)
    for (long i=1; i < n+1; i++) {
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    }
    
    // this probably won't speed it up, but maybe?
    #pragma omp for schedule(static,chunky) 
    for (long i=chunky; i < n+1; i += chunky) {
      holder[(i/chunky)] = prefix_sum[i];
    }
    
    #pragma omp for schedule(static,chunky)
    for (long i=1; i < n+1; i++) {
      int tot = (i-1)/chunky;
      for (long j=1; j < tot+1; j++){
        prefix_sum[i] += holder[j];
      }
    }
    
    
    
  }
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc((1+N) * sizeof(long));
  long* B1 = (long*) malloc((1+N) * sizeof(long));
  long* hld= (long*) malloc(N * sizeof(long));
  
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N+1; i++) B1[i] = 0;
  for (long i = 0; i < N; i++) hld[i] = 0;
  
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  float hold1 = omp_get_wtime() - tt;
  printf("sequential-scan = %fs\n", hold1);

  tt = omp_get_wtime();
  scan_omp(B1, A, N, hld);
  float hold2 = omp_get_wtime() - tt;
  printf("parallel-scan   = %fs\n", hold2);

  long err = 0;
  for (long i = 0; i < N+1; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);
  
  


  free(A);
  free(B0);
  free(B1);
  free(hld);
  return 0;
}
