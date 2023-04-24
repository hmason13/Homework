#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
  
  MPI_Init(&argc, &argv);
  
  int N = 400; // size of vector to be scanned
  int size=4;
  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
//  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  
  
  // allocate memory for vector and its sums
  float* vectorSum = (float*) malloc((N/size)*sizeof(float));
  float* totals = (float*) malloc((size)*sizeof(float));
  int mytotal = 0;
  float* vectorFull = (float*) malloc((N)*sizeof(float));

  
  // initialize the vector and send to all other processes
  // initialize with 1 everywhere for bugfix
  for (int i = 0; i < N; i++) vectorFull[i] = 1.0;
  MPI_Scatter(vectorFull, N/size, MPI_FLOAT,vectorSum, N/size, MPI_FLOAT, 0, comm);
  
  
  // sum the part of vector for which each process is responsible
  printf("rank %i\n",rank);
  totals[rank] = 0;
  for (int i = 0; i < N/size; i++) {
    mytotal += vectorSum[i];
    vectorSum[i] = 1.0*mytotal;
  }
  
  printf(" rank %i \n",rank);
  
  // send each process total to other processes
  MPI_Allgather(&mytotal, 1, MPI_FLOAT, totals, 1, MPI_FLOAT, comm);
  
  // use totals to modify my sums
  for (int i = 0; i < rank; i++) {
    for (int j = 0; j < (N/size); j++){
      vectorSum[j] += totals[i];
    }
  }
  
  // print totals for each process to see if it worked
  printf("My total is %f, from rank %i\n", totals[rank], rank);

  MPI_Barrier(comm);
  MPI_Finalize();
  return 0;
}