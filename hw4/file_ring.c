#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  int N = 10; // ten loops around the nodes 
  int size;
  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  
  float* array = (float*) malloc(5e5*sizeof(float));
  for (int i = 0; i < 5e5; i++) array[i] = rand();
  
  int repeats = size*N;
  float tt = MPI_Wtime();
  
  for (int i=0; i < repeats; i++) {
    MPI_Status status;
    
    if (rank == i%size) {
      MPI_Send(&array, 1, MPI_INT, (i+1)%size, 101, comm);
    }
    
    if (rank == (i+1)%size) {
      MPI_Recv(&array, 1, MPI_INT, i%size, 101, comm, &status);
    }
  }
  tt = MPI_Wtime() - tt;
  if (rank == 0) {
    printf("Total amount of processes: %i\n", size);
    printf("Number of loops around ring: %i\n", N);
    printf("Time required for all loops: %f\n\n", tt);
  }
  
  MPI_Barrier(comm);
    
  MPI_Finalize();
  return 0;
}