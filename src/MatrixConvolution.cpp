#include <iostream>
#include <fstream>
#include <string>
#include <mpi.h>
#include "Matrix.h"


int main(int argc, char** argv)
{
	string fname, res_fname;
	unsigned int fsize;
	int rank, numtasks;
	int sizeA, sizeB=0;
	int recvsize;
	/*vars for counts calculation*/
	int count_iter_all, count_iter_base, count_iter_rem;
	int counts_base;
	int* sendcounts = NULL;
	//int* recvcounts = NULL;
	int* displs = NULL;

	/*bufs for arrays*/
	float* plainA = NULL;
	float* plainB = NULL;
	float* recvbuf = NULL;
	float* plainC = NULL;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

	if (rank == 0) {
		Matrix matrA, matrB;

		/* Get file name */
		std::cout << "Enter file name > " << std::endl;
		std::cin >> fname;
		std::ifstream fin(fname);
		try {
		matrA.readFromFile(fin);
		matrB.readFromFile(fin);

		/*Gen result filename*/
		fin.seekg(0, std::ios::end);
		fsize = fin.tellg() / (1024 * 1024);
		res_fname = "data\\res_" + to_string(fsize) + ".txt";
		fin.close();
		}
		catch (std::exception& e) {
			std::cerr << e.what() << std::endl;
			return -1;
		}

		sizeA = matrA.getSize();
		sizeB = matrB.getSize();
		plainA = matrA.to_plain_array();
		plainB = matrB.to_plain_array();

		//broadcast matrB to all tasks
		MPI_Bcast(&sizeB, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(plainB, sizeB * sizeB, MPI_FLOAT, 0, MPI_COMM_WORLD);
		//broadcast matrA size to all tasks
		MPI_Bcast(&sizeA, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
		count_iter_all = sizeA - sizeB + 1;
		plainC = new float[count_iter_all * count_iter_all];
	}
	else {
		MPI_Bcast(&sizeB, 1, MPI_INT, 0, MPI_COMM_WORLD);
		plainB = new float[sizeB * sizeB];
		MPI_Bcast(plainB, sizeB * sizeB, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&sizeA, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}
	
	/*calculations*/
	count_iter_all = sizeA - sizeB + 1; //num of all iterations - convolution res size
	count_iter_base = count_iter_all / numtasks; //num of iterations for each task
	count_iter_rem = count_iter_all % numtasks; // num of remained iterations
	counts_base = sizeA * (sizeB + count_iter_base - 1); //base num of A elements for each task

	int displs_sum = 0;
	sendcounts = new int[numtasks];
	displs = new int[numtasks];
	for (int i = 0; i < numtasks; i++) {
		sendcounts[i] = counts_base;
		displs[i] = displs_sum;
		displs_sum += count_iter_base * sizeA;
		if (count_iter_rem > 0 and i > 0) {
			sendcounts[i] += sizeA;
			displs_sum += sizeA;
			count_iter_rem--;
		}
	}
	/*end calculations*/

	/*allocate memory for matrA part of needed size*/
	recvsize = sendcounts[rank];
	recvbuf = new float[recvsize];
	
	MPI_Scatterv(plainA, sendcounts, displs, MPI_FLOAT, recvbuf, recvsize, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if (plainA) delete[] plainA;

	//double start_time = MPI_Wtime();

	/*CONVOLUTION*/
	float sum = 0;
	int vertic_iter_count = recvsize / sizeA - sizeB + 1; // number of interations for vertical steps for current task
	float* plain_res = new float[count_iter_all * vertic_iter_count]; //count_iter_all - iter num for horisontal steps for cur task
	double start_time = MPI_Wtime();
	for (int i = 0; i < vertic_iter_count; i++ )
		for (int j = 0; j < count_iter_all; ++j) {
			/* Calculate element of convolution */
			sum = 0;
			for (int k = 0; k < sizeB; ++k) {
				for (int l = 0; l < sizeB; ++l) {
					sum += recvbuf[((i+k)*sizeA+j + l)] *plainB[k * sizeB + l];//A[i+k][j+l] * B[k][l];
				}
			}
			plain_res[i* count_iter_all+j] = sum; //C[i][j]
		}

	/*Get convolution time*/
	double end_time = MPI_Wtime();
	double task_time = end_time - start_time;
	double diff_time;
	MPI_Reduce(&task_time, &diff_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	/*Free bufs*/
	delete[] recvbuf;
	delete[] plainB;

	/*Gather results*/
	/*transform sendcounts to new recvcounts for root task*/
	if (rank == 0) {
		displs_sum = 0;
		for (int i = 0; i < numtasks; i++) {
			sendcounts[i] = sendcounts[i] / sizeA - sizeB + 1;
			sendcounts[i] *= count_iter_all;
			displs[i] = displs_sum;
			displs_sum += sendcounts[i];
		}
	}
	
	MPI_Gatherv(plain_res, vertic_iter_count*count_iter_all, MPI_FLOAT, plainC, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	delete[] sendcounts;
	delete[] displs;
	delete[] plain_res;

	if (rank == 0) {
		Matrix matrC(count_iter_all, plainC);
		delete[] plainC;
		/* Save results */
		matrC.saveToFile(res_fname, diff_time, fsize);
		std::cout << "Time: " << diff_time << std::endl;
	}

	MPI_Finalize();
	return 0;
}