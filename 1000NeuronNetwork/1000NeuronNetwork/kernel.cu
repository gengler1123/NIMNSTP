#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <vector>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <cmath>
#include <fstream>

#include "kernels.cuh"
#include "helperfunctions.h"

#define numPatterns 20
#define numDriven 50
#define valDriven 100

#define maxExcitWeight 300
#define maxInhibWeight -400

/* Connection Probability */
#define tauConn 5.0f
#define pExcitMax .3
#define pInhibMax .4

/* Connection Delay */
#define xMax 5
#define yMax 5
#define maxDistance sqrt(xMax * xMax + yMax * yMax)

/* Pattern Generation */
#define eps .5

/* Psuedometric Calculation */
#define tauPSPF 15.0
#define PSPFCutoff 10
#define writeRawData false


int main()
{
	std::cout << "Setting Up Simulation Parameters" << std::endl;


	int numNeurons = 1000;
	int numExcit = 800;
	int T = 2000;
	int equilizationTime = 100;
	int transientTime = 300;
	int maxDelay = 15;

	std::mt19937 rd(time(NULL));
	std::uniform_real_distribution<float> dist(0.0, 1.0);
	std::uniform_int_distribution<int> intDist(1, maxDelay);

	std::vector<float> xCoordinate, yCoordinate;

	/* CUDA Parameters */
	int numThreads = 512;

	/* Neurons */
	float *h_v, *d_v, *h_u, *d_u, *h_I, *d_I, *h_driven, *d_driven;
	bool *d_cf, *h_cf;

	h_v = new float[numNeurons];
	h_u = new float[numNeurons];
	h_I = new float[numNeurons*maxDelay];
	h_cf = new bool[numNeurons];
	h_driven = new float[numNeurons];

	bool **SpikeTrainYard = new bool*[T];
	float **VoltageTrace = new float *[T];

	float x, y;

	std::cout << "Generating Network Structure" << std::endl;

	for (int i = 0; i < numNeurons; i++)
	{
		x = 5 * dist(rd);
		y = 5 * dist(rd);
		xCoordinate.push_back(x);
		yCoordinate.push_back(y);

		for (int j = 0; j < maxDelay; j++)
		{
			h_I[i*maxDelay + j] = 0;
		}
		h_v[i] = -60;
		h_u[i] = 0;
		h_cf[i] = false;
		if (i < 100)
		{
			h_driven[i] = valDriven;
		}
		else
		{
			h_driven[i] = 0;
		}
	}

	for (int t = 0; t < T; t++)
	{
		SpikeTrainYard[t] = new bool[numNeurons];
		VoltageTrace[t] = new float[numNeurons];
	}


	/* Edges */

	std::vector<int> h_source; int *d_source;
	std::vector<int> h_target; int *d_target;
	std::vector<float> h_weight; float *d_weight;
	std::vector<int> h_delay; int *d_delay;
	float distance, xHold, yHold;
	int holdDelay;
	int MaxDistance = maxDistance;

	for (int n = 0; n < numNeurons; n++)
	{
		for (int m = 0; m < numNeurons; m++)
		{
			if (n != m)
			{
				xHold = pow(xCoordinate[n] - xCoordinate[m], 2);
				yHold = pow(yCoordinate[n] - yCoordinate[m], 2);
				distance = sqrt(xHold + yHold);

				if (n < numExcit)
				{
					if (dist(rd) < pExcitMax * exp(-distance / tauConn))
					{
						h_source.push_back(n);
						h_target.push_back(m);
						holdDelay = maxDelay * ceil(distance / MaxDistance);
						h_delay.push_back(holdDelay);
						h_weight.push_back(dist(rd) * maxExcitWeight);
					}
				}

				else
				{
					if (dist(rd) < pInhibMax * exp(-distance / tauConn))
					{
						h_source.push_back(n);
						h_target.push_back(m);
						holdDelay = maxDelay * ceil(distance / MaxDistance);
						h_delay.push_back(holdDelay);
						h_weight.push_back(dist(rd) * maxInhibWeight);
					}
				}

			}
		}
	}

	int numEdges = h_source.size();

	/* Pseudo and Metric Code */
	std::cout << "Pseudometric and Metric Code " << std::endl;

	float ***h_PSPFS, ***d_PSPFS, ***D_PSPFS;
	float ***h_PVRS, ***d_PVRS, ***D_PVRS;

	h_PSPFS = new float**[numPatterns]; h_PVRS = new float**[numPatterns];
	d_PSPFS = new float**[numPatterns]; d_PVRS = new float**[numPatterns];
	cudaMalloc((void**)&D_PSPFS, numPatterns * sizeof(float**));
	cudaMalloc((void**)&D_PVRS, numPatterns * sizeof(float**));
	for (int num = 0; num < numPatterns; num++)
	{
		h_PSPFS[num] = new float*[numNeurons];
		h_PVRS[num] = new float*[numNeurons];
		cudaMalloc((void**)&d_PSPFS[num], numNeurons * sizeof(float*));
		cudaMalloc((void**)&d_PVRS[num], numNeurons*sizeof(float));
		for (int n = 0; n < numNeurons; n++)
		{
			cudaMalloc((void**)&h_PSPFS[num][n], T * sizeof(float));
			cudaMalloc((void**)&h_PVRS[num][n], T * sizeof(float));
		}
		cudaMemcpy(d_PSPFS[num], h_PSPFS[num], numNeurons * sizeof(float*), cudaMemcpyHostToDevice);
		cudaMemcpy(d_PVRS[num], h_PVRS[num], numNeurons * sizeof(float*), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(D_PSPFS, d_PSPFS, numPatterns * sizeof(float**), cudaMemcpyHostToDevice);
	cudaMemcpy(D_PVRS, d_PVRS, numPatterns * sizeof(float**), cudaMemcpyHostToDevice);

	float normalizer = std::max(abs(maxExcitWeight), abs(maxInhibWeight));
	

	/* CUDA Memory Functions */

	cudaMalloc((void**)&d_v, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_u, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_I, numNeurons * maxDelay * sizeof(float));
	cudaMalloc((void**)&d_driven, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_cf, numNeurons * sizeof(bool));


	cudaMalloc((void**)&d_source, numEdges * sizeof(int));
	cudaMalloc((void**)&d_target, numEdges * sizeof(int));
	cudaMalloc((void**)&d_weight, numEdges * sizeof(float));
	cudaMalloc((void**)&d_delay, numEdges*sizeof(float));


	cudaMemcpy(d_v, h_v, numNeurons * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u, h_u, numNeurons * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, h_I, numNeurons * sizeof(float), cudaMemcpyHostToDevice);


	cudaMemcpy(d_source, h_source.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_target, h_target.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, h_weight.data(), numEdges * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_delay, h_delay.data(), numEdges * sizeof(float), cudaMemcpyHostToDevice);

	/* Generate Input Patterns */
	std::vector<float> xPattern, yPattern;
	for (int i = 0; i < numPatterns; i++)
	{
		x = (xMax - 1) * dist(rd) + .5;
		y = (yMax - 1) * dist(rd) + .5;
		xPattern.push_back(x);
		yPattern.push_back(y);
	}

	std::vector<std::vector<int>> PatternStorage;
	for (int idx = 0; idx < numPatterns; idx++)
	{
		std::vector<int> vec;
		for (int i = 0; i < numNeurons; i++)
		{
			xHold = pow((xPattern[idx] - xCoordinate[i]), 2);
			yHold = pow((yPattern[idx] - yCoordinate[i]), 2);
			distance = sqrt(xHold + yHold);
			if (distance < eps)
			{
				vec.push_back(i);
			}
		}
		PatternStorage.push_back(vec);
	}
	
	/* Firings Storage */
	std::vector<std::vector<std::vector<int>>> Firings;

	/* Run Simulation */
	for (int num = 0; num < numPatterns; num++)
	{

		std::cout << "Pattern number " << num << std::endl;

		for (int i = 0; i < numNeurons; i++)
		{
			h_driven[i] = 0;

		}

		for (int i = 0; i < PatternStorage[num].size(); i++)
		{
			int idx = PatternStorage[num][i];
			h_driven[i] = valDriven;
		}

		cudaMemcpy(d_driven, h_driven, numNeurons * sizeof(float), cudaMemcpyHostToDevice);


		for (int t = 0; t < equilizationTime; t++)
		{
			/* Run Timesteps, No Communication */
			NeuronTimestepNoWrite <<<(numNeurons + numThreads - 1) / numThreads, numThreads >>>(
				numNeurons,
				numExcit,
				d_v,
				d_u,
				d_I,
				d_cf,
				d_driven,
				t,
				maxDelay);
		}

		for (int t = 0; t < transientTime; t++)
		{
			/* Run Timesteps, Communication, No Writing */
			NeuronTimestepNoWrite << <(numNeurons + numThreads - 1) / numThreads, numThreads >> >(
				numNeurons,
				numExcit,
				d_v,
				d_u,
				d_I,
				d_cf,
				d_driven,
				t,
				maxDelay);

			CommunicationPhase << <(numEdges + numThreads - 1) / numThreads, numThreads >> >(
				numEdges,
				d_cf,
				d_source,
				d_target,
				d_weight,
				d_I,
				t,
				maxDelay);

		}
		std::vector<std::vector<int>> firings;
		for (int t = 0; t < T; t++)
		{
			/* Run Timesteps, Communication, Write Results*/
			NeuronTimestepWrite <<<(numNeurons + numThreads - 1) / numThreads, numThreads >>>(
				numNeurons,
				numExcit,
				d_v,
				d_u,
				d_I,
				d_cf,
				d_driven,
				t,
				maxDelay,
				D_PSPFS,
				D_PVRS,
				num,
				normalizer);

			CommunicationPhase <<<(numEdges + numThreads - 1) / numThreads, numThreads >>>(
				numEdges,
				d_cf,
				d_source,
				d_target,
				d_weight,
				d_I,
				t,
				maxDelay);

			cudaMemcpy(VoltageTrace[t], d_v, numNeurons * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_cf, d_cf, numNeurons * sizeof(float), cudaMemcpyDeviceToHost);
			for (int i = 0; i < numNeurons; i++)
			{
				if (h_cf[i])
				{
					std::vector<int> vec;
					vec.push_back(t);
					vec.push_back(i);
					firings.push_back(vec);
					std::cout << "Fired!" << std::endl;
				}
			}
		}
		Firings.push_back(firings);
	}
	/* Analyzing Run */
	std::cout << "Network Firings Information" << std::endl;
	for (int i = 0; i < Firings.size(); i++)
	{
		std::cout << "The number of firings for Input Pattern " << i << " is " << Firings[i].size() << std::endl;
	}


	std::cout << "Analyzing Run" << std::endl;

	float decayTable[PSPFCutoff];
	float *d_decayTable;
	cudaMalloc((void**)&d_decayTable, PSPFCutoff * sizeof(float));
	for (int i = 0; i < PSPFCutoff;i++)
	{
		decayTable[i] = exp(-(i+1)/tauPSPF);
	}
	cudaMemcpy(d_decayTable, decayTable, PSPFCutoff * sizeof(float), cudaMemcpyHostToDevice);
	for (int num = 0; num < numPatterns; num++)
	{
		ExtendKernelValues << <(numNeurons + numThreads - 1) / numThreads, numThreads >> >(
			numNeurons,
			T,
			d_decayTable,
			PSPFCutoff,
			d_PSPFS[num]);

		ExtendKernelValues << <(numNeurons + numThreads - 1) / numThreads, numThreads >> >(
			numNeurons,
			T,
			d_decayTable,
			PSPFCutoff,
			d_PVRS[num]);
	}

	int calculations = sumNumbers(numPatterns - 1);

	float *pspmvalues = new float[calculations];
	float *pvrvalues = new float[calculations];

	float *d_pspmvalues, *d_pvrvalues;
	cudaMalloc((void**)&d_pspmvalues, calculations * sizeof(float));
	cudaMalloc((void**)&d_pvrvalues, calculations * sizeof(float));

	for (int N = 0; N < numPatterns; N++)
	{
		for (int M = N + 1; M < numPatterns; M++)
		{

			CalculateDistances <<<(calculations + numThreads - 1)/numThreads, numThreads >>>(
				calculations,
				numNeurons,
				T,
				tauPSPF,
				D_PSPFS,
				N,M,
				d_pspmvalues);

			CalculateDistances <<<(calculations + numThreads - 1) / numThreads, numThreads >>>(
				calculations,
				numNeurons,
				T,
				tauPSPF,
				D_PVRS,
				N, M,
				d_pvrvalues);
		}
	}


	cudaMemcpy(pspmvalues, d_pspmvalues, calculations * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(pvrvalues, d_pvrvalues, calculations* sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "PSPM Values" << std::endl;

	for (int i = 0; i < calculations; i++)
	{
		std::cout << pspmvalues[i] << "," << pvrvalues[i] << std::endl;
	}



	float ***arr = new float**[numPatterns];
	for (int i = 0; i < numPatterns; i++)
	{
		arr[i] = new float*[numNeurons];
		for (int j = 0; j < numNeurons; j++)
		{
			arr[i][j] = new float[T];
			cudaMemcpy(arr[i][j], h_PSPFS[i][j], T*sizeof(float), cudaMemcpyDeviceToHost);
		}
	}
	/*
	for (int i = 0; i < numPatterns; i++)
	{
		for (int j = 0; j < numNeurons; j++)
		{
			std::cout << arr[i][j][2] << " ";
		}
		std::cout << std::endl;
	}
	*/





	std::cout << "Done With Program, Cleaning Up Reserved Memory" << std::endl;

	/* Clean Up Code */

	cudaDeviceReset();

	for (int t = 0; t < T; t++)
	{
		delete[] SpikeTrainYard[t];
		delete[] VoltageTrace[t];
	}

	delete[] h_v; delete[] h_u; delete[] h_I; delete[] h_cf; delete[] SpikeTrainYard; delete[] h_driven;
	delete[] VoltageTrace;


	for (int num = 0; num < numPatterns; num++)
	{
		for (int n = 0; n < numNeurons; n++)
		{
			delete[] h_PSPFS[num][n];
			delete[] h_PVRS[num][n];
			delete[] arr[num][n];
		}
		delete[] h_PSPFS[num];
		delete[] h_PVRS[num];
		delete[] d_PSPFS[num];
		delete[] d_PVRS[num];
		delete[] arr[num];
	}

	delete[] arr;
	delete[] h_PVRS; delete[] d_PVRS; delete[] h_PSPFS; delete[] d_PSPFS;

	return 0;
}