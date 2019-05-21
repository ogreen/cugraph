#pragma once

#include <stdint.h>
#include <cuda_profiler_api.h>

// #include <thrust/device_vector.h>
// #include <thrust/pair.h>
// #include <thrust/sort.h>

// #include "types.cuh"


template <typename KEY_T, typename LEN_T, int KEY_LEN_BITS>
__global__ void binCounting(KEY_T* array, LEN_T numKeys, LEN_T* binSizes, LEN_T shiftRight)
{
	LEN_T pos = blockIdx.x*blockDim.x +  threadIdx.x;
	if(pos>=numKeys)
		return;

	LEN_T myBin = (array[pos]>>shiftRight);

	atomicAdd((LEN_T*) binSizes+myBin,(LEN_T)1L);

}

template <typename KEY_T, typename VAL_T, typename LEN_T, int THREADS, int MAX_GPUS, int THREADS_MID>
__global__ void partitionRelabel(	
										KEY_T* array, KEY_T* reorgArray,
										VAL_T* vals, VAL_T* reorgVals,
										LEN_T numKeys, 
										LEN_T* binOffsets,
										KEY_T* binSplitters, int numPartitions)
{
	LEN_T pos = blockIdx.x*blockDim.x +  threadIdx.x;
	LEN_T tid = threadIdx.x;
	__shared__ LEN_T counter[2][MAX_GPUS+1];
	__shared__ LEN_T counter2[MAX_GPUS+1];
	__shared__ LEN_T prefix[MAX_GPUS+1];
	__shared__ LEN_T globalPositions[MAX_GPUS+1];


	__shared__ KEY_T reOrderedLocalKey[THREADS];
	__shared__ VAL_T reOrderedLocalVal[THREADS];

	__shared__ LEN_T reOrderedPositions[THREADS];

	if(tid<numPartitions){
		counter[0][tid]=0L;
		counter[1][tid]=0L;
		counter2[tid]=0L;
		prefix[tid] =0L;
	}
	__syncthreads();
	KEY_T key;
	VAL_T val;
	LEN_T gpuBin=0L;

	if(pos<numKeys){
		key		=	array[pos];
		val		=	vals[pos];
		while(gpuBin<numPartitions){
			if(key<binSplitters[gpuBin])
				break;
			gpuBin++;
		}
		if(gpuBin==numPartitions){
			gpuBin--;
		}
		if(tid<THREADS_MID)
			atomicAdd(counter[0]+(gpuBin),1L);
		else
			atomicAdd(counter[1]+(gpuBin),1L);

	}
	__syncthreads();

	if(tid<numPartitions){
			globalPositions[tid]=atomicAdd(binOffsets+tid,counter[0][tid]+counter[1][tid]);
	}
	if(tid==0){
		for(int p=0; p<numPartitions;p++){
			prefix[p+1]=prefix[p]+counter[0][p]+counter[1][p];
		}
	}

	__syncthreads();

	LEN_T posWithinBin;
	if(pos<numKeys){
		posWithinBin = atomicAdd(counter2+gpuBin,1L);
		reOrderedLocalKey[prefix[gpuBin]+posWithinBin]=key;
		reOrderedLocalVal[prefix[gpuBin]+posWithinBin]=val;

		reOrderedPositions[prefix[gpuBin]+posWithinBin]=posWithinBin+globalPositions[gpuBin];

	}
	__syncthreads();

	if(pos<numKeys){
		reorgArray[reOrderedPositions[tid]]  = reOrderedLocalKey[tid];
		reorgVals[reOrderedPositions[tid]] 	 = reOrderedLocalVal[tid];
	}	
	__syncthreads();

}
