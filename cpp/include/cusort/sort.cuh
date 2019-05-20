#pragma once

#include <stdlib.h>
// #include <random>
// #include <cuda_profiler_api.h>
#include <omp.h>
// #include <unistd.h>
// #include <nvToolsExt.h>


#include "binning.cuh"


#include <cub/cub.cuh>

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>



#define MAX_NUM_GPUS 16
#define BLOCK_DIM 128
#define CHECK_ERROR(str) \
	{cudaError_t err; err = cudaGetLastError(); if(err!=0) {printf("ERROR %s:  %d %s\n", str, err, cudaGetErrorString(err)); fflush(stdout); exit(0);}}



// template <typename keySort_t,typename valSort_t, typename length_t>
// struct sortDataOut{
//     // Device
//     keySort_t* d_keys;
//     valSort_t* d_vals;

//     // Host
//     length_t  h_Length;
// };



template <typename keySort_t,typename valSort_t, typename length_t>
class cuSort{
public:

	struct sortData{
	    // Device
	    keySort_t* d_keys;
	    valSort_t* d_vals;

	    // Host
	    length_t  h_Length;
	};

	// This structu is used for allocating memory once for CUB's sorting function. 
	struct bufferData{
		unsigned char* buffer;
		sortData sd;
		unsigned char* cubBuffer;

	};

	// template <typename keySort_t,typename valSort_t, typename length_t>
	struct threadData{

	    sortData sdInput;
	    sortData sdOutput;
	    // sortData sdReorder;
	    bufferData bdReorder;


	    // Device data -- accessible to a specific GPU\Device
	    unsigned char*  buffer;

	    length_t*       binSizes;

	    length_t*       binPrefix;
	    length_t*       tempPrefix;

	    keySort_t*       binSplitters;
	    // keySort_t*      partitionVals;

	    unsigned char*      cubSmallBuffer;


	    size_t cubSortBufferSize;

	    // Host data -- accessible to all threads on the CPU
	    length_t*   h_binSizes;
	    length_t*   h_binPrefix;


	};

	// A recursive binary search function. It returns location of x in given array arr[l..r] is present, 
	// otherwise it returns the bin id with the smallest value larger than x
	length_t binarySearch(length_t *bins, length_t l, length_t r, length_t x) 
	{ 
	    if (r >= l) { 
	        length_t mid = l + (r - l) / 2; 
	  
	        // If the element is present at the middle itself 
	        if (bins[mid] == x) 
	            return mid; 
	  
	        // If element is smaller than mid, then it can only be present in left subarray 
	        if (bins[mid] > x) 
	            return binarySearch(bins, l, mid - 1, x); 
	 
	        // Else the element can only be present in right subarray 
	        return binarySearch(bins, mid + 1, r, x); 
	    } 
	  
	    // We reach here when element is not present in array and return the bin id of the smallest value greater than x
	    // return r;
	    return l; 
	} 

	void sort(	sortData* sdInput,
				sortData* sdOutput,
	    		length_t numGPUs = 1,
	    		length_t numBins = (1<<16L), 
	    		length_t binScale = 16L,
	    		bool useThrust = false) {

		if(numGPUs>MAX_NUM_GPUS){
			printf("The maximal number of GPUs (compile time variable) is set to %d\n",MAX_NUM_GPUS);
			printf("Users has specified %llu GPUs.\nThe maximal number of GPUs is being set to %d\n", numGPUs,MAX_NUM_GPUS);

			numGPUs=MAX_NUM_GPUS;

		}


		omp_set_num_threads(numGPUs);

	   	length_t keyCount = 0;
	   	for(int g=0; g<numGPUs; g++){
		   	keyCount +=sdInput[g].h_Length;
	   	}

	   	printf("Number of GPUS in the sort function is %d \n", numGPUs);

	   	printf("Number of keys in the sort function is %lld \n", keyCount);
	   	fflush(stdout);

		// Used for partitioning the output and ensuring that each GPU sorts a near equal number of elements.
		length_t keyCountUpper = keyCount;
		if(keyCount%numGPUs){
			keyCountUpper = keyCount - keyCount%numGPUs + numGPUs;
		}


		cudaEvent_t start, stop;
		float milliseconds = 0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	    
	    cudaEventRecord(start);

	    threadData tData[numGPUs];

		initThreads(sdInput,tData,numBins,numGPUs);



		cudaSetDevice(0);
		cudaEventRecord(stop); cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
	    printf("Initialization time %f\n",milliseconds);

		cudaSetDevice(0);
		cudaEventRecord(start);

		length_t h_readPositions[numGPUs+1][numGPUs+1];
		length_t h_writePositions[numGPUs+1][numGPUs+1];
		length_t h_writePositionsTransposed[numGPUs+1][numGPUs+1];

		length_t* h_globalPrefix	= (length_t*)malloc((numBins+1)*sizeof(length_t));
		keySort_t h_binSplitters[numGPUs]={0};



	    #pragma omp parallel
		{       	
	    	int64_t cpu_tid = omp_get_thread_num();
	       	cudaSetDevice(cpu_tid);

		   	length_t arraySize  = tData[cpu_tid].sdInput.h_Length;
			length_t blocks = (arraySize / BLOCK_DIM) + ((arraySize % BLOCK_DIM)?1L:0L); 

			binCounting<keySort_t,length_t,sizeof(keySort_t)*8> <<<blocks,BLOCK_DIM>>>(tData[cpu_tid].sdInput.d_keys,arraySize, 
																					   tData[cpu_tid].binSizes,sizeof(keySort_t)*8-binScale);
			
			void* _d_temp_storage_bin=tData[cpu_tid].cubSmallBuffer; size_t _temp_storage_bytes_bin=2047;

			cub::DeviceScan::ExclusiveSum(_d_temp_storage_bin, _temp_storage_bytes_bin,tData[cpu_tid].binSizes, tData[cpu_tid].binPrefix, numBins+1);

			cudaMemcpy(tData[cpu_tid].h_binPrefix, tData[cpu_tid].binPrefix,(numBins+1)*sizeof(length_t),cudaMemcpyDeviceToHost);
		#pragma omp barrier

			// The following section can still be improved.
			#pragma omp master
			{
				// Initialize the arrays that will point to the read and write positions for the partitions
				for(int64_t r=0; r<=numGPUs; r++){
					for(int64_t c=0; c<=numGPUs; c++){
						h_readPositions[r][c]=h_writePositions[r][c]=0;
					}
				}

				// Computing global prefix sum array to find partition points.
				for(int64_t b=0; b<=numBins; b++){
					h_globalPrefix[b]=0;
					for(int64_t g=0; g<numGPUs;g++){
						h_globalPrefix[b] += tData[g].h_binPrefix[b];
					}
				}
				
				length_t avgArraySize = keyCountUpper/numGPUs;
				for (int64_t thread_id=0; thread_id < (numGPUs); thread_id++){

					length_t valOfInterest = avgArraySize*(thread_id+1);
					if(valOfInterest>keyCount)
					// binSplitters[thread_id] = binarySearch(h_globalPrefix, 0, numBins, valOfInterest);
						valOfInterest=keyCount;

					h_binSplitters[thread_id] = binarySearch(h_globalPrefix, 0L, numBins, valOfInterest);
				}


				// Each thread (row) knows the length of the partitions it needs to write to the other threads
				for (int64_t r=0; r < numGPUs; r++){
					for(int64_t c=0; c<numGPUs;c++){
						h_readPositions[r+1][c+1]=tData[r].h_binPrefix[h_binSplitters[c]];
					}
				}
				// Each thread learns the position in the array other threads inputKey that it will copy its data into
				for (int64_t r=0; r < numGPUs; r++){
					for(int64_t c=0; c<numGPUs;c++){
						h_writePositions[r+1][c]=h_writePositions[r][c]+ (h_readPositions[r+1][c+1]-h_readPositions[r+1][c]);
					}
				}

				for (int64_t r=0; r <= numGPUs; r++){
					for(int64_t c=0; c<= numGPUs;c++){
						h_writePositionsTransposed[r][c]=h_writePositions[c][r];
					}
				}

				for (int64_t thread_id=0; thread_id < (numGPUs-1); thread_id++){

					h_binSplitters[thread_id] = (h_binSplitters[thread_id] << (sizeof(keySort_t)*8L-binScale));

				}

				h_binSplitters[numGPUs-1]=0;
	
				for(int s=0; s<sizeof(keySort_t); s++){
					h_binSplitters[numGPUs-1]+=((0xFF)<<s*8);
				}
			}

		#pragma omp barrier

	       	cudaMemcpy(tData[cpu_tid].binSplitters,h_binSplitters,(numGPUs)*sizeof(keySort_t), cudaMemcpyHostToDevice);
	       	cudaMemcpy(tData[cpu_tid].tempPrefix,h_readPositions[cpu_tid+1],(numGPUs+1)*sizeof(length_t), cudaMemcpyHostToDevice);

			// cudaDeviceSynchronize();
			// allocateSortData(&(tData[cpu_tid].sdReorder),arraySize);

			// Creating a temporary buffer that will be used for both reordering the input in the binning phase
			// and possibly in the sorting phase if CUB's sort is used. 
			// Therefore, the maximal buffer size is taken in this phase, where max=(array size of input, array size of output) 
			length_t elements = arraySize;
			if(h_writePositionsTransposed[cpu_tid][numGPUs]> elements)
				elements=h_writePositionsTransposed[cpu_tid][numGPUs];

			if(elements>(1L<<31L)){
				printf("The size of the array, after sampling\\binning is too large to fit on a single GPU\n");fflush(stdout);
				exit(0);
			}

			// void     *d_temp_storage = NULL; size_t   temp_storage_bytes = 0;
			tData[cpu_tid].cubSortBufferSize=0;
			cub::DeviceRadixSort::SortPairs<keySort_t,valSort_t>(NULL, tData[cpu_tid].cubSortBufferSize,
			    NULL, NULL, NULL, NULL, elements);

			// printf("CUB Allocation size request %ld\n", tData[cpu_tid].cubSortBufferSize);

			allocateBufferData(&(tData[cpu_tid].bdReorder),elements, tData[cpu_tid].cubSortBufferSize);
			cudaDeviceSynchronize();

		#pragma omp barrier

			partitionRelabel<keySort_t, valSort_t, length_t, BLOCK_DIM,32,BLOCK_DIM/2><<<blocks,BLOCK_DIM>>>
										(
										tData[cpu_tid].sdInput.d_keys,tData[cpu_tid].bdReorder.sd.d_keys,
										tData[cpu_tid].sdInput.d_vals,tData[cpu_tid].bdReorder.sd.d_vals,
										arraySize,
										tData[cpu_tid].tempPrefix,
										tData[cpu_tid].binSplitters, numGPUs);
			cudaDeviceSynchronize();

			CHECK_ERROR ("Failing after partitioning")
		}


		cudaSetDevice(0);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
	    printf("Binning time on GPUS %f\n",milliseconds);

		cudaEventRecord(start);

	    #pragma omp parallel
		{       	
	    	int64_t cpu_tid = omp_get_thread_num();
	       	cudaSetDevice(cpu_tid);


			 //    threads[cpu_tid].h_outputKeySize=h_writePositionsTransposed[cpu_tid][numGPUs];
				// fuzzyMalloc(&threads[cpu_tid].outputVal, (sizeof(valSort_t))*threads[cpu_tid].h_outputKeySize);
				// fuzzyMalloc(&threads[cpu_tid].outputKey, (sizeof(keySort_t))*threads[cpu_tid].h_outputKeySize);
	       		// printf("%d %llu\n",cpu_tid,h_writePositionsTransposed[cpu_tid][numGPUs]);



	   		allocateSortData(&tData[cpu_tid].sdOutput,h_writePositionsTransposed[cpu_tid][numGPUs]);

			CHECK_ERROR ("Failing after allocation")

				// #pragma omp barrier

				// CHECK_ERROR ("Failing in first round of mallocs")

			cudaDeviceSynchronize();
		    // for (int64_t dest_id=0; dest_id < numGPUs; dest_id++){
		    for (int64_t dest=0; dest < numGPUs; dest++){
		    		int64_t dest_id = (cpu_tid+dest)%numGPUs;
		    		// int64_t dest_id = dest;

		       		cudaMemcpyAsync   (tData[cpu_tid].sdOutput.d_keys+h_writePositionsTransposed[cpu_tid][dest_id],
		       							  tData[dest_id].bdReorder.sd.d_keys+h_readPositions[dest_id+1][cpu_tid],
		       							  (h_readPositions[dest_id+1][cpu_tid+1]-h_readPositions[dest_id+1][cpu_tid])*sizeof(keySort_t),
					       				  cudaMemcpyDeviceToDevice);

		       		cudaMemcpyAsync   (tData[cpu_tid].sdOutput.d_vals+h_writePositionsTransposed[cpu_tid][dest_id],
		       							  tData[dest_id].bdReorder.sd.d_vals+h_readPositions[dest_id+1][cpu_tid],
		       							  (h_readPositions[dest_id+1][cpu_tid+1]-h_readPositions[dest_id+1][cpu_tid])*sizeof(valSort_t),
			       						  cudaMemcpyDeviceToDevice);

			}
			cudaDeviceSynchronize();

		#pragma omp barrier

			if(useThrust){
				freeBufferData(&(tData[cpu_tid].bdReorder));

				thrust::sort_by_key(thrust::device, tData[cpu_tid].sdOutput.d_keys, tData[cpu_tid].sdOutput.d_keys+ tData[cpu_tid].sdOutput.h_Length, 
						tData[cpu_tid].sdOutput.d_vals);
				sdOutput[cpu_tid] = tData[cpu_tid].sdOutput;

			}else{

				void     *d_temp_storage = NULL; //size_t   temp_storage_bytes = 0;
				// // cub::DeviceRadixSort::SortPairs<keySort_t,valSort_t>(d_temp_storage, temp_storage_bytes,
				// //     tData[cpu_tid].sdOutput.d_keys, tData[cpu_tid].sdOutput.d_keys, 
				// //     tData[cpu_tid].sdOutput.d_vals, tData[cpu_tid].sdOutput.d_vals, tData[cpu_tid].sdOutput.h_Length);

				if(0){
					// cudaMalloc(&d_temp_storage, temp_storage_bytes);
					sortData sortDataBackup;
			   		allocateSortData(&sortDataBackup,tData[cpu_tid].sdOutput.h_Length);
					d_temp_storage = (void*)tData[cpu_tid].bdReorder.sd.d_keys;

					cub::DeviceRadixSort::SortPairs<keySort_t,valSort_t>(d_temp_storage, tData[cpu_tid].cubSortBufferSize,
					    tData[cpu_tid].sdOutput.d_keys, sortDataBackup.d_keys, 
					    tData[cpu_tid].sdOutput.d_vals, sortDataBackup.d_vals, tData[cpu_tid].sdOutput.h_Length);
					// cudaFree(d_temp_storage);


					CHECK_ERROR ("Failing after sort")

					freeBufferData(&(tData[cpu_tid].bdReorder));

					sdOutput[cpu_tid] = sortDataBackup;

					freeSortData(&(tData[cpu_tid].sdOutput));					
				}else{
					d_temp_storage = (void*)tData[cpu_tid].bdReorder.cubBuffer;
					cub::DeviceRadixSort::SortPairs<keySort_t,valSort_t>(d_temp_storage,  tData[cpu_tid].cubSortBufferSize,
					    tData[cpu_tid].sdOutput.d_keys, tData[cpu_tid].bdReorder.sd.d_keys, 
					    tData[cpu_tid].sdOutput.d_vals, tData[cpu_tid].bdReorder.sd.d_vals, tData[cpu_tid].sdOutput.h_Length);

					CHECK_ERROR ("Failing after sort")
					cudaDeviceSynchronize();
					cudaMemcpy(tData[cpu_tid].sdOutput.d_keys,tData[cpu_tid].bdReorder.sd.d_keys,tData[cpu_tid].sdOutput.h_Length*sizeof(keySort_t), cudaMemcpyDeviceToDevice);
					cudaMemcpy(tData[cpu_tid].sdOutput.d_vals,tData[cpu_tid].bdReorder.sd.d_vals,tData[cpu_tid].sdOutput.h_Length*sizeof(valSort_t), cudaMemcpyDeviceToDevice);

					cudaDeviceSynchronize();

					freeBufferData(&(tData[cpu_tid].bdReorder));

					sdOutput[cpu_tid] = tData[cpu_tid].sdOutput;

				}

			}

			CHECK_ERROR ("Failing after deallocation")
		}


		cudaSetDevice(0);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
	    printf("Sorting time on GPUS, %f\n",milliseconds);

		cudaDeviceSynchronize();

		cudaSetDevice(0);
		cudaEventRecord(start);


		freeThreadData(tData);
		
		CHECK_ERROR ("Releasing data")

		if(h_globalPrefix!=NULL)
			free(h_globalPrefix);
	  

		cudaSetDevice(0);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
	        printf("Deallocation time GPUS, %f\n", milliseconds);

	}






	void duplicateSortData(sortData* src, sortData* dest){
	allocateSortData(dest, src->h_Length);
		if(dest->h_Length>0){
			cudaMemcpy(dest->d_keys,src->d_keys,sizeof(keySort_t)*dest->h_Length,cudaMemcpyDeviceToDevice);
			cudaMemcpy(dest->d_vals,src->d_vals,sizeof(valSort_t)*dest->h_Length,cudaMemcpyDeviceToDevice);		
		}
	}



#define MEM_ALIGN 512L
	void initThreads(sortData* sdInput, threadData* tdArray, int32_t numBins, int64_t numGPUs){
		length_t binsAligned = ((numBins+1L)/MEM_ALIGN)*MEM_ALIGN+ (((numBins+1L)%MEM_ALIGN)?MEM_ALIGN:0L);
		length_t gpusAligned = ((numGPUs+1L)/MEM_ALIGN)*MEM_ALIGN+ (((numGPUs+1L)%MEM_ALIGN)?MEM_ALIGN:0L);


		length_t mallocSizeBits= 
							((binsAligned)+(binsAligned)+(gpusAligned))*sizeof(length_t) + 
							(gpusAligned)*sizeof(keySort_t) + 
							(1L << 16L); // cubSmallBuffer;

	    #pragma omp parallel
		{       	
			int64_t gpuId = omp_get_thread_num();

	       	cudaSetDevice(gpuId);
			threadData td;

			cudaMalloc((void**)&td.buffer,mallocSizeBits);

			CHECK_ERROR ("First alloc failed")

			int64_t pos=0;

			td.binSizes 		= (length_t*)((unsigned char*)(td.buffer)+pos);		pos+=(sizeof(length_t))*(binsAligned);
			td.binPrefix 		= (length_t*)((unsigned char*)(td.buffer)+pos);		pos+=(sizeof(length_t))*(binsAligned);
			td.tempPrefix 		= (length_t*)((unsigned char*)(td.buffer)+pos);		pos+=(sizeof(length_t))*(gpusAligned);

			td.binSplitters		= (keySort_t*)((unsigned char*)(td.buffer)+pos);	pos+=(sizeof(keySort_t))*(gpusAligned);

			td.cubSmallBuffer 	= ((unsigned char*)(td.buffer)+pos);				pos+=(1L<<16L);

			CHECK_ERROR ("First alloc failed")

			td.sdInput = sdInput[gpuId];

			cudaMemset(td.binSizes,0,(numBins+1)*sizeof(keySort_t));

			allocateBufferData(&td.bdReorder,0);

			// Host memory allocations

			td.h_binSizes	= (length_t*)malloc((numBins+1)*sizeof(length_t));
			td.h_binPrefix	= (length_t*)malloc((numBins+1)*sizeof(length_t));


	       	tdArray[gpuId]=td;
		}


	}
	void freeThreadData(threadData* tdArray){
	    #pragma omp parallel
		{       	
			int64_t gpuId = omp_get_thread_num();
		    cudaSetDevice(gpuId);

			threadData td = tdArray[gpuId];
			cudaFree(td.buffer);

			free(td.h_binSizes);
			free(td.h_binPrefix);
		}

	}


public:
	// Helper functions

	static void initializeMGOneGPU(int64_t gpuId, int64_t numGPUs){
		for(int64_t g=0; g<numGPUs; g++){
			if(g!=gpuId){
				int isCapable;
					cudaDeviceCanAccessPeer(&isCapable,gpuId,g);
				if(isCapable==1){
					cudaError_t err = cudaDeviceEnablePeerAccess(g,0);
					if (err == cudaErrorPeerAccessAlreadyEnabled ){
						cudaGetLastError();
					}
				}
			}
		}

	}
	static void initializeMGCommunication(int64_t numGPUs){
		omp_set_num_threads(numGPUs);

	    #pragma omp parallel 
		{       		
			int64_t gpuId = omp_get_thread_num();

		   	cudaSetDevice(gpuId);
			cuSort::initializeMGOneGPU(gpuId, numGPUs);	
		}

	}

	static void allocateSortData(sortData* sd, length_t len=0){
			if(len==0){
		   		sd->d_keys=NULL;
		   		sd->d_vals=NULL;
		   		sd->h_Length=0;
			}else{
				cudaMalloc((void**)&(sd->d_keys),(sizeof(keySort_t))*len);
				cudaMalloc((void**)&(sd->d_vals),(sizeof(valSort_t))*len);
		   		sd->h_Length=len;
			}


	}
	
	static void freeSortData(sortData* sd){
		if(sd!=NULL){
			if(sd->d_keys!=NULL)
				cudaFree(sd->d_keys);
			if(sd->d_vals!=NULL)
				cudaFree(sd->d_vals);
		}	
	}

	static void allocateBufferData(bufferData* bd, length_t len=0, length_t cubData=0){
		if(len==0){
	   		bd->buffer=NULL;
	   		allocateSortData(&bd->sd,0);
		}else{

			length_t cubDataSize =  ((cubData)/MEM_ALIGN)*MEM_ALIGN+ (((cubData)%MEM_ALIGN)?MEM_ALIGN:0L);
				// length_t sdSize = (sizeof(keySort_t))+(sizeof(valSort_t))*len+(MEM_ALIGN*2);
				// 	 sdSize =  ((sdSize)/MEM_ALIGN)*MEM_ALIGN+ (((sdSize)%MEM_ALIGN)?MEM_ALIGN:0L);

				length_t sdSize = ((len)/MEM_ALIGN)*MEM_ALIGN+ (((len)%MEM_ALIGN)?MEM_ALIGN:0L)+10*MEM_ALIGN;
			// length_t structSize=((sizeof(keySort_t)>sizeof(valSort_t))?sizeof(keySort_t):sizeof(valSort_t));
				length_t keyLen = sdSize*sizeof(keySort_t); 				
			// length_t sdSize2 =  sdSize*2*structSize;
			length_t sdSize2 =  keyLen +  sdSize*sizeof(valSort_t);


			// printf("allocating %ld\n",cubDataSize+sdSize2);

			cudaMalloc((void**)&(bd->buffer),cubDataSize+sdSize2);

			bd->sd.d_keys = (keySort_t*)((unsigned char*) bd->buffer);

			length_t startingPoint =  keyLen;// sdSize*structSize;
			// startingPoint = ((startingPoint)/MEM_ALIGN)*MEM_ALIGN+ (((startingPoint)%MEM_ALIGN)?MEM_ALIGN:0L);
			bd->sd.d_vals = (valSort_t*)((unsigned char*) bd->buffer + startingPoint);

			bd->cubBuffer = ((unsigned char*) bd->buffer + sdSize2);


			bd->sd.h_Length=len;

		}
	}

	static void freeBufferData(bufferData* bd){
		if(bd!=NULL){
			if(bd->buffer!=NULL)
				cudaFree(bd->buffer);
		}
	}


};





