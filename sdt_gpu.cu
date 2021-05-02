/*
 References:
 1) Nvidia GeForce GTX 1080 Whitepaper
*/

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <IL/il.h>
#include <IL/ilu.h>
#include <vector>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

using namespace std;

#define SQRT_2 1.4142
#define THREADS_PER_BLOCK 1024

ILuint loadImage(const char *filename, unsigned char ** bitmap, int &width, int &height);
void saveImage(const char *filename, float * sdt, int width, int height, unsigned char *bitmap);
void compareSDT(float *sdt1, float *sdt2, int width, int height);
void computeSDT_CPU(unsigned char * bitmap, float *sdt, int width, int height);
void computeSDT_GPU(unsigned char* bitmap, float* sdt, int width, int height);
__global__ void SDT_kernel(unsigned char* bitmap, int* edge_pixels, float* sdt);
double diff(timeval start, timeval end);

bool doSave  = false;
__device__ __constant__ int dev_width;
__device__ __constant__ int dev_height;
__device__ __constant__ int dev_sz_edge;

int main(int argc, char **argv)
{
	if(argc < 3) {
		fprintf(stderr, "Usage: %s <inputImage.png> <outputSDTImage.png>\n", argv[0]);
		return(0);
	}
	string inputImage(argv[1]);
	string outputImage(argv[2]);

	ilInit();

	int width, height;
	unsigned char *image;
	ILuint image_id = loadImage(inputImage.c_str(), &image, width, height);
	if(image_id == 0) {fprintf(stderr, "Error while reading image... aborting.\n"); exit(0);}

	//timespec time1, time2;
	float *sdt = new float[width*height];
	float *sdt_gpu = new float[width*height];
	struct timeval start, end;
	gettimeofday(&start, NULL);
	computeSDT_CPU(image, sdt, width, height);
	gettimeofday(&end, NULL);
	fprintf(stderr, "CPU computation took: %f sec.\n", diff(start, end));
	// saveImage("results/output_cpu.png", sdt, width, height, image);

	computeSDT_GPU(image, sdt_gpu, width, height);
	saveImage(outputImage.c_str(), sdt_gpu, width, height, image);
	compareSDT(sdt, sdt_gpu, width, height); // Change the second argument to SDT computed on GPU.

	delete[] sdt;
	delete[] sdt_gpu;
	ilBindImage(0);
	ilDeleteImage(image_id);
}

void computeSDT_CPU(unsigned char * bitmap, float *sdt, int width, int height)
{
	//In the input image 'bitmap' a value of 255 represents edge pixel,
	// and a value of 127 represents interior.

	fprintf(stderr, "Computing SDT on CPU...\n");
	//Collect all edge pixels in an array
	int sz = width*height;
	int sz_edge = 0;
	for(int i = 0; i<sz; i++) if(bitmap[i] == 255) sz_edge++;
	int *edge_pixels = new int[sz_edge];
	for(int i = 0, j = 0; i<sz; i++) if(bitmap[i] == 255) edge_pixels[j++] = i;
	fprintf(stderr, "\t %d edge pixels in the image of size %d x %d\n", sz_edge, width, height);

	//Compute the SDT
	float min_dist, dist2;
	float _x, _y;
	float sign;
	float dx, dy;
	int x, y, k;
#pragma omp parallel for collapse(2) private(x, y, _x, _y, sign, dx, dy, min_dist, dist2, k) //Use multiple CPU cores to speedup
	for(y = 0; y<height; y++) // Compute SDT using brute force method
		for(x=0; x<width; x++)
		{
			min_dist = FLT_MAX;
			for(k=0; k<sz_edge; k++)
			{
				_x = edge_pixels[k] % width;
				_y = edge_pixels[k] / width;
				dx = _x - x;
				dy = _y - y;
				dist2 = dx*dx + dy*dy;
				if(dist2 < min_dist) min_dist = dist2;
			}
			sign  = (bitmap[x + y*width] >= 127)? 1.0f : -1.0f;
			sdt[x + y*width] = sign * sqrtf(min_dist);
		}
	delete[] edge_pixels;
}

void computeSDT_GPU(unsigned char* bitmap, float* sdt, int width, int height){
	cudaEvent_t begin, begin_kernel, stop_kernel, stop;
	cudaEventCreate(&begin);
	cudaEventCreate(&begin_kernel);
	cudaEventCreate(&stop_kernel);
	cudaEventCreate(&stop);
	fprintf(stderr, "Computing SDT on GPU... \n");

	cudaEventRecord(begin);
	int sz = width*height;
	int sz_edge = 0;
	for(int i = 0; i < sz; i++){
		if(bitmap[i] == 255){
			sz_edge++;
		}
	}
	int *edge_pixels = new int[sz_edge];
	for(int i = 0, j = 0; i<sz; i++) if(bitmap[i] == 255) edge_pixels[j++] = i;

	unsigned char* dev_bitmap;
	cudaMalloc( (void**) &dev_bitmap, sz*sizeof(unsigned char));
	cudaMemcpy(dev_bitmap, bitmap, sz*sizeof(unsigned char), cudaMemcpyHostToDevice);

	int* dev_edge_pixels;
	cudaMalloc( (void**) &dev_edge_pixels, sz_edge*sizeof(int));
	cudaMemcpy(dev_edge_pixels, edge_pixels, sz_edge*sizeof(int), cudaMemcpyHostToDevice);

	float* dev_sdt;
	cudaMalloc( (void**) &dev_sdt, sz*sizeof(float));

	cudaMemcpyToSymbol(dev_width, &width, sizeof(int));
	cudaMemcpyToSymbol(dev_height, &height, sizeof(int));
	cudaMemcpyToSymbol(dev_sz_edge, &sz_edge, sizeof(int));

	cudaEventRecord(begin_kernel);
	SDT_kernel<<<sz/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dev_bitmap, dev_edge_pixels, dev_sdt);
	cudaEventRecord(stop_kernel);

	cudaMemcpy(sdt, dev_sdt, sz*sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop_kernel);
	cudaEventSynchronize(stop);

	float kernelTime, totalTime; // Initialize elapsedTime;
	cudaEventElapsedTime(&kernelTime, begin_kernel, stop_kernel);
	cudaEventElapsedTime(&totalTime, begin, stop);
	printf("Time of KERNEL for SDT calculation is: %fms\n", kernelTime);
	printf("Total time for SDT calculation is: %fms\n", totalTime);

	cudaFree(dev_edge_pixels);
	cudaFree(dev_bitmap);
	cudaFree(dev_sdt);
	delete[] edge_pixels;
}

__global__ void SDT_kernel(unsigned char* bitmap, int* edge_pixels, float* sdt){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	int width = dev_width;
	int height = dev_height;
	if(i > width * height){
		return;
	}
	__shared__ int sedge_pixels[THREADS_PER_BLOCK];
	int sz_edge = dev_sz_edge;
	int x = i % width;
	int y = i / width;
	float min_dist, dist2;
	float _x, _y;
	float dx, dy;
	float sign;
	
	min_dist = FLT_MAX;
	for(int j = 0; j <= (sz_edge/THREADS_PER_BLOCK); j++){
		if((j * THREADS_PER_BLOCK + threadIdx.x) < sz_edge ){
			sedge_pixels[threadIdx.x] = edge_pixels[j * THREADS_PER_BLOCK + threadIdx.x];
		}
		__syncthreads();			
		for(int k = 0; k < THREADS_PER_BLOCK; k++){
			_x = sedge_pixels[k] % width;
			_y = sedge_pixels[k] / width;
			dx = _x - x;
			dy = _y - y;
			dist2 = dx*dx + dy*dy;
			if(dist2 < min_dist) min_dist = dist2;		
		}
	}
	sign  = (bitmap[i] >= 127)? 1.0f : -1.0f;
	sdt[i] = sign * sqrtf(min_dist);
}

void compareSDT(float *sdt1, float *sdt2, int height, int width)
{
	//Compare Mean Square Error between the two distance maps
	float mse = 0.0f;
	int sz = width*height;
	for(int i=0; i<sz; i++)
		mse += (sdt1[i] - sdt2[i])*(sdt1[i] - sdt2[i]);
	mse  = sqrtf(mse/sz);
	fprintf(stderr, "Mean Square Error (MSE): %f\n", mse);
}

ILuint loadImage(const char *filename, unsigned char ** bitmap, int &width, int &height)
{
	ILuint imageID = ilGenImage();
	ilBindImage(imageID);
	ilLoadImage(filename);
	width = ilGetInteger(IL_IMAGE_WIDTH);
	height = ilGetInteger(IL_IMAGE_HEIGHT);
	*bitmap = ilGetData();
	return imageID;
}

void saveImage(const char *filename, float * sdt, int width, int height, unsigned char *bitmap)
{
	float mind = FLT_MAX, maxd = -FLT_MAX;
	
	int sz  = width*height;
	float val;
	for(int i=0; i<sz; i++) // Find min/max of data
	{
		val  = sdt[i];
		if(val < mind) mind = val;
		if(val > maxd) maxd  = val;
	}
	unsigned char *data = new unsigned char[3*sz*sizeof(unsigned char)];
	for(int y = 0; y<height; y++) // Convert image to 24 bit
		for(int x=0; x<width; x++)
		{
			val = sdt[x + y*width];
			data[(x + y*width)*3 + 1] = 0;
			if(val<0) 
			{
				data[(x + y*width)*3 + 0] = 0;
				data[(x + y*width)*3 + 2] = 255*val/mind;
			} else {
				data[(x + y*width)*3 + 0] = 255*val/maxd;
				data[(x + y*width)*3 + 2] = 0;
			}
		}
	for(int i=0; i<sz; i++) // Mark boundary
		if(bitmap[i] == 255) {data[i*3] = 255; data[i*3+1] = 255; data[i*3+2] = 255;}

	ILuint imageID = ilGenImage();
	ilBindImage(imageID);
	ilTexImage(width, height, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, data);
	ilEnable(IL_FILE_OVERWRITE);
	iluFlipImage();
	ilSave(IL_PNG, filename);
	fprintf(stderr, "Image saved as: %s\n", filename);
}

double diff(timeval start, timeval end)
{
	double delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
         end.tv_usec - start.tv_usec) / 1.e6;
	return delta;
}
