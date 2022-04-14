import numpy as np
import pycuda.autoinit  # 以自动的方式对pycuda进行初始
from pycuda.compiler import SourceModule  # 编译kernel函数的类
import pycuda.gpuarray as gpuarray

# 通过字符串定义kernel函数
from torch.version import cuda

kernel_code = r"""
void __global__ add(const float *x, const float *y, float *z, int *m)
{
    const int n = threadIdx.y*m[0]+threadIdx.x;
    const int n1 = threadIdx.y*blockDim.x +threadIdx.x;
    z[n] = x[n1] + y[n1];
}

void __global__ convolution(const float *image, const float *weight, const int *index, const int *image_range, 
float *output_array ) { 
    int output_index = blockIdx.y * gridDim.x + blockIdx.x;
    int offset = image_range[blockIdx.x];
    int weight_index = blockIdx.y * 3 + threadIdx.x;
    int image_index = index[weight_index] + offset;
    output_array[output_index] = output_array[output_index] + image[image_index] * weight[weight_index];
}

void __global__ convolution1(const float *image, const float *weight, const int *index, const int *image_range, 
float *output_array ) { 
    int output_index = blockIdx.y * gridDim.y * blockDim.x + blockIdx.x;
    int offset = image_range[blockIdx.x];
    for(int i = 0; i<3; ++i){
        int weight_index = blockIdx.y * 3 + i;
        int image_index = index[weight_index] + offset;
        output_array[output_index] += image[image_index] * weight[weight_index];
    }
}

void __global__ convolution2(const float *image, const float *weight, const int *index, const int *image_range, 
float *output_array, const int *filter_length, const int *start_point, int a) { 
    int output_index = blockIdx.y * gridDim.x + blockIdx.x;
    if (output_index == 3) printf("%d\n", a);
    int offset = image_range[blockIdx.x];
    for(int i = 0; i<filter_length[blockIdx.y]; i++){
        int location = start_point[blockIdx.y] + i;
        // printf("%f ", weight[location]);
        int image_index = index[location] + offset;
        output_array[output_index] += image_index * weight[location];
    }
}

void __global__ print_all(const float *image){
    printf("%d: %f ", image[threadIdx.x]);
}
"""
mod = SourceModule(kernel_code)
conv = mod.get_function("convolution1")
conv1 = mod.get_function("convolution2")
print_all = mod.get_function("print_all")
image = np.arange(0, 25).reshape([1, 1, 5, 5])
print(image)
filter = np.array([[1, 2, 0],
                   [3, 4, 0],
                   [5, 6, 7]])
print(filter)
filter_index = np.array([[2, 11, 0],
                         [0, 11, 0],
                         [1, 10, 22]], dtype=np.int32)
col_index = filter_index % 10
row_index = filter_index // 10
image_index = row_index * image.shape[3] + col_index
H_weight = np.array([0, 2])
W_weight = np.array([[0], [2]]) * 5
image_range = (H_weight + W_weight).reshape(-1)
output_array = np.zeros([3, 2, 2])
image_gpu = gpuarray.to_gpu(image.astype(np.float32))
print_all(image_gpu, grid=(1, 1, 1), block=(25, 1, 1))
filter_gpu = gpuarray.to_gpu(filter.astype(np.float32))
index_gpu = gpuarray.to_gpu(image_index.astype(np.int32))
image_range_gpu = gpuarray.to_gpu(image_range.astype(np.int32))
output_array_gpu = gpuarray.to_gpu(output_array.astype(np.float32))
conv(image_gpu, filter_gpu, index_gpu, image_range_gpu, output_array_gpu, grid=(4, 3, 1), block=(1, 1, 1))
print(output_array_gpu.get())
a = filter > 0
total = np.sum(np.sum(a, axis=0), axis=0)
print(total)
filter_1d = np.zeros(total)
index_1d = np.zeros(total, dtype=np.int32)
filter_length = np.zeros(filter.shape[0], dtype=np.int32)
start_point = np.zeros(filter.shape[0], dtype=np.int32)
for i in range(filter.shape[0]):
    temp = filter[i, :]
    temp_index = image_index[i, :]
    index = temp > 0
    length = np.sum(index)
    filter_length[i] = length
    if i != 0:
        start_point[i] = start_point[i - 1] + filter_length[i - 1]
    filter_1d[start_point[i]: start_point[i] + length] = temp[index]
    index_1d[start_point[i]: start_point[i] + length] = temp_index[index]
print(filter_1d)
print(filter_length)
print(start_point)
print(index_1d)
filter_1d_gpu = gpuarray.to_gpu(filter_1d.astype(np.float32))
filter_length_gpu = gpuarray.to_gpu(filter_length.astype(np.int32))
start_point_gpu = gpuarray.to_gpu(start_point.astype(np.int32))
index_1d_gpu = gpuarray.to_gpu(index_1d.astype(np.int32))
output_array1 = np.zeros([3, 2, 2])
output_array1_gpu = gpuarray.to_gpu(output_array1.astype(np.float32))
print(output_array1_gpu.shape)
print(image_range_gpu.get())
a = np.int32(1)
conv1(image_gpu, filter_1d_gpu, index_1d_gpu, image_range_gpu, output_array1_gpu, filter_length_gpu, start_point_gpu, a,
      grid=(4, 3, 1), block=(1, 1, 1))
print(output_array1_gpu.get())
# a = output_array1_gpu.get()
# a[0, 0, 0] = 0
output_array1_gpu.get()[0, 0, 0] = 0
print(output_array1_gpu)


