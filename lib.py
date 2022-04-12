import os
import psutil
import time
import numpy as np
import pycuda.autoinit  # 以自动的方式对pycuda进行初始
from pycuda.compiler import SourceModule  # 编译kernel函数的类
import pycuda.gpuarray as gpuarray
import torch


def show_memory_info(hint):
    # 获取当前进程的进程号
    pid = os.getpid()

    # psutil 是一个获取系统信息的库
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print(f"{hint} memory used: {memory} MB ")
    return memory


src = torch.cuda.ByteTensor(8)
kernel_code = r"""
void __global__ convolution_with_bias(const float *image, const float *weight, const int *index, const int *image_range, 
float *output_array, const int *filter_length, const int *start_point, int input_image_size, const int *bias_index, 
const float * bias_value) { 
    int output_index = blockIdx.x * gridDim.y + blockIdx.y + threadIdx.x * gridDim.x * gridDim.y;
    int offset = image_range[blockIdx.y] + threadIdx.x * input_image_size;
    for(int i = 0; i<filter_length[blockIdx.x]; i++){
        int location = start_point[blockIdx.x] + i;
        int image_index = index[location] + offset;
        output_array[output_index] += image[image_index] * weight[location];
    }
    if (blockIdx.x == bias_index[threadIdx.y]) output_array[output_index] += bias_value[threadIdx.y];
}
"""

mod = SourceModule(kernel_code)
conv = mod.get_function("convolution_with_bias")


class ReConv(torch.nn.Module):
    def __init__(self, device, in_channels, out_channels, kernel_size, stride, row_size, col_size, weight_state_dict,
                 bias_state_dict):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.H_out = int((row_size - self.kernel_size) / self.stride + 1)
        self.W_out = int((col_size - self.kernel_size) / self.stride + 1)
        self.device = device
        self.image_size = self.W_out * self.H_out
        np_input_size = row_size * col_size
        self.np_input_size = np.int32(np_input_size * in_channels)
        bias_index = bias_state_dict[0]
        bias_value = bias_state_dict[1]
        self.bias_length = len(bias_index)
        self.bias_index = gpuarray.to_gpu(bias_index.astype(np.int32))
        self.bias_value = gpuarray.to_gpu(bias_value.astype(np.float32))
        # 0: store index, 1: store weight, 2: store the length of each filter one by one in filter length array
        # 3 store the start point of each filter in the 1d weight
        weight_index = weight_state_dict[0]
        self.weight_value = gpuarray.to_gpu(weight_state_dict[1].astype(np.float32))
        image_weight_index = weight_index // 100 * np_input_size + weight_index // 10 % 10 * col_size + weight_index % 10
        # print(weight_index)
        # print(image_weight_index)
        self.image_weight_index = gpuarray.to_gpu(image_weight_index)
        self.filter_lengths = gpuarray.to_gpu(weight_state_dict[2].astype(np.int32))
        self.start_points = gpuarray.to_gpu(weight_state_dict[3].astype(np.int32))
        w_range = np.expand_dims(np.arange(0, self.stride * self.W_out, self.stride, dtype=np.int32), 0)
        h_range = np.expand_dims(np.arange(0, self.stride * self.W_out, self.stride, dtype=np.int32), 1) * col_size
        self.image_range = gpuarray.to_gpu((w_range + h_range))

    def forward(self, images):
        batches = images.shape[0]
        output = gpuarray.to_gpu(np.zeros([batches, self.out_channels, self.W_out, self.H_out], dtype=np.float32))
        # self.image_size is the output image size(row * col), self.np_input_size is the input image size
        conv(images, self.weight_value, self.image_weight_index, self.image_range, output, self.filter_lengths,
             self.start_points, self.np_input_size, self.bias_index, self.bias_value,
             grid=(self.out_channels, self.image_size, 1), block=(batches, self.bias_length, 1))

        return output


def state_dict_compress(state_dict):
    new_state_dict = {}
    for key in state_dict:
        index_list = []
        weight_list = []
        if key[-4:] != 'bias':
            weight = state_dict[key]
            non_zero_index = weight != 0
            non_zero_num = torch.sum(non_zero_index).item()
            output_channel = state_dict[key].shape[0]
            filter_length = np.zeros(output_channel, dtype=np.int32)
            start_point = np.zeros(output_channel, dtype=np.int32)
            filter_weight_1d = np.zeros(non_zero_num)
            filter_index_1d = np.zeros(non_zero_num, dtype=np.int32)
            for filter_index in range(output_channel):
                filter_weight = state_dict[key][filter_index]
                non_zero_index = filter_weight != 0
                non_zero_parameter = filter_weight[non_zero_index].numpy()
                location = torch.nonzero(filter_weight, as_tuple=True)
                channel_index = location[0]
                row_index = location[1]
                col_index = location[2]
                final_index = channel_index * 100 + row_index * 10 + col_index
                length = len(non_zero_parameter)
                filter_length[filter_index] = length
                if filter_index != 0:
                    start_point[filter_index] = start_point[filter_index - 1] + filter_length[filter_index - 1]
                filter_weight_1d[start_point[filter_index]: start_point[filter_index] + length] = non_zero_parameter
                filter_index_1d[start_point[filter_index]: start_point[filter_index] + length] = final_index.numpy()
                # 0: store index, 1: store weight, 2: store the length of each filter one by one in filter length array
                # 3 store the start point of each filter in the 1d weight
                new_state_dict[key] = {0: filter_index_1d, 1: filter_weight_1d, 2: filter_length, 3: start_point}
        else:
            for i in range(len(state_dict[key])):
                if state_dict[key][i] != 0:
                    index_list.append(i)
                    weight_list.append(state_dict[key][i].item())
            index_list = np.array(index_list, dtype=np.int32)
            weight_list = np.array(weight_list, dtype=np.float32)
            new_state_dict[key] = {0: index_list, 1: weight_list}
    return new_state_dict


def weight_prune(weight, prune_ratio):
    weight = weight.cpu().detach().numpy()
    # print(weight[0, :, :, :])
    percent = prune_ratio * 100
    shape = weight.shape
    # print(shape)
    weight2d = weight.reshape(1, -1)
    shape2d = weight2d.shape
    weight_l2_norm = np.abs(weight2d)
    # print(weight_l2_norm)
    percentile = np.percentile(weight_l2_norm, percent)
    # print(percentile)
    under_threshold = weight_l2_norm <= percentile
    # print(under_threshold)
    weight2d[under_threshold] = 0
    weight = weight2d.reshape(shape)
    # print(weight[0, :, :, :])

    return torch.from_numpy(weight)