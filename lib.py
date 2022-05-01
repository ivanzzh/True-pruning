import os
from math import ceil

import psutil
import time
import numpy as np
import pycuda.autoinit  # 以自动的方式对pycuda进行初始
from pycuda.compiler import SourceModule  # 编译kernel函数的类
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
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


class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()

    def shape(self):
        return self.t.shape


# output_array[output_index] += bias_value[threadIdx.y];
# src = torch.cuda.IntTensor(32)
b = torch.cuda.FloatTensor(9)
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
    if (blockIdx.x == bias_index[blockIdx.z]) atomicAdd(&output_array[output_index], bias_value[blockIdx.z]);
}

void __global__ convolution_with_bias1(const float *image, const float *weight, const int *index, const int *image_range, 
float *output_array, const int *filter_length, const int *start_point, int input_image_size, int whole_image_size) { 
    int output_index = blockIdx.x * gridDim.y + blockIdx.y + threadIdx.x * whole_image_size;
    int offset = image_range[blockIdx.y] + threadIdx.x * input_image_size;
    int length = filter_length[blockIdx.x];
    if(length < 0){
        for(int i = 0; i<-length; i++){
                int location = start_point[blockIdx.x] + i;
                int image_index = index[location] + offset;
                output_array[output_index] += image[image_index] * weight[location];
        }
        output_array[output_index] += weight[start_point[blockIdx.x] - length];
    }else{
        for(int i = 0; i<length; i++){
            int location = start_point[blockIdx.x] + i;
            int image_index = index[location] + offset;
            output_array[output_index] += image[image_index] * weight[location];
        }
    }
}

void __global__ convolution(const float *image, const float *weight, const int *index, const int *image_range, 
float *output_array, const int *filter_length, const int *start_point, int input_image_size, int whole_image_size) { 
    int output_index = blockIdx.x * gridDim.y + blockIdx.y + threadIdx.x * whole_image_size;
    int offset = image_range[blockIdx.y] + threadIdx.x * input_image_size;
    for(int i = 0; i<filter_length[blockIdx.x]; i++){
        int location = start_point[blockIdx.x] + i;
        int image_index = index[location] + offset;
        output_array[output_index] += image[image_index] * weight[location];
    }
}

void __global__ add_bias(const float *bias_weight, const int *bias_index, float *output_array, 
int whole_image_size){
    int output_index = blockIdx.x * whole_image_size + bias_index[blockIdx.y] * gridDim.z + blockIdx.z;
    output_array[output_index] += bias_weight[blockIdx.y];
}
"""

mod = SourceModule(kernel_code)
conv = mod.get_function("convolution_with_bias")
conv1 = mod.get_function("convolution_with_bias1")
conv_only = mod.get_function('convolution')
add_bias = mod.get_function("add_bias")


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


class ReConv_torch(torch.nn.Module):
    def __init__(self, batches, device, in_channels, out_channels, kernel_size, stride, row_size, col_size,
                 weight_state_dict, bias_state_dict, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.batches = batches
        self.H_out = int((row_size - self.kernel_size + 2 * padding) / self.stride + 1)
        self.W_out = int((col_size - self.kernel_size + 2 * padding) / self.stride + 1)
        self.padding = padding
        self.zero_padding = torch.nn.ZeroPad2d(padding)
        # print('input size{}, output size{}'.format(row_size, self.H_out))
        self.device = device
        # self.image_size is the size of one frame of the output image
        self.image_size = self.W_out * self.H_out
        self.whole_image_size = np.int32(self.image_size * self.out_channels)
        self.row_size = row_size + 2 * padding
        self.col_size = col_size + 2 * padding
        np_input_size = self.row_size * self.col_size
        self.np_input_size = np.int32(np_input_size * in_channels)
        bias_index = bias_state_dict[0]
        bias_value = bias_state_dict[1]
        self.bias_length = len(bias_index)
        self.bias_index = torch.tensor(bias_index, dtype=torch.int32, device=device)
        self.bias_value = torch.tensor(bias_value, dtype=torch.float32, device=device)
        # 0: store index, 1: store weight, 2: store the length of each filter one by one in filter length array
        # 3 store the start point of each filter in the 1d weight
        weight_index = weight_state_dict[0]
        self.weight_value = torch.tensor(weight_state_dict[1], dtype=torch.float32, device=device)
        image_weight_index = weight_index // 100 * np_input_size + weight_index // 10 % 10 * self.col_size + weight_index % 10
        # print(weight_index)
        # print(image_weight_index)
        self.max_length = np.int32(np.max(weight_state_dict[2]))
        self.total_weight_bias_length = int(np.max(weight_state_dict[2]) + self.bias_length)
        self.image_weight_index = torch.tensor(image_weight_index, dtype=torch.int32, device=device)
        self.filter_lengths = torch.tensor(weight_state_dict[2], dtype=torch.int32, device=device)
        self.start_points = torch.tensor(weight_state_dict[3], dtype=torch.int32, device=device)
        w_range = np.expand_dims(np.arange(0, self.stride * self.W_out, self.stride, dtype=np.int32), 0)
        h_range = np.expand_dims(np.arange(0, self.stride * self.W_out, self.stride, dtype=np.int32), 1) * self.col_size
        self.image_range = torch.tensor((w_range + h_range), dtype=torch.int32, device=device)
        self.device = device
        # print(self.bias_length)

    def forward(self, images):
        # batches = images.shape[0]
        batches = self.batches
        if self.padding:
            images = self.zero_padding(images)
        output = torch.zeros(size=[batches, self.out_channels, self.W_out, self.H_out], dtype=torch.float32,
                             device=self.device)
        # self.image_size is the output image size(row * col), self.np_input_size is the input image size
        # conv(images, self.weight_value, self.image_weight_index, self.image_range, output, self.filter_lengths,
        #      self.start_points, self.np_input_size, self.bias_index, self.bias_value,
        #      grid=(self.out_channels, self.image_size, self.bias_length), block=(batches, 1, 1))

        conv_only(images, self.weight_value, self.image_weight_index, self.image_range, output, self.filter_lengths,
                  self.start_points, self.np_input_size, self.whole_image_size,
                  grid=(self.out_channels, self.image_size, 1), block=(batches, 1, 1))
        add_bias(self.bias_value, self.bias_index, output, self.whole_image_size,
                 grid=(batches, self.bias_length, self.image_size), block=(1, 1, 1))
        # conv1(images, self.weight_value, self.image_weight_index, self.image_range, output, self.filter_lengths,
        #       self.start_points, self.np_input_size, self.bias_index, self.bias_value, self.max_length,
        #       self.whole_image_size,
        #       grid=(self.out_channels, self.image_size, self.total_weight_bias_length),
        #       block=(batches, 1, 1))
        # print('grid shape:[{}, {}, {}], block shape:[{}, 1, 1]'.format(self.out_channels, self.image_size,
        #                                                                self.total_weight_bias_length, batches))
        return output


class ReConv_torch1(torch.nn.Module):
    def __init__(self, batches, device, in_channels, out_channels, kernel_size, stride, row_size, col_size,
                 weight_state_dict, bias_state_dict, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.batches = batches
        self.H_out = int((row_size - self.kernel_size + 2 * padding) / self.stride + 1)
        self.W_out = int((col_size - self.kernel_size + 2 * padding) / self.stride + 1)
        self.padding = padding
        self.zero_padding = torch.nn.ZeroPad2d(padding)
        self.device = device
        # self.image_size is the size of one frame of the output image
        self.image_size = self.W_out * self.H_out
        self.whole_image_size = np.int32(self.image_size * self.out_channels)
        self.row_size = row_size + 2 * padding
        self.col_size = col_size + 2 * padding
        np_input_size = self.row_size * self.col_size
        self.np_input_size = np.int32(np_input_size * in_channels)
        bias_index = bias_state_dict[0]
        bias_value = bias_state_dict[1]
        self.bias_length = len(bias_index)
        # 0: store index, 1: store weight, 2: store the length of each filter one by one in filter length array
        # 3 store the start point of each filter in the 1d weight
        weight_index = weight_state_dict[0]
        weight_value = weight_state_dict[1]
        image_weight_index = weight_index // 100 * np_input_size + weight_index // 10 % 10 * self.col_size + weight_index % 10
        filter_lengths = weight_state_dict[2]
        start_points = weight_state_dict[3]
        total_length = len(weight_value) + self.bias_length
        temp_index = np.zeros(total_length, dtype=np.int32)
        temp_value = np.zeros(total_length, dtype=np.float32)
        temp_start_point = np.zeros(len(start_points), dtype=np.int32)
        temp_filter_lengths = np.zeros(len(filter_lengths), dtype=np.int32)
        next_start_point = 0
        for i in range(self.out_channels):
            length = filter_lengths[i]
            start_point = start_points[i]
            temp_start_point[i] = next_start_point
            temp_value[next_start_point: next_start_point + length] = weight_value[
                                                                      start_point: start_point + length]
            temp_index[next_start_point: next_start_point + length] = image_weight_index[
                                                                      start_point: start_point + length]
            if i in bias_index:
                temp_value[next_start_point + length] = bias_value[np.where(bias_index == i)[0]]
                temp_index[next_start_point + length] = 0
                next_start_point = next_start_point + length + 1
                temp_filter_lengths[i] = -filter_lengths[i]
            else:
                next_start_point = next_start_point + length
                temp_filter_lengths[i] = filter_lengths[i]
        self.weight_value = torch.tensor(temp_value, dtype=torch.float32, device=device)
        self.image_weight_index = torch.tensor(temp_index, dtype=torch.int32, device=device)
        self.start_points = torch.tensor(temp_start_point, dtype=torch.int32, device=device)
        self.filter_lengths = torch.tensor(temp_filter_lengths, dtype=torch.int32, device=device)
        w_range = np.expand_dims(np.arange(0, self.stride * self.W_out, self.stride, dtype=np.int32), 0)
        h_range = np.expand_dims(np.arange(0, self.stride * self.W_out, self.stride, dtype=np.int32), 1) * self.col_size
        self.image_range = torch.tensor((w_range + h_range), dtype=torch.int32, device=device)
        self.device = device

    def forward(self, images):
        # batches = images.shape[0]
        batches = self.batches
        if self.padding:
            images = self.zero_padding(images)
        output = torch.zeros(size=[batches, self.out_channels, self.W_out, self.H_out], dtype=torch.float32,
                             device=self.device)
        # self.image_size is the output image size(row * col), self.np_input_size is the input image size
        conv1(images, self.weight_value, self.image_weight_index, self.image_range, output, self.filter_lengths,
              self.start_points, self.np_input_size, self.whole_image_size,
              grid=(self.out_channels, self.image_size, 1), block=(batches, 1, 1))
        return output


def state_dict_compress(state_dict):
    new_state_dict = {}
    for key in state_dict:
        index_list = []
        weight_list = []
        if key[-4:] != 'bias':
            # print(key)
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


def state_dict_compress1(state_dict):
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
            filter_weight_1d = torch.zeros(non_zero_num, dtype=torch.float32)
            filter_index_1d = torch.zeros(non_zero_num, dtype=torch.int32)
            for filter_index in range(output_channel):
                filter_weight = state_dict[key][filter_index]
                non_zero_index = filter_weight != 0
                non_zero_parameter = filter_weight[non_zero_index]
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
                filter_index_1d[start_point[filter_index]: start_point[filter_index] + length] = final_index
                # 0: store index, 1: store weight, 2: store the length of each filter one by one in filter length array
                # 3 store the start point of each filter in the 1d weight
            max_length = np.max(filter_length)
            filter_weight_2d = torch.zeros(size=[output_channel, max_length], dtype=torch.float32)
            filter_index_2d = torch.zeros(size=[output_channel, max_length], dtype=torch.float32)
            for i in range(output_channel):
                filter_weight_2d[i, 0: filter_length[i]] = filter_weight_1d[
                                                           start_point[i]: start_point[i] + filter_length[i]]
                filter_index_2d[i, 0: filter_length[i]] = filter_index_1d[
                                                          start_point[i]: start_point[i] + filter_length[i]]
            new_state_dict[key] = {0: filter_index_2d, 1: filter_weight_2d}
        else:
            for i in range(len(state_dict[key])):
                if state_dict[key][i] != 0:
                    index_list.append(i)
                    weight_list.append(state_dict[key][i].item())
            index_list = torch.tensor(np.array(index_list), )
            weight_list = torch.tensor(np.array(weight_list))
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


def conv_output_shape(row_size, col_size, kernel_size, stride, padding=0):
    row_size2 = int((row_size - kernel_size + 2 * padding) / stride + 1)
    col_size2 = int((col_size - kernel_size) / stride + 1)
    return row_size2, col_size2


def pooling_output_shape(row_size, col_size, kernel_size, stride, padding=0):
    row_size2 = ceil((row_size - kernel_size + 2 * padding) / stride + 1)
    col_size2 = ceil((col_size - kernel_size + 2 * padding) / stride + 1)
    return row_size2, col_size2
