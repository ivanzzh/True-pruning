import os
import torch
from Networks import *
from lib import *
import psutil
import matplotlib.pyplot as plt
gpu = 0
device = torch.device(gpu if torch.cuda.is_available() else "cpu")


def model_compress(model, dir_path, model_name, prune_ratio):
    torch_name = "{}_{}.pkl".format(model_name, prune_ratio)
    npy_name = "{}_compressed_{}.npy".format(model_name, prune_ratio)
    torch_path = os.path.join(dir_path, torch_name)
    npy_path = os.path.join(dir_path, npy_name)
    for name, W in model.named_parameters():
        W.data = weight_prune(W, prune_ratio)
        print(name)
    torch.save(model.state_dict(), torch_path)
    checkpoint = torch.load(torch_path)
    compressed_checkpoint = state_dict_compress(checkpoint)
    np.save(npy_path, compressed_checkpoint)


def TestCNN_compare(dir_path, model_name, prune_ratio, whether_compare, whether_rec, epochs):
    torch_name = "{}_{}.pkl".format(model_name, prune_ratio)
    npy_name = "{}_compressed_{}.npy".format(model_name, prune_ratio)
    torch_path = os.path.join(dir_path, torch_name)
    npy_path = os.path.join(dir_path, npy_name)
    test_input = np.random.uniform(0, 255, size=(4, 3, 224, 224)).astype(np.float32)
    if whether_compare:
        # test_gpu = gpuarray.to_gpu(test_input)
        test_gpu = Holder(torch.tensor(test_input, device=device))
        test_tensor = torch.tensor(test_input).to(device)
        checkpoint = torch.load(torch_path)
        state_dict = np.load(npy_path, allow_pickle=True)
        model = TestCNN().to(device)
        model.load_state_dict(checkpoint)
        re_model = SpReTestCNNTorch(test_tensor.shape[0], state_dict.item(), device, test_input.shape[2], test_input.shape[3])
        original_output = model(test_tensor)
        recons_output = re_model(test_gpu)
        # difference = original_output.cpu().detach().numpy() - recons_output.get()
        difference = original_output - recons_output
        print(original_output)
        print(recons_output)
        print(difference)
    else:
        if whether_rec:
            # test_gpu = gpuarray.to_gpu(test_input)
            test_gpu = torch.tensor(test_input, device=device)
            state_dict = np.load(npy_path, allow_pickle=True)
            re_model = SpReTestCNNTorch(test_gpu.shape[0], state_dict.item(), device, test_input.shape[2], test_input.shape[3])
            epoch_list = np.arange(epochs).tolist()
            memory_record = []
            times = 0
            memorys = 0
            for epoch in range(epochs):
                s = time.time()
                re_model(test_gpu)
                e = time.time()
                if epoch != 0:
                    times += (e - s)
                memory = show_memory_info('epoch {}'.format(epoch))
                memorys += memory
                memory_record.append(memory)
            average_time = times / (epochs - 1)
            average_memory = memorys / epochs
            plt.plot(epoch_list, memory_record, color='red', linewidth=2.0)
            plt.xlabel('epoch')
            plt.ylabel('memory usage')
            plt.title('reconstructed_{}_{}_on_'.format(model_name, prune_ratio) + device)
            plt.show()
            print("Average reconstructed {} time consumption: {:.6f}s".format(model_name, average_time))
            print("Average reconstructed {} memory usage: {:.6f}MB".format(model_name, average_memory))
        else:
            test_tensor = torch.tensor(test_input).to(device)
            checkpoint = torch.load(torch_path)
            model = TestCNN().to(device)
            model.load_state_dict(checkpoint)
            epoch_list = np.arange(epochs).tolist()
            memory_record = []
            times = 0
            memorys = 0
            for epoch in range(epochs):
                s = time.time()
                model(test_tensor)
                e = time.time()
                if epoch != 0:
                    times += (e - s)
                memory = show_memory_info('epoch {}'.format(epoch))
                memorys += memory
                memory_record.append(memory)
            average_time = times / (epochs - 1)
            average_memory = memorys / epochs
            plt.plot(epoch_list, memory_record, color='red', linewidth=2.0)
            plt.xlabel('epoch')
            plt.ylabel('memory usage')
            plt.title('original_{}_{}_on_'.format(model_name, prune_ratio) + device)
            plt.show()
            print("Average original {} time consumption: {:.6f}s".format(model_name, average_time))
            print("Average original {} memory usage: {:.6f}MB".format(model_name, average_memory))


def AlexNet_compare(dir_path, model_name, prune_ratio, whether_compare, whether_rec, epochs):
    torch_name = "{}_{}.pkl".format(model_name, prune_ratio)
    npy_name = "{}_compressed_{}.npy".format(model_name, prune_ratio)
    torch_path = os.path.join(dir_path, torch_name)
    npy_path = os.path.join(dir_path, npy_name)
    test_input = np.random.uniform(0, 255, size=(4, 3, 227, 227)).astype(np.float32)
    if whether_compare:
        # test_gpu = gpuarray.to_gpu(test_input)
        test_gpu = torch.tensor(test_input, device=device)
        test_tensor = torch.tensor(test_input).to(device)
        checkpoint = torch.load(torch_path)
        state_dict = np.load(npy_path, allow_pickle=True)
        model = AlexNet().to(device)
        model.load_state_dict(checkpoint)
        re_model = SpReAlexNetTorch(test_tensor.shape[0], state_dict.item(), device, test_input.shape[2], test_input.shape[3])
        original_output = model(test_tensor)
        recons_output = re_model(test_gpu)
        # difference = original_output.cpu().detach().numpy() - recons_output.get()
        difference = original_output - recons_output
        print(original_output.device)
        print(recons_output.device)
        # print(type(original_output))
        # print(original_output)
        # print(recons_output)
        print(difference)
    else:
        if whether_rec:
            # test_gpu = gpuarray.to_gpu(test_input)
            test_gpu = torch.tensor(test_input, device=device)
            state_dict = np.load(npy_path, allow_pickle=True)
            re_model = SpReAlexNetTorch(test_gpu.shape[0], state_dict.item(), device, test_input.shape[2], test_input.shape[3])
            epoch_list = np.arange(epochs).tolist()
            memory_record = []
            times = 0
            memorys = 0
            for epoch in range(epochs):
                s = time.time()
                re_model(test_gpu)
                e = time.time()
                if epoch != 0:
                    times += (e - s)
                memory = show_memory_info('epoch {}'.format(epoch))
                memorys += memory
                memory_record.append(memory)
            average_time = times / (epochs - 1)
            average_memory = memorys / epochs
            plt.plot(epoch_list, memory_record, color='red', linewidth=2.0)
            plt.xlabel('epoch')
            plt.ylabel('memory usage')
            plt.title('reconstructed_{}_{}_on_cuda'.format(model_name, prune_ratio))
            plt.show()
            print("Average reconstructed {} time consumption: {:.6f}s".format(model_name, average_time))
            print("Average reconstructed {} memory usage: {:.6f}MB".format(model_name, average_memory))
        else:
            test_tensor = torch.tensor(test_input).to(device)
            checkpoint = torch.load(torch_path)
            model = AlexNet().to(device)
            model.load_state_dict(checkpoint)
            epoch_list = np.arange(epochs).tolist()
            memory_record = []
            times = 0
            memorys = 0
            for epoch in range(epochs):
                s = time.time()
                model(test_tensor)
                e = time.time()
                if epoch != 0:
                    times += (e - s)
                memory = show_memory_info('epoch {}'.format(epoch))
                memorys += memory
                memory_record.append(memory)
            average_time = times / (epochs - 1)
            average_memory = memorys / epochs
            plt.plot(epoch_list, memory_record, color='red', linewidth=2.0)
            plt.xlabel('epoch')
            plt.ylabel('memory usage')
            plt.title('original_{}_{}_on_'.format(model_name, prune_ratio))
            plt.show()
            print("Average original {} time consumption: {:.6f}s".format(model_name, average_time))
            print("Average original {} memory usage: {:.6f}MB".format(model_name, average_memory))


def VGG16_compare(dir_path, model_name, prune_ratio, whether_compare, whether_rec, epochs):
    torch_name = "{}_{}.pkl".format(model_name, prune_ratio)
    npy_name = "{}_compressed_{}.npy".format(model_name, prune_ratio)
    torch_path = os.path.join(dir_path, torch_name)
    npy_path = os.path.join(dir_path, npy_name)
    test_input = np.random.uniform(0, 255, size=(16, 3, 227, 227)).astype(np.float32)
    if whether_compare:
        # test_gpu = gpuarray.to_gpu(test_input)
        test_gpu = torch.tensor(test_input, device=device)
        test_tensor = torch.tensor(test_input).to(device)
        checkpoint = torch.load(torch_path)
        state_dict = np.load(npy_path, allow_pickle=True)
        model = VGG16().to(device)
        model.load_state_dict(checkpoint)
        re_model = SpReVGG16Torch(test_tensor.shape[0], state_dict.item(), device, test_input.shape[2], test_input.shape[3])
        original_output = model(test_tensor)
        recons_output = re_model(test_gpu)
        # difference = original_output.cpu().detach().numpy() - recons_output.get()
        difference = original_output - recons_output
        print(original_output.device)
        print(recons_output.device)
        # print(type(original_output))
        # print(original_output)
        # print(recons_output)
        print(difference)
    else:
        if whether_rec:
            # test_gpu = gpuarray.to_gpu(test_input)
            test_gpu = torch.tensor(test_input, device=device)
            state_dict = np.load(npy_path, allow_pickle=True)
            re_model = SpReVGG16Torch(test_gpu.shape[0], state_dict.item(), device, test_input.shape[2], test_input.shape[3])
            epoch_list = np.arange(epochs).tolist()
            memory_record = []
            times = 0
            memorys = 0
            for epoch in range(epochs):
                s = time.time()
                re_model(test_gpu)
                e = time.time()
                if epoch != 0:
                    times += (e - s)
                memory = show_memory_info('epoch {}'.format(epoch))
                memorys += memory
                memory_record.append(memory)
            average_time = times / (epochs-1)
            average_memory = memorys / epochs
            plt.plot(epoch_list, memory_record, color='red', linewidth=2.0)
            plt.xlabel('epoch')
            plt.ylabel('memory usage')
            plt.title('reconstructed_{}_{}_on_cuda'.format(model_name, prune_ratio))
            plt.show()
            plt.close()
            print("Average reconstructed {} time consumption: {:.6f}s".format(model_name, average_time))
            print("Average reconstructed {} memory usage: {:.6f}MB".format(model_name, average_memory))
        else:
            test_tensor = torch.tensor(test_input).to(device)
            checkpoint = torch.load(torch_path)
            model = VGG16().to(device)
            model.load_state_dict(checkpoint)
            epoch_list = np.arange(epochs).tolist()
            memory_record = []
            times = 0
            memorys = 0
            for epoch in range(epochs):
                s = time.time()
                model(test_tensor)
                e = time.time()
                if epoch != 0:
                    times += (e - s)
                memory = show_memory_info('epoch {}'.format(epoch))
                memorys += memory
                memory_record.append(memory)
            average_time = times / (epochs-1)
            average_memory = memorys / epochs
            plt.plot(epoch_list, memory_record, color='red', linewidth=2.0)
            plt.xlabel('epoch')
            plt.ylabel('memory usage')
            plt.title('original_{}_{}_on_cuda'.format(model_name, prune_ratio))
            plt.show()
            plt.close()
            print("Average original {} time consumption: {:.6f}s".format(model_name, average_time))
            print("Average original {} memory usage: {:.6f}MB".format(model_name, average_memory))


dir_path = './checkpoint'
# model = TestCNN()
# model_name = 'TestCNN'
# model_compress(model, dir_path, model_name, 0.95)
# TestCNN_compare(dir_path, model_name, 0.9, True, True, 20)


# model = AlexNet()
# model_name = 'AlexNet'
# # model_compress(model, dir_path, model_name, 0.95)
# AlexNet_compare(dir_path, model_name, 0.9, False, True, 20)


# model = VGG16()
# model_name = 'VGG16'
# # model_compress(model, dir_path, model_name, 0.5)
# VGG16_compare(dir_path, model_name, 0.9, False, True, 20)
#

model = ResNet50()
model_name = 'ResNet'
model_compress(model, dir_path, model_name, 0.8)


# for i in range(1, 10):
#     print("layer{}.0.bottleneck.0.weight".format(i))