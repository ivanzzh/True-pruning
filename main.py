import os
import torch
from Networks import *
from lib import *
import psutil
import matplotlib.pyplot as plt


def model_compress(model, dir_path, model_name, prune_ratio):
    torch_name = "{}_{}.pkl".format(model_name, prune_ratio)
    npy_name = "{}_compressed_{}.npy".format(model_name, prune_ratio)
    torch_path = os.path.join(dir_path, torch_name)
    npy_path = os.path.join(dir_path, npy_name)
    for name, W in model.named_parameters():
        W.data = weight_prune(W, prune_ratio)
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
    device = 'cuda'
    if whether_compare:
        test_gpu = gpuarray.to_gpu(test_input)
        test_tensor = torch.tensor(test_input).to(device)
        checkpoint = torch.load(torch_path)
        state_dict = np.load(npy_path, allow_pickle=True)
        model = TestCNN().to(device)
        model.load_state_dict(checkpoint)
        re_model = SpReTestCNN(state_dict.item(), device, test_input.shape[2], test_input.shape[3])
        original_output = model(test_tensor)
        recons_output = re_model(test_gpu)
        difference = original_output.cpu().detach().numpy() - recons_output.get()
        print(original_output)
        print(recons_output)
        print(difference)
    else:
        if whether_rec:
            test_gpu = gpuarray.to_gpu(test_input)
            state_dict = np.load(npy_path, allow_pickle=True)
            re_model = SpReTestCNN(state_dict.item(), device, test_input.shape[2], test_input.shape[3])
            epoch_list = np.arange(epochs).tolist()
            memory_record = []
            times = 0
            memorys = 0
            for epoch in range(epochs):
                s = time.time()
                re_model(test_gpu)
                e = time.time()
                times += (e - s)
                memory = show_memory_info('epoch {}'.format(epoch))
                memorys += memory
                memory_record.append(memory)
            average_time = times / epochs
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
                times += (e - s)
                memory = show_memory_info('epoch {}'.format(epoch))
                memorys += memory
                memory_record.append(memory)
            average_time = times / epochs
            average_memory = memorys / epochs
            plt.plot(epoch_list, memory_record, color='red', linewidth=2.0)
            plt.xlabel('epoch')
            plt.ylabel('memory usage')
            plt.title('original_{}_{}_on_'.format(model_name, prune_ratio) + device)
            plt.show()
            print("Average original {} time consumption: {:.6f}s".format(model_name, average_time))
            print("Average original {} memory usage: {:.6f}MB".format(model_name, average_memory))


model = TestCNN()
dir_path = './checkpoint'
model_name = 'TestCNN'
# model_compress(model, dir_path, model_name, 0.95)
TestCNN_compare(dir_path, model_name, 0.9, True, True, 20)