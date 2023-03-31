import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import tenseal as ts
from torch import nn
from tqdm import tqdm
import logging
import datetime
import os
import psutil


class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = x * x
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x


def conv2d_plain_encryption(x, encryption_conv, kernel_size, stride):
    conv_weight = encryption_conv.get('weight')
    conv_bias = encryption_conv.get('bias')
    unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)
    x = unfold(x.reshape((1, 28, 28)))
    # plain_x = [plain_x[i].tolist() for i in range(plain_x.shape[0])]
    plain_x = ts.plain_tensor(x)

    enc_channels = []
    for weight, bias in zip(conv_weight, conv_bias):
        y = weight.mm(plain_x) + bias
        enc_channels.append(y)

    return enc_channels


def fc_encryption_encryption(encryption_x, encryption_fc):
    fc_weight = encryption_fc.get('weight')
    fc_bias = encryption_fc.get('bias')

    enc_channels = []
    for weight, bias in zip(fc_weight, fc_bias):
        y = encryption_x.dot(weight) + bias
        enc_channels.append(y)

    return enc_channels


class EncConvNet:
    def __init__(self, torch_nn, context):
        conv1_weight = [ts.ckks_vector(context, torch_nn.conv1.weight.data[i].reshape(-1).tolist()) for i in
                        range(torch_nn.conv1.weight.data.shape[0])]
        conv1_bias = [ts.ckks_vector(context, [bias]) for bias in torch_nn.conv1.bias.data.tolist()]
        self.conv1 = {'weight': conv1_weight, 'bias': conv1_bias}

        fc1_weight = [ts.ckks_vector(context, torch_nn.fc1.weight.data[i].tolist()) for i in
                      range(torch_nn.fc1.weight.data.shape[0])]
        fc1_bias = [ts.ckks_vector(context, [bias]) for bias in torch_nn.fc1.bias.data.tolist()]
        self.fc1 = {'weight': fc1_weight, 'bias': fc1_bias}

        fc2_weight = [ts.ckks_vector(context, torch_nn.fc2.weight.data[i].tolist()) for i in
                      range(torch_nn.fc2.weight.data.shape[0])]
        fc2_bias = [ts.ckks_vector(context, [bias]) for bias in torch_nn.fc2.bias.data.tolist()]
        self.fc2 = {'weight': fc2_weight, 'bias': fc2_bias}

    def forward(self, plain_x):
        # conv layer
        output_tensor_1 = conv2d_plain_encryption(plain_x, self.conv1, kernel_size=7, stride=3)
        output_tensor_1 = ts.CKKSVector.pack_vectors(output_tensor_1)
        # square activation
        output_tensor_1.square_()
        # fc1 layer
        output_tensor_2 = fc_encryption_encryption(output_tensor_1, self.fc1)
        output_tensor_2 = ts.CKKSVector.pack_vectors(output_tensor_2)
        # square activation
        output_tensor_2.square_()
        # fc2 layer
        output_tensor = fc_encryption_encryption(output_tensor_2, self.fc2)
        output_tensor = ts.CKKSVector.pack_vectors(output_tensor)
        return output_tensor

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride, logging):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for idx, (data, target) in enumerate(tqdm(test_loader)):
        enc_output = enc_model(data)
        # Decryption of result
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)

        # compute loss
        loss = criterion(output, target)
        test_loss += loss.item()

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1

        if (idx + 1) % 50 == 0:
            logging.info(f'[{int(np.sum(class_total))}|{len(test_loader)}]: {100 * np.sum(class_correct) / np.sum(class_total): .2f}%')

    # calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    logging.info(f'Test Loss: {test_loss:.6f}\n')

    for label in range(10):
        logging.info(
            f'Test Accuracy of {label}: {100 * class_correct[label] / class_total[label]: .2f}% '
            f'({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})'
        )

    logging.info(
        f'\nTest Accuracy (Overall): {100 * np.sum(class_correct) / np.sum(class_total): .2f}% '
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
    )


if __name__ == '__main__':
    # logging setting
    os.makedirs("./result", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s => %(message)s')
    now = datetime.datetime.now()
    log_filename = os.path.join(f"./result", now.strftime('%Y-%m-%d_%H-%M-%S') + '.log')
    file_handler = logging.FileHandler(filename=log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s => %(message)s')
    formatter.datefmt = '%Y-%m-%d %H:%M:%S'
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    logging.info('start')

    # model and dataset
    print(u'Memory usage of the current process: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    model = torch.load("./checkpoint/best.pt")
    torch.manual_seed(73)
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()

    # Load one element at a time
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    # required for encoding
    kernel_shape = model.conv1.kernel_size
    stride = model.conv1.stride[0]
    # CKKS setting
    poly_mod_degree = 16384
    coeff_mod_bit_sizes = [59, 39, 39, 39, 39, 39, 39, 39, 39, 59]
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    print(u'Memory usage of the current process: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    # test
    enc_model = EncConvNet(model, context)
    enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride, logging)