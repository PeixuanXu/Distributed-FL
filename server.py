#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

from connect.ConnectHandler_server import ConnectHandler as cc

if __name__ == '__main__':
    # data = server.receiveData()
    # print(data)
    # server.sendData(0, '123')
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    print("start")
    server = cc(args.num_users, '10.24.116.58', 6688)
    print("end")

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=False, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=False, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    
    # net_glob_cpu = net_glob.to('cpu')
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    # w_glob = net_glob_cpu.state_dict()

    # training
    loss_train = []

    if args.all_clients: 
        print("Aggregation over all clients")
        args.frac = 1

        # w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        # if not args.all_clients:
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            data = {'w_glob': w_glob, 'idxs': dict_users[idx]}
            server.sendData(idx, data)
        for idx in idxs_users:
            item = server.receiveData()
            # print(item)
            w = item[0]['w']
            loss = item[0]['loss']
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # copy weight to net_glob
    # net_glob = net_glob.to('cuda')
    net_glob.load_state_dict(w_glob)
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    print(loss_train)
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    # dataset_train = dataset_train.to('cuda')
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))




# num = 10
# nums = [0, 0]

# while True:
#     server.sendData(0, num)
#     print("Send to client 0 successfully! num={}".format(num))
#     server.sendData(1, num)
#     print("Send to client 1 successfully! num={}".format(num))
#     item1 = server.receiveData()
#     nums[item1[1]] = int(item1[0])
#     item2 = server.receiveData()
#     nums[item2[1]] = int(item2[0])
#     print("Receive from client 0 successfully! num0={}".format(nums[0]))
#     print("Receive from client 1 successfully! num1={}".format(nums[1]))

#     num = nums[0] + nums[1]
    
    # print("idx={}\t{}".format(idx, item))
    # idx += 1
    # data = server.receiveData()
    # print("data={}".format(data))
    # print("data[0]={}, data[1]={}".format(data[0], data[1]))
