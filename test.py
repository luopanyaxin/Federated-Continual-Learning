import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate,DatasetSplit
from models.test import test_img_local_all, test_img_local_all_WEIT
from LongLifeMethod.WEIT import Appr,LongLifeTest,LongLifeTrain
from models.Nets import RepTail,Cifar100WEIT
from torch.utils.data import DataLoader
import time

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.wd = 1e-4
    args.lambda_l1 = 1e-3
    args.lambda_l2 = 1
    args.lambda_mask = 0
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist' or 'miniimagenet' in args.dataset or 'FC100' in args.dataset or 'Corn50' in args.dataset:
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])
    else:
        if 'femnist' in args.dataset:
            train_path = './leaf-master/data/' + args.dataset + '/data/mytrain'
            test_path = './leaf-master/data/' + args.dataset + '/data/mytest'
        else:
            train_path = './leaf-master/data/' + args.dataset + '/data/train'
            test_path = './leaf-master/data/' + args.dataset + '/data/test'
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys())
        dict_users_test = list(dataset_test.keys())
        print(lens)
        print(clients)
        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    print(args.alg)
    write = SummaryWriter('./log/WEIT' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac))
    # build model
    # net_glob = get_model(args)
    net_glob = Cifar100WEIT([3,32,32])
    net_glob.train()

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    if args.alg == 'fedavg' or args.alg == 'prox':
        w_glob_keys = []

    print(total_num_layers)
    print(w_glob_keys)
    print(net_keys)
    if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'lg':
        num_param_glob = 0
        num_param_local = 0
        for key in net_glob.state_dict().keys():
            num_param_local += net_glob.state_dict()[key].numel()
            print(num_param_local)
            if key in w_glob_keys:
                num_param_glob += net_glob.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    task=-1
    apprs = [Appr(copy.deepcopy(net_glob).cuda(), None,lr=args.lr, nepochs=args.local_ep, args=args) for i in range(args.num_users)]
    client_aws = []
    for i,appr in enumerate(apprs):
        model_save_path = './save/WEIT/client'+str(i)+'.pt'
        d = {k:v for k,v in torch.load(model_save_path).items() if 'atten' not in k}
        aws = [v for k,v in d.items() if 'aw' in k]
        client_aws.append(aws)
        appr.model.load_state_dict(d)
    from_kb = []
    for aw in client_aws[0]:
        shape = np.concatenate([aw.shape, [int(args.num_users)]], axis=0)
        from_kb_l = np.zeros(shape)
        from_kb.append(from_kb_l)
    for c,aws in enumerate(client_aws):
        for i,aw in enumerate(aws):
            shape = np.concatenate([aw.shape, [int(args.num_users)]], axis=0)
            if len(shape) == 5:
                from_kb[i][:, :, :, :, c] = aw.cpu().detach().numpy()
            else:
                from_kb[i][:, :, c] = aw.cpu().detach().numpy()
    from_kb = [torch.from_numpy(i) for i in from_kb]
    for i in range(args.num_users):
        apprs[i].model.set_knowledge(9, from_kb)
    print(args.round)