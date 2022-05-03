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
from models.test import test_img_local_all
from LongLifeMethod.Co2L import Appr,LongLifeTrain
from models.Nets import RepTailResNet18
from torch.utils.data import DataLoader
import time
if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist' or 'miniimagenet' in args.dataset or 'FC100' in args.dataset or 'SVHN' in args.dataset:
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
    # build model
    # net_glob = get_model(args)
    net_glob = RepTailResNet18()
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    if args.alg == 'fedrep' or args.alg == 'fedper':
        if 'cifar' in args.dataset or 'FC100' in args.dataset:
            # w_glob_keys = [[k] for k,_ in net_glob.feature_net.named_parameters()]
            w_glob_keys = [net_glob.weight_keys[i] for i in [j for j in range(14)]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
        else:
            w_glob_keys = net_keys[:-2]
    elif args.alg == 'lg':
        if 'cifar' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1, 2]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [2, 3]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0, 6, 7]]
        else:
            w_glob_keys = net_keys[total_num_layers - 2:]

    if args.alg == 'fedavg' or args.alg == 'prox':
        w_glob_keys = []
    if 'sent140' not in args.dataset:
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

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
    apprs = [Appr(copy.deepcopy(net_glob).to(args.device), None,lr=args.lr, nepochs=args.local_ep, args=args) for i in range(args.num_users)]
    print(args.round)
    if args.Co2Lis_train:
        for iter in range(args.epochs):
            if iter % (args.round) == 0:
                task+=1
            w_glob = {}
            loss_locals = []
            m = max(int(args.frac * args.num_users), 1)
            if iter == args.epochs:
                m = args.num_users
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            w_keys_epoch = w_glob_keys
            times_in = []
            total_len = 0
            tr_dataloaders= None
            for ind, idx in enumerate(idxs_users):
                start_in = time.time()
                if 'femnist' in args.dataset or 'sent140' in args.dataset:
                    if args.epochs == iter:
                        local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_ft]],
                                            idxs=dict_users_train, indd=indd)
                    else:
                        local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]],
                                            idxs=dict_users_train, indd=indd)
                else:
                    tr_dataloaders = DataLoader(DatasetSplit(dataset_train[task],dict_users_train[idx][:args.m_ft]),batch_size=args.local_bs, shuffle=True)
                    # if args.epochs == iter:
                    #     local = LocalUpdate(args=args, dataset=dataset_train[task], idxs=dict_users_train[idx][:args.m_ft])
                    # else:
                    #     local = LocalUpdate(args=args, dataset=dataset_train[task], idxs=dict_users_train[idx][:args.m_tr])

                    # appr = Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name)
                net_local = copy.deepcopy(net_glob)
                w_local = net_local.state_dict()
                if args.alg != 'fedavg' and args.alg != 'prox':
                    for k in w_locals[idx].keys():
                        if k not in w_glob_keys:
                            w_local[k] = w_locals[idx][k]
                net_local.load_state_dict(w_local)
                appr = apprs[idx]
                appr.set_model(net_local.to(args.device))
                appr.set_trData(tr_dataloaders)
                last = iter == args.epochs
                if 'femnist' in args.dataset or 'sent140' in args.dataset:
                    w_local, loss, indd = local.train(net=net_local.to(args.device), ind=idx, idx=clients[idx],
                                                      w_glob_keys=w_glob_keys, lr=args.lr, last=last)
                else:
                    w_local,loss, indd = LongLifeTrain(args,appr,iter,None,idx)
                loss_locals.append(copy.deepcopy(loss))
                total_len += lens[idx]
                if len(w_glob) == 0:
                    w_glob = copy.deepcopy(w_local)
                    for k, key in enumerate(net_glob.state_dict().keys()):
                        w_glob[key] = w_glob[key] * lens[idx]
                        w_locals[idx][key] = w_local[key]
                else:
                    for k, key in enumerate(net_glob.state_dict().keys()):
                        if key in w_glob_keys:
                            w_glob[key] += w_local[key] * lens[idx]
                        else:
                            w_glob[key] += w_local[key] * lens[idx]
                        w_locals[idx][key] = w_local[key]
                times_in.append(time.time() - start_in)
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)

            # get weighted average for global weights
            for k in net_glob.state_dict().keys():
                w_glob[k] = torch.div(w_glob[k], total_len)
            w_local = net_glob.state_dict()
            for k in w_glob.keys():
                w_local[k] = w_glob[k]
            if args.epochs != iter:
                net_glob.load_state_dict(w_glob)

            if iter % args.round == args.round-1:
                model_save_path = './save/Co2L/0.4/accs_Fedavg_lambda_'+str(args.lamb) +str('_') + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                    args.shard_per_user) + '_iter' + str(iter) + '_frac_'+str(args.frac)+'.pt'
                torch.save(net_glob.state_dict(), model_save_path)

    # print('Average accuracy final 10 rounds: {}'.format(accs10))
    # if args.alg == 'fedavg' or args.alg == 'prox':
    #     print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end - start)
    print(times)