# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/test.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        image, label = self.dataset[d]
        return image, label

class DatasetSplit_leaf(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label
def compute_offsets(task, nc_per_task, is_cifar=True):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2
def test_img_local(net_g, dataset, args,t,idx=None,indd=None, user_idx=-1, idxs=None,appr = None):
    net_g.eval()
    test_loss = 0
    correct = 0

    # put LEAF data into proper format
    if 'femnist' in args.dataset:
        leaf=True
        datatest_new = []
        usr = idx
        for j in range(len(dataset[usr]['x'])):
            datatest_new.append((torch.reshape(torch.tensor(dataset[idx]['x'][j]),(1,28,28)),torch.tensor(dataset[idx]['y'][j])))
    elif 'sent140' in args.dataset:
        leaf=True
        datatest_new = []
        for j in range(len(dataset[idx]['x'])):
            datatest_new.append((dataset[idx]['x'][j],dataset[idx]['y'][j]))
    else:
        leaf=False
    
    if leaf:
        data_loader = DataLoader(DatasetSplit_leaf(datatest_new,np.ones(len(datatest_new))), batch_size=args.local_bs, shuffle=False)
    else:
        data_loader = DataLoader(DatasetSplit(dataset,idxs), batch_size=args.local_bs,shuffle=False)
    if 'sent140' in args.dataset:
        hidden_train = net_g.init_hidden(args.local_bs)
    count = 0
    for idx, (data, target) in enumerate(data_loader):
        data = data.cuda()
        target = (target-10*t).cuda()
        offset1, offset2 = compute_offsets(t, 10)
        if 'sent140' in args.dataset:
            input_data, target_data = process_x(data, indd), process_y(target, indd)
            if args.local_bs != 1 and input_data.shape[0] != args.local_bs:
                break

            data, targets = torch.from_numpy(input_data).to(args.device), torch.from_numpy(target_data).to(args.device)
            net_g.zero_grad()

            hidden_train = repackage_hidden(hidden_train)
            output, hidden_train = net_g(data, hidden_train)

            loss = F.cross_entropy(output.t(), torch.max(targets, 1)[1])
            _, pred_label = torch.max(output.t(), 1)
            correct += (pred_label == torch.max(targets, 1)[1]).sum().item()
            count += args.local_bs
            test_loss += loss.item()

        else:
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            net_g.cuda()
            if appr is not None:
                appr.pernet.cuda()
                output1 = appr.pernet(data,t)[:, offset1:offset2]
                output2 = net_g(data,t)[:, offset1:offset2]
                log_probs = appr.alpha * output1 + (1-appr.alpha)*output2
            else:
                log_probs = net_g(data,t)[:, offset1:offset2]
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    if 'sent140' not in args.dataset:
        count = len(data_loader.dataset)
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    return  accuracy, test_loss

def test_img_local_all(net, args, dataset_test, dict_users_test,t,w_locals=None,w_glob_keys=None, indd=None,dataset_train=None,dict_users_train=None, return_all=False,write =None,apprs = None):
    print('test begin'+'*'*100)
    print('task '+str(t)+' finish train')
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(net)
        if w_locals is not None:
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
        net_local.eval()
        all_task_acc = 0
        all_task_loss = 0
        for u in range(t+1):

            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                a, b =  test_img_local(net_local, dataset_test, args,t,idx=dict_users_test[idx],indd=indd, user_idx=idx)
                # tot += len(dataset_test[dict_users_test[idx]]['x'])
            else:
                if apprs is not None:
                    appr = apprs[idx]
                else:
                    appr = None
                a, b = test_img_local(net_local, dataset_test[u], args,u, user_idx=idx, idxs=dict_users_test[idx],appr = appr)
                all_task_acc+=a
                all_task_loss+=b
        all_task_acc /= (t+1)
        all_task_loss /= (t+1)
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            tot += len(dataset_test[dict_users_test[idx]]['x'])
        else:
            tot += len(dict_users_test[idx])
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            acc_test_local[idx] = all_task_acc*len(dataset_test[dict_users_test[idx]]['x'])
            loss_test_local[idx] = all_task_acc*len(dataset_test[dict_users_test[idx]]['x'])
        else:
            acc_test_local[idx] = all_task_acc*len(dict_users_test[idx])
            loss_test_local[idx] = all_task_loss*len(dict_users_test[idx])
        del net_local
    
    if return_all:
        return acc_test_local, loss_test_local
    if write is not None:
        write.add_scalar('task_finish_and _agg', sum(acc_test_local)/tot, t + 1)
    return  sum(acc_test_local)/tot, sum(loss_test_local)/tot


def test_img_local_all_WEIT(appr, args, dataset_test, dict_users_test, t, w_locals=None, w_glob_keys=None, indd=None,
                       dataset_train=None, dict_users_train=None, return_all=False, write=None):
    print('test begin' + '*' * 100)
    print('task ' + str(t) + ' finish train')
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(appr[idx].model)
        net_local.eval()
        all_task_acc = 0
        all_task_loss = 0
        for u in range(t + 1):

            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                a, b = test_img_local(net_local, dataset_test, args, t, idx=dict_users_test[idx], indd=indd,
                                      user_idx=idx)
                # tot += len(dataset_test[dict_users_test[idx]]['x'])
            else:
                a, b = test_img_local(net_local, dataset_test[u], args, u, user_idx=idx, idxs=dict_users_test[idx])
                all_task_acc += a
                all_task_loss += b
        all_task_acc /= (t + 1)
        all_task_loss /= (t + 1)
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            tot += len(dataset_test[dict_users_test[idx]]['x'])
        else:
            tot += len(dict_users_test[idx])
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            acc_test_local[idx] = all_task_acc * len(dataset_test[dict_users_test[idx]]['x'])
            loss_test_local[idx] = all_task_acc * len(dataset_test[dict_users_test[idx]]['x'])
        else:
            acc_test_local[idx] = all_task_acc * len(dict_users_test[idx])
            loss_test_local[idx] = all_task_loss * len(dict_users_test[idx])
        del net_local

    if return_all:
        return acc_test_local, loss_test_local
    if write is not None:
        write.add_scalar('task_finish_and _agg', sum(acc_test_local) / tot, t + 1)
    return sum(acc_test_local) / tot, sum(loss_test_local) / tot
