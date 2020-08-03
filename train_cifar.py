'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torchvision import *

import numpy as np
import os
import argparse
import random

from models import *
from utils import progress_bar
from solver.lr_scheduler import WarmupMultiStepLR

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

#optimizer
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_decay_1', default=40, type=int)
parser.add_argument('--lr_decay_2', default=50, type=int)
parser.add_argument('--wd', default=0.0005, type=float)
parser.add_argument('--warmup_epoch', default=3, type=int)
#resume
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--resume_proxy', default=None, type=str)
#architecture
parser.add_argument('--model_name', default='ResNet18', type=str, help='model name')
parser.add_argument('--num_classes', default=10, type=int)
#attack
parser.add_argument('--step_size', default=0.007*255, type=float)
parser.add_argument('--train_attack_eps', default=8.0, type=float)
parser.add_argument('--train_attack_iters', default=7, type=int)
parser.add_argument('--test_attack_type', default="pgd", type=str, help='pgd, cw, fgsm')
parser.add_argument('--test_attack_eps', default=8.0, type=float)
parser.add_argument('--test_attack_iters', default=40, type=int)
#api
parser.add_argument('--test_api_eps', default=14, type=float)
parser.add_argument('--test_api_iters', default=8, type=int)
parser.add_argument('--train_api_eps', default=14, type=float)
parser.add_argument('--train_api_iters', default=8, type=int)
#others
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--test_batch_size', default=48, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--api_start_epoch', default=50, type=int)
parser.add_argument('--out_path', default="./", type=str)
parser.add_argument('--mode', default='api_net_train', type=str)
parser.add_argument('--random_seed', default=random.randint(0,100), type=int)

args = parser.parse_args()

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.model_name == 'ResNet18':
    torch.random.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    net = ResNet18(num_classes=args.num_classes, input_channel=3)
    proxy_net = ResNet18(num_classes=args.num_classes, input_channel=3)
else:
    assert(0), "Error: un-support model"

def make_optimizer(model):
    params = []
    for key, value in model.named_parameters():

        if not value.requires_grad:
            continue

        lr = args.lr
        weight_decay = args.wd

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = getattr(torch.optim, 'SGD')(params, momentum=0.9)
    return optimizer

optimizer = make_optimizer(net)
scheduler = WarmupMultiStepLR(optimizer, (args.lr_decay_1, args.lr_decay_2), 0.1, 1.0/3.0, args.warmup_epoch, 'linear')

def api_net_train(args, net, epoch, summary_writer):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss_adv = train_loss_adv_api = 0
    correct_adv = correct_adv_api = 0
    total = 0
    for batch_idx, (x_natural, y) in enumerate(trainloader):

        x_natural, y = x_natural.to(device), y.to(device)

        batch_size, ch, h, w = x_natural.shape
        num_classes = len(classes)

        attack_params = AdvParams()
        attack_params.type         = "pgd"
        attack_params.iters        = args.train_attack_iters
        attack_params.eps          = args.train_attack_eps
        attack_params.step_size    = args.step_size
        attack_params.direction    ='leave'
        attack_params.random_start = 'eps'

        api_params = AdvParams()
        api_params.type            = 'pgd'
        api_params.iters           = args.train_api_iters
        api_params.eps             = args.train_api_eps
        api_params.step_size       = args.step_size
        api_params.direction       = 'towards'
        api_params.random_start    = 'constant'

        
        if epoch < args.api_start_epoch:
            x_adv = net(x_natural, mode="get_adv", y=y, attack_params=attack_params)
            optimizer.zero_grad()

            outputs_adv = net(x_adv, mode="forward")
            loss_adv = F.cross_entropy(outputs_adv, y)
            loss = loss_adv 
            loss.backward()

            optimizer.step()

            total += y.size(0)
            train_loss_adv += loss_adv.item()
            _, predicted = outputs_adv.max(1)
            correct_adv += predicted.eq(y).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Lr: %.3f | Loss_adv: %.3f | Acc_adv: %.3f%% (%d)'
                % (  scheduler.get_lr()[0] , train_loss_adv/(batch_idx+1), 100.*correct_adv/total, total))

        else:
            x_adv = net(x_natural, mode="get_adv_fbda", y=y, attack_params=attack_params, api_params=api_params) 
           
            z = torch.zeros(num_classes, batch_size, ch, h, w).to(x_adv.device).to(x_adv.dtype)

            for c in range(num_classes):
                yc = c * torch.ones(batch_size).to(x_adv.device).to(torch.long)
                z[c] = net(x_adv, mode="get_adv", y=yc, attack_params=api_params)

            outputs_adv_api = torch.zeros(batch_size, num_classes).to(x_adv.device).to(x_adv.dtype)
            optimizer.zero_grad()

            for c in range(num_classes):
                label_outputs = net(z[c], mode="forward")
                label_outputs = F.log_softmax(label_outputs, dim=1)
                outputs_adv_api[:, c] = label_outputs[:, c]

            loss_adv_api = F.cross_entropy(outputs_adv_api, y)

            outputs_natural = net(x_natural, mode="forward")
            loss_natural = F.cross_entropy(outputs_natural, y)

            outputs_adv = net(x_adv, mode="forward")
            loss_robust = F.cross_entropy(outputs_adv, y)

            loss = 0*loss_robust + loss_adv_api 
            loss.backward()

            optimizer.step()

            total += y.size(0)
            train_loss_adv_api += loss_adv_api.item()
            _, predicted = outputs_adv_api.max(1)
            correct_adv_api += predicted.eq(y).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Lr: %.3f Loss_adv_api: %.3f Acc_adv_api: %.3f%%(%d)'
                % (  scheduler.get_lr()[0] , train_loss_adv_api/(batch_idx+1), 100.*correct_adv_api/total, total))

    scheduler.step()

def trades(args, net, epoch, summary_writer):
    
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct_adv  = 0
    total = 0
    for batch_idx, (x_natural, y) in enumerate(trainloader):

        x_natural, y = x_natural.to(device), y.to(device)

        batch_size, c, h, w = x_natural.shape
        num_classes = len(classes)

        attack_params = AdvParams()
        attack_params.type         = "kl"
        attack_params.iters        = args.train_attack_iters
        attack_params.eps          = args.train_attack_eps
        attack_params.step_size    = args.step_size
        attack_params.random_start = 'eps'

        x_adv = net(x_natural, mode="get_adv", y=y, attack_params=attack_params)
        optimizer.zero_grad()

        outputs_natural = net(x_natural, mode="forward")
        outputs_adv = net(x_adv, mode="forward")
        loss_natural = F.cross_entropy(outputs_natural, y)
        criterion_kl = nn.KLDivLoss(reduction="sum")
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(outputs_adv, dim=1),
                                                        F.softmax(outputs_natural, dim=1))
        loss = loss_natural + 6 * loss_robust
        loss.backward()

        optimizer.step()

        total += y.size(0)
        train_loss += loss.item()
        _, predicted = outputs_adv.max(1)
        correct_adv += predicted.eq(y).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Lr: %.3f | Loss: %.3f | Acc_adv: %.3f%% (%d)'
            % (  scheduler.get_lr()[0] , train_loss/(batch_idx+1), 100.*correct_adv/total, total))

    scheduler.step()

def api_net_test(args, net, proxy_net, epoch, best_acc, summary_writer, test_num=10000):
    
    test_loss = 0
    correct_api_clean = correct_api_fbda = correct_api_bpda=0
    total = 0
    
    for batch_idx, (x_clean, y) in enumerate(testloader):

        x_clean, y = x_clean.to(device), y.to(device)
        batch_size = len(x_clean)
        net.eval()

        if batch_idx*batch_size > test_num:
            break

        test_attack_params = AdvParams()
        test_attack_params.type         = args.test_attack_type
        test_attack_params.iters        = args.test_attack_iters
        test_attack_params.eps          = args.test_attack_eps
        test_attack_params.step_size    = args.step_size
        test_attack_params.direction    ='leave'
        test_attack_params.random_start = 'eps'

        test_api_params = AdvParams()
        test_api_params.type            = 'pgd'
        test_api_params.iters           = args.test_api_iters
        test_api_params.eps             = args.test_api_eps
        test_api_params.step_size       = args.step_size
        test_api_params.direction       = 'towards'
        test_api_params.random_start    = 'constant'

        # test clean img
        with torch.no_grad():
            outputs_api_clean = net(x_clean, mode="forward_api", api_params=test_api_params)
            
            _, predicted = outputs_api_clean.max(1)
            total += y.size(0)
            correct_api_clean += predicted.eq(y).sum().item()

        #forward backward approximation test
        x_fbda = proxy_net(x_clean, mode="get_adv_fbda", y=y, attack_params=test_attack_params, api_params=test_api_params)
        
        net.eval()
        outputs_api_fbda = net(x_fbda, mode="forward_api", api_params=test_api_params)
        _, predicted = outputs_api_fbda.max(1)
        correct_api_fbda += predicted.eq(y).sum().item()
        
        #backward approximation test
        x_bpda = proxy_net(x_clean, mode="get_adv_bpda", y=y, attack_params=test_attack_params, api_params=test_api_params)
        
        net.eval()
        outputs_api_bpda = net(x_bpda, mode="forward_api", api_params=test_api_params)
        _, predicted = outputs_api_bpda.max(1)
        correct_api_bpda += predicted.eq(y).sum().item()

        worst_case_acc = min(correct_api_fbda/total, correct_api_bpda/total)

        progress_bar(batch_idx, len(testloader), 'acc_api_clean: %.3f%% worst_case_acc %.3f%% (%d)'
            % (100.*correct_api_clean/total, 100.*worst_case_acc, total))

    # Save checkpoint.
    acc = worst_case_acc
    model_path=os.path.join(args.out_path, "{}_epoch{}.pth".format(args.model_name, epoch))
    state = {
        'net': net.state_dict(),
        'acc': acc, 
        'epoch': epoch,
    }
    torch.save(state, model_path)
    
    if acc > best_acc:
        model_path=os.path.join(args.out_path, "{}.pth".format(args.model_name))
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc, 
            'epoch': epoch,
        }
        torch.save(state, model_path)
        best_acc = acc

    return worst_case_acc


def test(args, net, proxy_net, epoch, best_acc, summary_writer,test_num=10000):
    
    test_loss = 0
    correct_clean = correct_adv = 0
    total = 0
    
    for batch_idx, (x_clean, y) in enumerate(testloader):

        x_clean, y = x_clean.to(device), y.to(device)
        batch_size = len(x_clean)
        net.eval()

        if batch_idx*batch_size > test_num:
            break

        test_attack_params = AdvParams()
        test_attack_params.type         = args.test_attack_type
        test_attack_params.iters        = args.test_attack_iters
        test_attack_params.eps          = args.test_attack_eps
        test_attack_params.step_size    = args.step_size
        test_attack_params.direction    ='leave'
        test_attack_params.random_start = 'eps'

        # test clean img
        with torch.no_grad():
            outputs_clean = net(x_clean, mode="forward")
            
            _, predicted = outputs_clean.max(1)
            total += y.size(0)
            correct_clean += predicted.eq(y).sum().item()

        x_adv = proxy_net(x_clean, mode="get_adv", y=y, attack_params=test_attack_params)
        
        net.eval()
        outputs_adv = net(x_adv, mode="forward")
        _, predicted = outputs_adv.max(1)
        correct_adv += predicted.eq(y).sum().item()
        
        progress_bar(batch_idx, len(testloader), 'acc_clean: %.3f%% acc_adv %.3f%% (%d)'
            % (100.*correct_clean/total, 100.*correct_adv/total, total))

    worst_case_acc = correct_adv/total

    # Save checkpoint.
    acc = worst_case_acc
    model_path=os.path.join(args.out_path, "{}_epoch{}.pth".format(args.model_name, epoch))
    state = {
        'net': net.state_dict(),
        'acc': acc, 
        'epoch': epoch,
    }
    torch.save(state, model_path)
    
    if acc > best_acc:
        model_path=os.path.join(args.out_path, "{}.pth".format(args.model_name))
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc, 
            'epoch': epoch,
        }
        torch.save(state, model_path)
        best_acc = acc

    return worst_case_acc


def set_params(net, resume_model_path, data_parallel=False):
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(resume_model_path), 'Error: ' + resume_model_path + 'checkpoint not found!'
    checkpoint = torch.load(resume_model_path)
    state_dict = checkpoint['net']
    from collections import OrderedDict
    sdict = OrderedDict()
    for key in state_dict.keys():
        new_key = key.split('module.')[-1]
        if data_parallel:
            new_key = 'module.'+new_key
        sdict[new_key]=state_dict[key]
    net.load_state_dict(sdict)
    return net



if __name__ == "__main__":
    print(args)
    
    tensorboard_path=os.path.join(args.out_path, "TensorBoardSummary")
    summary_writer=SummaryWriter(tensorboard_path)

    # set net
    net = net.to(device)
    proxy_net = proxy_net.to(device)
    
    # Resume
    if args.resume != None:
        net = set_params(net, args.resume)
    if args.resume_proxy != None:
        proxy_net = set_params(proxy_net, args.resume_proxy)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        proxy_net = torch.nn.DataParallel(proxy_net)
        cudnn.benchmark = True

    #set random seed
    torch.random.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    if args.mode == 'api_net_train':
        for epoch in range(args.epochs):
            api_net_train(args, net, epoch, summary_writer)
            if epoch %5 == 0:
                if epoch < args.api_start_epoch:
                    test(args, net, net, epoch, best_acc, summary_writer)
                else:
                    api_net_test(args, net, net, epoch, best_acc, summary_writer, test_num=2000)

        api_net_test(args, net, net, epoch, best_acc, summary_writer)

    elif args.mode == 'trades':
        for epoch in range(args.epochs):
            trades(args, net, epoch, summary_writer)
            test(args, net,net, 0, best_acc, summary_writer)

    elif args.mode == 'api_net_test':
        api_net_test(args, net, net, 0, best_acc, summary_writer)

    elif args.mode == 'test':
        test(args, net, net, 0, best_acc, summary_writer)

    summary_writer.close()