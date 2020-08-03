
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdvParams(object):
    def __init__(self):
        self.type=None
        self.iters=None
        self.eps=None
        self.step_size=None
        self.target=None
        self.direction=None
        self.random_start=None
    def gatherAttrs(self):
        return ",".join("{}={}".format(k, getattr(self, k)) for k in self.__dict__.keys())
    def __str__(self):
        return "[{}:{}]".format(self.__class__.__name__, self.gatherAttrs())

def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss


class AdvModel(nn.Module):
    def __init__(self, num_classes):
        super(AdvModel, self).__init__()
        self.num_classes=num_classes

    def mode_forward(self, x):
        raise NotImplementedError

    def mode_forward_api(self, x, api_params):
        
        num_classes=self.num_classes
        batch_size, c, h, w = x.shape
        find_back_inputs = torch.zeros(num_classes, batch_size, c, h, w).to(x.device).to(x.dtype)

        for label in range(num_classes):
            fake_target = label * torch.ones(batch_size).to(x.device).to(torch.long)
            find_back_inputs[label] = self.mode_get_adv(x, fake_target, api_params)

        find_back_outputs = torch.zeros(batch_size, num_classes).to(x.device).to(x.dtype)

        for label in range(num_classes):
            label_outputs = self.mode_forward(find_back_inputs[label])
            label_outputs = F.log_softmax(label_outputs, dim=1)
            find_back_outputs[:, label] = label_outputs[:, label]

        return find_back_outputs

    def mode_get_adv(self, x, y, attack_params):

        # save context and set eval
        context_is_train = self.training
        self.eval()

        batch_size=len(x)
        x_natural = x

        epsilon = attack_params.eps/255.0
        # set random start
        if attack_params.random_start=="eps":
            x_adv = x_natural.detach() + torch.zeros_like(x_natural).uniform_(-epsilon, epsilon)
        elif attack_params.random_start=="constant":
            x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        elif attack_params.random_start=="zero":
            x_adv = x_natural.detach()
        else:
            raise NotImplementedError

        for step in range(attack_params.iters):
            step_size = attack_params.step_size/255.0
            target = y

            x_adv.requires_grad_()

            with torch.enable_grad():
                if attack_params.type == "kl":
                    logit = self.mode_forward(x)
                    logit_adv = self.mode_forward(x_adv)
                    criterion_kl = nn.KLDivLoss(reduction="sum")
                    loss = criterion_kl(F.log_softmax(logit_adv, dim=1),
                                        F.softmax(logit, dim=1))
                elif attack_params.type == "ce" or attack_params.type == "pgd" or attack_params.type == "fgsm":
                    logit_adv = self.mode_forward(x_adv)
                    if attack_params.direction == "towards":
                        loss = -F.cross_entropy(logit_adv, target, reduction='sum')
                    elif attack_params.direction == "leave":
                        loss = F.cross_entropy(logit_adv, target, reduction='sum')
                    else:
                        raise NotImplementedError
                elif attack_params.type == "cw":
                    logit_adv = self.mode_forward(x_adv)
                    cw_loss=CWLoss(num_classes=self.num_classes)
                    if attack_params.direction == "towards":
                        loss = -cw_loss(logit_adv, target)
                    elif attack_params.direction == "leave":
                        loss = cw_loss(logit_adv, target)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                # calculate grad
                grad = torch.autograd.grad(loss, [x_adv])[0]

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # resume context
        if context_is_train:
            self.train()
        else:
            self.eval()

        return x_adv.detach()

    def mode_get_adv_bpda(self, x, y, attack_params, api_params):
    
        # save context and set eval
        context_is_train = self.training
        self.eval()

        batch_size=len(x)
        x_natural = x
        
        epsilon = attack_params.eps/255.0
        # set random start
        if attack_params.random_start=="eps":
            x_adv = x_natural.detach() + torch.zeros_like(x_natural).uniform_(-epsilon, epsilon)
        elif attack_params.random_start=="constant":
            x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        elif attack_params.random_start=="zero":
            x_adv = x_natural.detach()
        else:
            raise NotImplementedError

        for step in range(attack_params.iters):
            step_size = attack_params.step_size/255.0
            target = y

            with torch.enable_grad():
                num_classes=self.num_classes
                batch_size, c, h, w = x_adv.shape
                find_back_inputs = torch.zeros(num_classes, batch_size, c, h, w).to(x_adv.device).to(x_adv.dtype)

                for label in range(num_classes):
                    fake_target = label * torch.ones(batch_size).to(x_adv.device).to(torch.long)
                    find_back_inputs[label] = self.mode_get_adv(x_adv, fake_target, api_params)

                find_back_inputs.requires_grad_()
                find_back_outputs = torch.zeros(batch_size, num_classes).to(x.device).to(x.dtype)

                for label in range(num_classes):
                    label_outputs = self.mode_forward(find_back_inputs[label])
                    label_outputs = F.log_softmax(label_outputs, dim=1)
                    find_back_outputs[:, label] = label_outputs[:, label]

                if attack_params.type == "kl":
                    logit = self.mode_forward(x)
                    logit_adv = find_back_outputs
                    criterion_kl = nn.KLDivLoss(reduction="sum")
                    loss = criterion_kl(F.log_softmax(logit_adv, dim=1),
                                        F.softmax(logit, dim=1))
                elif attack_params.type == "ce" or attack_params.type == "pgd" or attack_params.type == "fgsm":
                    logit_adv = find_back_outputs
                    if attack_params.direction == "towards":
                        loss = -F.cross_entropy(logit_adv, target, reduction='sum')
                    elif attack_params.direction == "leave":
                        loss = F.cross_entropy(logit_adv, target, reduction='sum')
                    else:
                        raise NotImplementedError
                elif attack_params.type == "cw":
                    logit_adv = find_back_outputs
                    cw_loss=CWLoss(num_classes=self.num_classes)
                    if attack_params.direction == "towards":
                        loss = -cw_loss(logit_adv, target)
                    elif attack_params.direction == "leave":
                        loss = cw_loss(logit_adv, target)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                # calculate grad
                grad = torch.autograd.grad(loss, [find_back_inputs], allow_unused=True)[0]
                grad = grad.sum(dim=0) #accumulate across dim 0 which denotes classes


            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # resume context
        if context_is_train == True:
            self.train()
        else:
            self.eval()

        return x_adv.detach()


    def forward(self, x, mode, y=None, attack_params=None, api_params=None):

        if mode == "forward":
            return self.mode_forward(x)
        elif mode == "forward_api":
            assert(api_params is not None)
            return self.mode_forward_api(x, api_params)
        elif mode == "get_adv" or mode == "get_adv_fbda": # fbda equals to attacking the underlying model directly
            assert(attack_params is not None)
            return self.mode_get_adv(x, y, attack_params)
        elif mode == "get_adv_bpda":
            assert(attack_params is not None)
            assert(api_params is not None)
            return self.mode_get_adv_bpda(x, y, attack_params, api_params)
        else:
            raise NotImplementedError
