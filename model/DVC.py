import torch
from torch.utils import data
from model.utils_DVC.buffer.buffer import Buffer
import torch.optim as optim
# from agents.base import ContinualLearner
# from continuum.data_utils import dataset_transform
import torch.nn as nn
from model.utils_DVC.setup_elements import transforms_match, input_size_match
from model.utils_DVC.utils import maybe_cuda, AverageMeter
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from model.utils_DVC.loss import agmax_loss, cross_entropy_loss

class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        
        self.net = args.net
        self.buffer = Buffer(self.net, args)
        self.mem_size = args.n_memories
        # self.agent = params.agent
        params = args
        self.dl_weight = params.dl_weight

        # self.eps_mem_batch = params.eps_mem_batch
        # self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(32, 32), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
           ColorJitter(0.4, 0.4, 0.4, 0.1),
           RandomGrayscale(p=0.2)

        )
        self.L2loss = torch.nn.MSELoss()
        self.n_iters = args.n_iter
        self.opt = optim.SGD(self.parameters(), args.lr)
        
        # self.old_labels = []
        # self.new_labels = []
        # self.task_seen = 0
        # self.kd_manager = KdManager()
        # self.error_list = []
        # self.new_class_score = []
        # self.old_class_score = []
        # self.fc_norm_new = []
        # self.fc_norm_old = []
        # self.bias_norm_new = []
        # self.bias_norm_old = []
        # self.lbl_inv_map = {}
        # self.class_task_map = {}
      
    # def before_train(self, x_train, y_train):
    #     new_labels = list(set(y_train.tolist()))
    #     self.new_labels += new_labels
    #     for i, lbl in enumerate(new_labels):
    #         self.lbl_inv_map[lbl] = len(self.old_labels) + i

    #     for i in new_labels:
    #         self.class_task_map[i] = self.task_seen  
            
    def forward(self, x, t):
        output, _, _, _ = self.net(x)
        return output
            
    def observe(self, batch_x, t, batch_y):
        try:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            
        except:
            print(batch_x.shape)
            print(batch_y)
            exit()
        try:
            batch_x = batch_x.view(-1, 3, 32, 32)
        except:
            batch_x = batch_x.view(-1, 28, 28)
        batch_x_aug = self.transform(batch_x)
        for iter_i in range(self.n_iters):
            y = self.net(batch_x, batch_x_aug)
            z, zt, _,_ = y
            ce = cross_entropy_loss(z, zt, batch_y, label_smoothing=0)

            agreement_loss, dl = agmax_loss(y, batch_y, dl_weight=self.dl_weight)
            loss  = ce + agreement_loss + dl
            # _, pred_label = torch.max(z, 1)
            # correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
            # update tracker
            # acc_batch.update(correct_cnt, batch_y.size(0))
            # losses_batch.update(loss, batch_y.size(0))
            # backward
            self.opt.zero_grad()
            loss.backward()

                    # mem update
            if  True: #self.params.retrieve == 'MGI':
                mem_x, mem_x_aug, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

            else:
                mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                if mem_x.size(0) > 0:
                    mem_x_aug = self.transform(mem_x)

            if mem_x.size(0) > 0:
                mem_x = maybe_cuda(mem_x, self.cuda)
                mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                mem_y = maybe_cuda(mem_y, self.cuda)
                y = self.net(mem_x, mem_x_aug)
                z, zt, _,_ = y
                ce = cross_entropy_loss(z, zt, mem_y, label_smoothing=0)
                agreement_loss, dl = agmax_loss(y, mem_y, dl_weight=self.dl_weight)
                loss_mem = ce  + agreement_loss + dl

                # update tracker
                # losses_mem.update(loss_mem, mem_y.size(0))
                # _, pred_label = torch.max(z, 1)
                # correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
                # acc_mem.update(correct_cnt, mem_y.size(0))

                loss_mem.backward()
            self.opt.step()

                # update mem
            self.buffer.update(batch_x, batch_y)

        # self.after_train()


 