# coding: utf-8
from model import *
from dataset import *
import torch

import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau




# 训练过程 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class train():
    def __init__(self, image, mask, name='model', hidden=32, gpu=True, epoch=50,
                 batch_size=16, kernel_size=25, fuyangben_weight=10, f_val=None, lr=0.005, auto_lr_decay=True, lr_decay_patience=4, lr_decay_factor=0.1, save_after=48,
                 plot_every=1, verbose=True, directory='./'):

        assert image.shape == mask.shape
        data_train = dataset(image, mask, part='train', f_val=f_val)
        data_val = dataset(image, mask, part='val', f_val=f_val)
        self.TrainLoader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8)
        self.ValLoader = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=8)
        self.shape = image.shape[1]
        self.name = name
        self.fuyangben_weight = fuyangben_weight

        if gpu:
            self.dtype = torch.cuda.FloatTensor
            self.dint = torch.cuda.ByteTensor
            self.network = nn.DataParallel(UNet1Sigmoid(1,1,hidden,kernel_size=kernel_size))
            self.network.type(self.dtype)
        else:
            self.dtype = torch.FloatTensor
            self.dint = torch.ByteTensor
            self.network = UNet1Sigmoid(1,1,hidden,kernel_size=kernel_size)
            self.network.type(self.dtype)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        if auto_lr_decay:
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=lr_decay_factor, patience=lr_decay_patience,
                                                  cooldown=2, verbose=True, threshold=0.005)
        else:
            self.lr_scheduler = self._void_lr_scheduler
        self.BCELoss = BCELosswithLogits()
        self.validation_loss = []
        self.epoch_mask = 0
        self.save_after = save_after
        self.n_epochs = epoch
        self.every = plot_every
        self.directory = directory
        self.verbose = verbose
        self.mode0_complete = False
        self.tqdm = tqdm
        
    def set_input(self, img0, mask):
        """
        :param img0: input image
        :param mask: CR mask
        :param ignore: loss mask
        :return: None
        """
        self.img0 = Variable(img0.type(self.dtype)).view(-1, 1, self.shape) 
        self.mask = Variable(mask.type(self.dtype)).view(-1, 1, self.shape) 

    def validate_mask(self):
        """
        :return: validation loss. print TPR and FPR at threshold = 0.5.
        """
        lmask = 0
        count = 0
        metric = np.zeros(4)
        pdt_mask__at_epoch_i = []
        for i_dat, dat in enumerate(self.ValLoader): 
            n = dat[0].shape[0]
            count += n
            self.set_input(*dat)
            self.pdt_mask = self.network(self.img0)
            loss = self.backward_network() 
            lmask += float(loss.detach()) * n
            pdt_mask_np = self.pdt_mask.reshape(-1, self.shape).detach().cpu().numpy()
            metric += maskMetric(pdt_mask_np, dat[1].numpy()) 
            for j in pdt_mask_np: pdt_mask__at_epoch_i.append(j)
        if self.epoch_mask == self.n_epochs:
            np.save('pdt_mask_'+self.name+'.npy', np.array(pdt_mask__at_epoch_i))
        lmask /= count
        TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]

        return (lmask)

    def train_mask(self):
        lmask = 0
        count = 0
        metric = np.zeros(4)
        for i, dat in enumerate(self.TrainLoader):
            n = dat[0].shape[0]
            count += n
            self.set_input(*dat)
            self.pdt_mask = self.network(self.img0)
            self.optimizer.zero_grad()
            loss = self.backward_network()
            loss.backward()
            self.optimizer.step()

            lmask += float(loss.detach()) * n
            metric += maskMetric(self.pdt_mask.reshape(-1, self.shape).detach().cpu().numpy(), dat[1].numpy())
        lmask /= count
        TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]

    def train_initial(self, epochs=None):
        if epochs is None:
            epochs = self.n_epochs
        self.network.train()
        for epoch in range(epochs):

            print_train_mask = True
            if print_train_mask == True:
                self.train_mask()
            else:
                for t, dat in enumerate(self.TrainLoader):
                    self.optimize_network(dat)
                
            self.epoch_mask += 1
            val_loss = self.validate_mask()
            self.validation_loss.append(val_loss)
            if self.verbose:
                if self.epoch_mask > (self.n_epochs-2):
                    print('validation_loss = %.4f' % (self.validation_loss[-1]))

            self.lr_scheduler.step(self.validation_loss[-1])
            if self.verbose:
                #print('')
                pass

    def set_to_eval(self):
        self.network.eval()

    def optimize_network(self, dat):
        self.set_input(*dat)
        self.pdt_mask = self.network(self.img0)
        self.optimizer.zero_grad()
        loss = self.backward_network()
        loss.backward()
        self.optimizer.step()

    def backward_network(self):
        loss = self.BCELoss(self.pdt_mask, self.mask, self.fuyangben_weight)
        return loss

    def plot_loss(self):
        import matplotlib.pyplot as plt
        """ plot validation loss vs. epoch
        :return: None
        """
        plt.figure(figsize=(10,5))
        plt.plot(range(self.epoch_mask), self.validation_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Validation loss')
        plt.show()

    def save(self):
        """ save trained network parameters to date_model_name_epoch*.pth
        :return: None
        """
        time = datetime.datetime.now()
        time = str(time)[:10]

        torch.save(self.network.state_dict(), self.directory + self.name+'.pth')

# 训练过程 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# 混淆矩阵 -----------------------------------------
def maskMetric(PD, GT):
        if len(PD.shape) == 2:
            PD = PD.reshape(1, *PD.shape)
        if len(GT.shape) == 2:
            GT = GT.reshape(1, *GT.shape)
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(GT.shape[0]):
            P = GT[i].sum()
            TP += (PD[i][GT[i] == 1] == 1).sum()
            TN += (PD[i][GT[i] == 0] == 0).sum()
            FP += (PD[i][GT[i] == 0] == 1).sum()
            FN += (PD[i][GT[i] == 1] == 0).sum()
        return np.array([TP, TN, FP, FN])
#---------------------------------------------------

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",default=50,type=int)
    parser.add_argument("--name",default='test_'+nowtime,type=str)
    parser.add_argument("--input",type=str)
    parser.add_argument("--outdir",default="./",type=str)
    return parser

def main(args):
    print(args)
    f_val=0.01
    npz = np.load(args.input)
    from scipy.ndimage import uniform_filter
    pre_flux = npz['flux']
    filtered_flux = uniform_filter(pre_flux, size=3, mode='reflect',axes=1)
    max_vals = np.max(filtered_flux, axis=1)
    flux = np.array([pre_flux[i,:] /max_vals[i] for i in range(0,len(max_vals))])

    red_line = npz['gauss']+ npz['pl'] + npz['Fe']
    mask = np.array([red_line[i,:] /max_vals[i] for i in range(0,len(max_vals))])
    output_filename = args.name
    output_dir = args.outdir
    trainer = train(flux, mask,  epoch=args.epoch,
                    hidden=32, batch_size=16, kernel_size=25, fuyangben_weight=10, f_val=f_val,
                    name=output_filename, directory=output_dir,plot_every=1)
    trainer.train_initial()
    trainer.save()

if __name__ == '__main__':
    nowtime = datetime.datetime.now().strftime("%y%m%d_%H%M")
    args = get_args_parser().parse_args()
    main(args)
    print("train complete!")