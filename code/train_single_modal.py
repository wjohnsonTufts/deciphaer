#!/usr/bin/env python
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import RNAseq_Dataset
from dataloader import Morph_Features_Dataset
from dataloader import Metabolomics_Dataset
from model import FC_Autoencoder, FC_Classifier, VAE, FC_VAE, Simple_Classifier

import sklearn
import sys
import os
import argparse
import numpy as np
import imageio
import csv

torch.manual_seed(1)

#============ PARSE ARGUMENTS =============

def setup_args():

    options = argparse.ArgumentParser()

    # filename, save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir", default="custom_train_rna_morph_save")
    options.add_argument('--save-freq', action="store", dest="save_freq", default=100, type=int)
    options.add_argument('-ds', action="store", dest="datadir", default = "kfold_validation")
    options.add_argument('-fn', action="store", dest="filename", default= "rna_train_4.csv")
    options.add_argument('-lb', action="store", dest="labels", default= "rna_train_4_labels.csv")    
    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=50, type=int)
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-5, type=float)
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-5, type=float)
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=1000, type=int)
    options.add_argument('--conditional', action="store_true", default = True)
    options.add_argument('--conditional-adv', action="store_true")
    options.add_argument('-alpha', action="store", default=0.1, type=float) #Weight for the total reconstruction loss
    options.add_argument('-beta', action="store", default=1000, type=float) #Weight for the classifier loss
    options.add_argument('-lamb', action="store", default=0.00000001, type=float) #Weight for the KL-divergence
    options.add_argument('-nz', action="store", default=128, type=int)

    # gpu options
    options.add_argument('-gpu', '--use-gpu', action="store_true", dest="use_gpu")

    return options.parse_args()


args = setup_args()
if not torch.cuda.is_available():
    args.use_gpu = False

os.makedirs(args.save_dir, exist_ok=True)

#============= TRAINING INITIALIZATION ==============

# load data
if "rna" in args.filename.lower():
    dataset = RNAseq_Dataset(filename=args.filename, labels=args.labels)
elif "morph" in args.filename.lower():
    dataset = Morph_Features_Dataset(filename=args.filename, labels=args.labels)
elif "meta" in args.filename.lower():
    dataset = Metabolomics_Dataset(filename=args.filename, labels=args.labels)
else:
    print("Dataset not recognized")
    sys.exit()

loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
print("Data loaded")

# initialize autoencoder
net = FC_VAE(nz=args.nz, n_input=dataset.nfeatures)
# delete
print("nfeatures:", dataset.nfeatures)
print("number of classes:", dataset.nclass)


if args.conditional:
    netCondClf = Simple_Classifier(nz=args.nz, n_out=dataset.nclass)

if args.use_gpu:
    net.cuda()
    if args.conditional:
        netCondClf.cuda()                   
                         
# setup optimizer
opt_net = optim.Adam(list(net.parameters()), lr=args.learning_rate_AE)

lab1 = np.array(dataset.labels)

if args.conditional:
    opt_netCondClf = optim.Adam(list(netCondClf.parameters()), lr=args.learning_rate_AE)
    class_weights=sklearn.utils.class_weight.compute_class_weight('balanced',classes = np.unique(dataset.labels),y = lab1)
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    
# loss criteria
criterion_reconstruct = nn.MSELoss()
criterion_classify_label = nn.CrossEntropyLoss(class_weights)
# Setup logger
with open(os.path.join(args.save_dir, 'log.txt'), 'w') as f:
    print(args, file=f)
    print(net, file=f)
    if args.conditional:
        print(netCondClf, file=f)
# Setup logger csv        
os.makedirs(args.save_dir, exist_ok=True)

lossCols = ["epoch", "recon", "clf_class"]
with open(os.path.join(args.save_dir, "loss.csv"), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(lossCols)
# define helper train functions

def compute_KL_loss(mu, logvar):
    if args.lamb>0:
        KLloss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return args.lamb * KLloss
    return 0

def train_autoencoders(inputs, class_labels=None):
   
    net.train()
    if args.conditional:
        netCondClf.train()
    
    # process input data
    inputs = Variable(inputs)

    if args.use_gpu:
        inputs = inputs.cuda()

    # reset parameter gradients
    net.zero_grad()

    # forward pass
    recon, latents, mu, logvar = net(inputs)
    if args.use_gpu:
            class_labels = class_labels.cuda()
    if args.conditional:
        class_scores = netCondClf(latents)
    # compute losses
    recon_loss = criterion_reconstruct(inputs, recon)
    kl_loss = compute_KL_loss(mu, logvar)
    loss = args.alpha * recon_loss + kl_loss
    
    if args.conditional:
        clf_class_loss = criterion_classify_label(class_scores, class_labels)
        loss += args.beta*clf_class_loss

    # backpropagate and update model
    loss.backward()
    opt_net.step()
    if args.conditional:
        opt_netCondClf.step()

    summary_stats = {'recon_loss': recon_loss.item()*class_labels.size(0), 'n_samples': class_labels.size(0)}
    
    if args.conditional:
        summary_stats['clf_class_loss'] = clf_class_loss.item()*(class_scores.size(0))

    return summary_stats

def train_classifier(inputs, class_labels=None):
    
    net.eval()

    # process input data
    inputs= Variable(inputs)

    if args.use_gpu:
        inputs= inputs.cuda()
    
    # reset parameter gradients
    netClf.zero_grad()

    # forward pass
    _, latents, _, _ = net(inputs)
    if args.conditional_adv:
        class_labels = class_labels.cuda()
        scores = netClf(torch.cat((latents, class_labels.float().view(-1,1).expand(-1,10)), dim=1))
    else:
        scores = netClf(latents)

    # compute losses
    loss = clf_loss

    # backpropagate and update model
    loss.backward()
    opt_netClf.step()

    summary_stats = {'clf_loss': clf_loss*(scores.size(0)), 'accuracy': accuracy(scores, labels), 'n_samples': scores.size(0)}

    return summary_stats

def accuracy(output, target):
    pred = output.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).float().sum().item()
    return correct


### main training loop
for epoch in range(args.max_epochs):
    if epoch%10 ==0:
        print(epoch)
    recon_loss = 0
    clf_class_loss = 0
    n_total = 0

    for idx, samples in enumerate(loader):
        inputs = samples['tensor']
        
        if args.conditional or args.conditional_adv:
            labels = samples['labels']
            out = train_autoencoders(inputs, labels)
        else:
            out = train_autoencoders(inputs)

        recon_loss += out['recon_loss']
        n_total += out['n_samples']
        if args.conditional:
            clf_class_loss += out['clf_class_loss']

    recon_loss /= n_total

    if args.conditional:
        clf_class_loss /= n_total

    with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
        print('Epoch: ', epoch, ', recon loss: %.8f' % float(recon_loss), ', clf class loss: %.8f' % float(clf_class_loss), file=f)
    
    #Add losses to csv table
    with open(os.path.join(args.save_dir, "loss.csv"), 'a') as f:
        writer = csv.writer(f)
        listedLoss = [epoch,float(recon_loss),float(clf_class_loss)]
        writer.writerow(map(lambda x: x, listedLoss))

    # save model
    if epoch % args.save_freq == 0:
        if "rna" in args.filename.lower():
            torch.save(net.cpu().state_dict(), os.path.join(args.save_dir,"netRNA_%s.pth" % epoch))
        elif "morph" in args.filename.lower():
            torch.save(net.cpu().state_dict(), os.path.join(args.save_dir,"netMorph_%s.pth" % epoch))
        elif "meta" in args.filename.lower():
            torch.save(net.cpu().state_dict(), os.path.join(args.save_dir,"netMeta_%s.pth" % epoch))
        if args.conditional:
            torch.save(netCondClf.cpu().state_dict(), os.path.join(args.save_dir,"netCondClf_%s.pth" % epoch))

    if args.use_gpu:
        net.cuda()
        if args.conditional:
            netCondClf.cuda()