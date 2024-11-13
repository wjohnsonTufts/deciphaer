#!/usr/bin/env python
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import RNAseq_Dataset
from dataloader import Morph_Features_Dataset
from model import FC_Autoencoder, FC_Classifier, VAE, FC_VAE, Simple_Classifier

import sklearn
import os
import argparse
import numpy as np
import imageio
import csv

# adapted from pytorch/examples/vae and uhlerlab/cross-modal-autoencoders

torch.manual_seed(1)

#============ PARSE ARGUMENTS =============

def setup_args():

    options = argparse.ArgumentParser()

    # filename, save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir", default="custom_train_rna_morph_save")
    options.add_argument('--save-freq', action="store", dest="save_freq", default=100, type=int)
    options.add_argument('-ptr', action="store", dest="pretrained_file_rna", default= None)
    options.add_argument('-ptm', action="store", dest="pretrained_file_morph", default= None)
    options.add_argument('-ds', action="store", dest="datadir", default = "kfold_validation")
    options.add_argument('-fnm', action="store", dest="filename_morph", default= "train_4.csv")
    options.add_argument('-lbm', action="store", dest="labels_morph", default= "train_4_labels.csv")
    options.add_argument('-fnr', action="store", dest="filename_rna", default= "rna_train_4.csv")
    options.add_argument('-lbr', action="store", dest="labels_rna", default= "rna_train_4_labels.csv")    
                         
    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=50, type=int)
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-5, type=float)
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-5, type=float)
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=1000, type=int)
    options.add_argument('-wd', '--weight-decay', action="store", dest="weight_decay", default=0, type=float)
    options.add_argument('--conditional', action="store_true", default = True)
    options.add_argument('--conditional-adv', action="store_true")
    options.add_argument('-alpha_morph', action="store", default=0.1, type=float) #Weight for the total reconstruction loss
    options.add_argument('-alpha_rna', action="store", default=0.1, type=float) #Weight for the total reconstruction loss
    options.add_argument('-beta', action="store", default=1000, type=float) #Weight for the classifier loss
    options.add_argument('-lamb', action="store", default=0.00000001, type=float) #Weight for the KL-divergence
    options.add_argument('-nz', action="store", default=128, type=int)
    options.add_argument('-gamm', action="store", default=1, type=int)#Weight for the modality classifier

    # gpu options
    options.add_argument('-gpu', '--use-gpu', action="store_true", dest="use_gpu")

    return options.parse_args()


args = setup_args()
if not torch.cuda.is_available():
    args.use_gpu = False

os.makedirs(args.save_dir, exist_ok=True)

#============= TRAINING INITIALIZATION ==============

# load data
morph_feature_dataset = Morph_Features_Dataset(filename=args.filename_morph, labels=args.labels_morph) 
morph_feature_loader = DataLoader(morph_feature_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
rna_dataset = RNAseq_Dataset(filename=args.filename_rna, labels=args.labels_rna) 
rna_loader = DataLoader(rna_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
print("Data loaded")

# initialize autoencoder
netRNA = FC_VAE(nz=args.nz, n_input=rna_dataset.nfeatures)
netMorph = FC_VAE(nz=args.nz, n_input=morph_feature_dataset.nfeatures)

if (args.pretrained_file_morph != None):
    netMorph.load_state_dict(torch.load(args.pretrained_file_morph))
    print("Pre-trained morph model loaded from %s" % args.pretrained_file_morph)
if (args.pretrained_file_rna != None):
    netRNA.load_state_dict(torch.load(args.pretrained_file_rna))
    print("Pre-trained RNA model loaded from %s" % args.pretrained_file_rna)

if args.conditional_adv: 
    netClf = FC_Classifier(nz=args.nz+10)
    assert(not args.conditional)
else:
    netClf = FC_Classifier(nz=args.nz)

if args.conditional:
    netCondClf = Simple_Classifier(nz=args.nz, n_out=morph_feature_dataset.nclass)

if args.use_gpu:
    netRNA.cuda()
    netMorph.cuda()
    netClf.cuda()
    if args.conditional:
        netCondClf.cuda()                   
                         
# setup optimizer
opt_netRNA = optim.Adam(list(netRNA.parameters()), lr=args.learning_rate_AE)
opt_netClf = optim.Adam(list(netClf.parameters()), lr=args.learning_rate_D, weight_decay=args.weight_decay)
opt_netMorph = optim.Adam(list(netMorph.parameters()), lr=args.learning_rate_AE)

lab1 = np.array(rna_dataset.labels)
lab2 = np.array(morph_feature_dataset.labels)
if args.conditional:
    opt_netCondClf = optim.Adam(list(netCondClf.parameters()), lr=args.learning_rate_AE)
    class_weights=sklearn.utils.class_weight.compute_class_weight('balanced',classes = np.unique(morph_feature_dataset.labels),y = np.concatenate((lab1, lab2)))
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    
# loss criteria
criterion_reconstruct = nn.MSELoss()
criterion_classify_label = nn.CrossEntropyLoss(class_weights)
criterion_classify_modality = nn.CrossEntropyLoss()
# Setup logger
with open(os.path.join(args.save_dir, 'log.txt'), 'w') as f:
    print(args, file=f)
    print(netRNA, file=f)
    print(netMorph, file=f)
    print(netClf, file=f)
    if args.conditional:
        print(netCondClf, file=f)
# Setup logger csv        
os.makedirs(args.save_dir, exist_ok=True)

lossCols = ["epoch", "rna_recon", "image_recon", "ae_clf", "clf", "clf_class", "clf_accuracy_RNA", "clf_accruacy_Image"]
with open(os.path.join(args.save_dir, "loss.csv"), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(lossCols)
# define helper train functions

def compute_KL_loss(mu, logvar):
    if args.lamb>0:
        KLloss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return args.lamb * KLloss
    return 0

def train_autoencoders(rna_inputs, image_inputs, rna_class_labels=None, image_class_labels=None):
   
    netRNA.train()
    netMorph.train()
    netClf.eval()
    if args.conditional:
        netCondClf.train()
    
    # process input data
    rna_inputs, image_inputs = Variable(rna_inputs), Variable(image_inputs)

    if args.use_gpu:
        rna_inputs, image_inputs = rna_inputs.cuda(), image_inputs.cuda()

    # reset parameter gradients
    netRNA.zero_grad()
    netMorph.zero_grad()


    # forward pass
    rna_recon, rna_latents, rna_mu, rna_logvar = netRNA(rna_inputs)
    image_recon, image_latents, image_mu, image_logvar = netMorph(image_inputs)
    if args.conditional_adv:
        rna_class_labels, image_class_labels = rna_class_labels.cuda(), image_class_labels.cuda()
        rna_scores = netClf(torch.cat((rna_latents, rna_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
        image_scores = netClf(torch.cat((image_latents, image_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
    else:
        rna_scores = netClf(rna_latents)
        image_scores = netClf(image_latents)
    rna_labels = torch.zeros(rna_scores.size(0),).long()
    image_labels = torch.ones(image_scores.size(0),).long()
    if args.conditional:
        rna_class_scores = netCondClf(rna_latents)
        image_class_scores = netCondClf(image_latents)
    
    if args.use_gpu:
        rna_labels, image_labels = rna_labels.cuda(), image_labels.cuda()
        if args.conditional:
            rna_class_labels, image_class_labels = rna_class_labels.cuda(), image_class_labels.cuda()
    # compute losses
    rna_recon_loss = criterion_reconstruct(rna_inputs, rna_recon)
    image_recon_loss = criterion_reconstruct(image_inputs, image_recon)
    kl_loss = compute_KL_loss(rna_mu, rna_logvar) + compute_KL_loss(image_mu, image_logvar)
    clf_loss = 0.5*criterion_classify_modality(rna_scores, image_labels) + 0.5*criterion_classify_modality(image_scores, rna_labels)
    loss = args.alpha_rna * rna_recon_loss + args.alpha_morph * image_recon_loss + args.gamm * clf_loss + kl_loss

    if args.conditional:
        clf_class_loss = 0.5*criterion_classify_label(rna_class_scores, rna_class_labels) + 0.5*criterion_classify_label(image_class_scores, image_class_labels)
        loss += args.beta*clf_class_loss

    # backpropagate and update model
    loss.backward()
    opt_netRNA.step()
    opt_netMorph.step()
    if args.conditional:
        opt_netCondClf.step()


    summary_stats = {'rna_recon_loss': rna_recon_loss.item()*rna_scores.size(0), 'image_recon_loss': image_recon_loss.item()*image_scores.size(0), 
            'clf_loss': clf_loss.item()*(rna_scores.size(0)+image_scores.size(0))}
    
    if args.conditional:
        summary_stats['clf_class_loss'] = clf_class_loss.item()*(rna_scores.size(0)+image_scores.size(0))

    return summary_stats

def train_classifier(rna_inputs, image_inputs, rna_class_labels=None, image_class_labels=None):
    
    netRNA.eval()
    netMorph.eval()
    netClf.train()

    # process input data
    rna_inputs, image_inputs = Variable(rna_inputs), Variable(image_inputs)

    if args.use_gpu:
        rna_inputs, image_inputs = rna_inputs.cuda(), image_inputs.cuda()
    
    # reset parameter gradients
    netClf.zero_grad()

    # forward pass
    _, rna_latents, _, _ = netRNA(rna_inputs)
    _, image_latents, _, _ = netMorph(image_inputs)
    if args.conditional_adv:
        rna_class_labels, image_class_labels = rna_class_labels.cuda(), image_class_labels.cuda()
        rna_scores = netClf(torch.cat((rna_latents, rna_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
        image_scores = netClf(torch.cat((image_latents, image_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
    else:
        rna_scores = netClf(rna_latents)
        image_scores = netClf(image_latents)
    rna_labels = torch.zeros(rna_scores.size(0),).long()
    image_labels = torch.ones(image_scores.size(0),).long()
    
    if args.use_gpu:
        rna_labels, image_labels = rna_labels.cuda(), image_labels.cuda()

    # compute losses
    clf_loss = 0.5*criterion_classify_modality(rna_scores, rna_labels) + 0.5*criterion_classify_modality(image_scores, image_labels)
    loss = clf_loss

    # backpropagate and update model
    loss.backward()
    opt_netClf.step()

    summary_stats = {'clf_loss': clf_loss*(rna_scores.size(0)+image_scores.size(0)), 'rna_accuracy': accuracy(rna_scores, rna_labels), 'rna_n_samples': rna_scores.size(0),
            'image_accuracy': accuracy(image_scores, image_labels), 'image_n_samples': image_scores.size(0)}

    return summary_stats

def accuracy(output, target):
    pred = output.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).float().sum().item()
    return correct


### main training loop
for epoch in range(args.max_epochs):
    if epoch%10 ==0:
        print(epoch)
    recon_rna_loss = 0
    recon_image_loss = 0
    clf_loss = 0
    clf_class_loss = 0
    AE_clf_loss = 0

    n_rna_correct = 0
    n_rna_total = 0
    n_morph_correct = 0
    n_morph_total = 0

    for idx, (rna_samples, image_samples) in enumerate(zip(rna_loader, morph_feature_loader)):
        rna_inputs = rna_samples['tensor']
        image_inputs = image_samples['tensor']

        if args.conditional or args.conditional_adv:
            rna_labels = rna_samples['labels']
            image_labels = image_samples['labels']
            out = train_autoencoders(rna_inputs, image_inputs, rna_labels, image_labels)
        else:
            out = train_autoencoders(rna_inputs, image_inputs)

        recon_rna_loss += out['rna_recon_loss']
        recon_image_loss += out['image_recon_loss']
        AE_clf_loss += out['clf_loss']

        if args.conditional:
            clf_class_loss += out['clf_class_loss']
        
        if args.conditional_adv:
            out = train_classifier(rna_inputs, image_inputs, rna_labels, image_labels)
        else:
            out = train_classifier(rna_inputs, image_inputs)

        clf_loss += out['clf_loss']
        n_rna_correct += out['rna_accuracy']
        n_rna_total += out['rna_n_samples']
        n_morph_correct += out['image_accuracy']
        n_morph_total += out['image_n_samples']

    recon_rna_loss /= n_rna_total
    clf_loss /= n_rna_total+n_morph_total
    AE_clf_loss /= n_rna_total+n_morph_total

    if args.conditional:
        clf_class_loss /= n_rna_total + n_morph_total

    with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
        print('Epoch: ', epoch, ', rna recon loss: %.8f' % float(recon_rna_loss), ', image recon loss: %.8f' % float(recon_image_loss),
                ', AE clf loss: %.8f' % float(AE_clf_loss), ', clf loss: %.8f' % float(clf_loss), ', clf class loss: %.8f' % float(clf_class_loss),
                ', clf accuracy RNA: %.4f' % float(n_rna_correct / n_rna_total), ', clf accuracy Morph: %.4f' % float(n_morph_correct / n_morph_total), file=f)
    
    #Add losses to csv table
    with open(os.path.join(args.save_dir, "loss.csv"), 'a') as f:
        writer = csv.writer(f)
        listedLoss = [epoch,float(recon_rna_loss), float(recon_image_loss),float(AE_clf_loss),float(clf_loss),float(clf_class_loss),float(n_rna_correct / n_rna_total),float(n_morph_correct / n_morph_total)]
        writer.writerow(map(lambda x: x, listedLoss))

    # save model
    if epoch % args.save_freq == 0:
        torch.save(netRNA.cpu().state_dict(), os.path.join(args.save_dir,"netRNA_%s.pth" % epoch))
        torch.save(netMorph.cpu().state_dict(), os.path.join(args.save_dir,"netMorph_%s.pth" % epoch))
        torch.save(netClf.cpu().state_dict(), os.path.join(args.save_dir,"netClf_%s.pth" % epoch))
        if args.conditional:
            torch.save(netCondClf.cpu().state_dict(), os.path.join(args.save_dir,"netCondClf_%s.pth" % epoch))

    if args.use_gpu:
        netRNA.cuda()
        netClf.cuda()
        netMorph.cuda()
        if args.conditional:
            netCondClf.cuda()
