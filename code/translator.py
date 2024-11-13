#!/usr/bin/env python
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import RNAseq_Dataset
from dataloader import Morph_Features_Dataset
from dataloader import Metabolomics_Dataset
from model import FC_Autoencoder, FC_Classifier, VAE, FC_VAE, Simple_Classifier, FC_Classifier_three

import sklearn
import os
import argparse
import numpy as np
import pandas as pd
import imageio
import csv

def setup_args():
 
    options = argparse.ArgumentParser()
    options.add_argument('-st', '--save-type', action="store", dest="save_type", default="test")
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir", default="/cluster/tufts/aldridgelab/wjohns07/MultiOmics/AE_iterations")
    options.add_argument('-fnm', action="store", dest="filename_morph", default= "/cluster/tufts/aldridgelab/wjohns07/MultiOmics/AE_iterations/data/final_safe_dataset_pt_scaled_correction_071123/PT/PT_morph_data.csv")
    options.add_argument('-lbm', action="store", dest="labels_morph", default= "/cluster/tufts/aldridgelab/wjohns07/MultiOmics/AE_iterations/data/final_safe_dataset_pt_scaled_correction_071123/PT/PT_morph_meta_lcd_labels_ae_input.csv")
    options.add_argument('-fnr', action="store", dest="filename_rna", default= "/cluster/tufts/aldridgelab/wjohns07/MultiOmics/AE_iterations/data/final_safe_dataset_pt_scaled_correction_071123/rna_data.csv")
    options.add_argument('-lbr', action="store", dest="labels_rna", default= "/cluster/tufts/aldridgelab/wjohns07/MultiOmics/AE_iterations/data/final_safe_dataset_pt_scaled_correction_071123/rna_meta_lcd_labels_ae_input.csv")
    options.add_argument('-ptr', action="store", dest="pretrained_rna", default= "/cluster/tufts/aldridgelab/wjohns07/MultiOmics/AE_iterations/inf_final_1/netRNA_500.pth")
    options.add_argument('-ptm', action="store", dest="pretrained_morph", default= "/cluster/tufts/aldridgelab/wjohns07/MultiOmics/AE_iterations/inf_final_1/netMorph_500.pth")
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=1, type=int)
    options.add_argument('-nz', action="store", dest="nz", default=50, type = int)
 
    return options.parse_args()

args = setup_args()
n=5#number of iterations to run for translator

# load labels
morph_df = pd.read_csv(args.labels_morph)
morph_df_rows = morph_df.shape[0]
rna_df = pd.read_csv(args.labels_rna)
rna_df_rows = rna_df.shape[0]
# load datasets
morph_dataset = Morph_Features_Dataset(filename=args.filename_morph, labels=args.labels_morph)
morph_loader = DataLoader(morph_dataset, batch_size=morph_df_rows, drop_last=True, shuffle=True)
rna_dataset = RNAseq_Dataset(filename=args.filename_rna, labels=args.labels_rna) 
rna_loader = DataLoader(rna_dataset, batch_size=rna_df_rows, drop_last=True, shuffle=True)
# initialize model before loading
netMorph = FC_VAE(nz=args.nz, n_input=morph_dataset.nfeatures)
netRNA = FC_VAE(nz=args.nz, n_input=rna_dataset.nfeatures)
# load pretrained model
netMorph.load_state_dict(torch.load(args.pretrained_morph))
netRNA.load_state_dict(torch.load(args.pretrained_rna))
netMorph.eval()
netRNA.eval()
# use GPU if available
if torch.cuda.is_available():
    print('Using GPU')
    netMorph.cuda()
    netRNA.cuda()
def translate_dataset(original_loader, original_network, translation_network, translation_name):
    # arrays to store decoded var (d), class labels, and cell ids
    d_array = []
    label_array = []
    feature_loss_array = []
    cell_id_array = []
    # go through all samples in original dataset. Use translation network to decode the original data. Also generate the feature wise loss.
    for batch_idx, samples in enumerate(original_loader):
        inputs = Variable(samples['tensor'])
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        #Iterate and take the median
        iter_array = []
        feat_iter_array = []
        for i in range(0,n):
            recon_inputs, latents, mu, logvar = original_network(inputs)
            z = original_network.get_latent_var(inputs)
            d = translation_network.generate(z)
            feature_loss = torch.square(torch.sub(inputs, recon_inputs))
            iter_array.append(d)
            feat_iter_array.append(feature_loss)
        # Take the median across all iterations
        concat_trans = torch.stack(iter_array)
        concat_loss = torch.stack(feat_iter_array)
        median_iter = torch.quantile(concat_trans, q = 0.5, dim = 0)
        feat_median_iter = torch.quantile(concat_loss, q = 0.5, dim = 0)
        d_array.append(median_iter)
        feature_loss_array.append(feat_median_iter)
        cell_id = samples['cell_id']
        cell_id_array.append(cell_id)
        label = samples['labels']
        label_array.append(label)
    # flatten the label and cell_id arrays
    label_flat = list(np.concatenate(label_array).flat)
    cell_id_flat = list(np.concatenate(cell_id_array).flat)
    # concatenate tensors for d to store in an array
    decoded_data = torch.cat(d_array)
    feature_loss_data = torch.cat(feature_loss_array)
    # turn concatenated tensors into a numpy array
    decoded_data_np = decoded_data.cpu().detach().numpy()
    feature_loss_data_np = feature_loss_data.cpu().detach().numpy()
    # turn np array into a dataframe
    df = pd.DataFrame(decoded_data_np)
    df_loss = pd.DataFrame(feature_loss_data_np)
    # add labels and cell_ids to dataframe
    df.insert(0, "Cell_id", cell_id_flat)
    df.insert(1, "Label", label_flat)
    df_loss.insert(0, "Cell_id", cell_id_flat)
    df_loss.insert(1, "Label", label_flat)
    # save latent_vis csv for further processing
    df.to_csv(str(args.save_dir + "/" + translation_name + "_" + args.save_type + ".csv"), index=False)
    if (original_network == translation_network):
        df_loss.to_csv(str(args.save_dir + "/" +translation_name + "_loss" + "_" + args.save_type + ".csv"), index=False)
# morph to rna
translate_dataset(original_loader = morph_loader, original_network = netMorph, translation_network = netRNA, translation_name = "morph_to_rna")
# morph to morph
translate_dataset(original_loader = morph_loader, original_network = netMorph, translation_network = netMorph, translation_name = "morph_to_morph")
# rna to rna
translate_dataset(original_loader = rna_loader, original_network = netRNA, translation_network = netRNA, translation_name = "rna_to_rna")
# rna to morph
translate_dataset(original_loader = rna_loader, original_network = netRNA, translation_network = netMorph, translation_name = "rna_to_morph")



