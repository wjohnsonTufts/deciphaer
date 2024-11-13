#!/usr/bin/env python
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import Morph_Features_Dataset, RNAseq_Dataset, Metabolomics_Dataset
from model import FC_VAE, Simple_Classifier

import argparse
import numpy as np
import pandas as pd
import sys
import os
import imageio

def setup_args():

    options = argparse.ArgumentParser()

    # filename, save and directory options
    options.add_argument('-st', '--save-type', action="store", dest="save_type", default="unseen")
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir", default="Models_Analysis/r_100622_1/FinalizedModel")
    options.add_argument('-fnm', action="store", dest="filename_morph", default= "data_aldridge/v16_cholesterol_SFN/morphEUS_merged_subSample_100522_remaining_data.csv")
    options.add_argument('-lbm', action="store", dest="labels_morph", default= "data_aldridge/v16_cholesterol_SFN/MetaData_morphEUS_merged_subSample_100522_remaining_data_labels_ae_input.csv")
    options.add_argument('-fnr', action="store", dest="filename_rna", default= "data_aldridge/v16_cholesterol_SFN/newRNAseq_rna_100522.csv")
    options.add_argument('-lbr', action="store", dest="labels_rna", default= "data_aldridge/v16_cholesterol_SFN/newRNAseq_rna_meta_100522_labels_ae_input.csv")
    options.add_argument('-ptr', action="store", dest="pretrained_rna", default= "data_aldridge/v16_cholesterol_SFN/newRNAseq_rna_meta_100522_labels_ae_input.csv")
    options.add_argument('-ptm', action="store", dest="pretrained_morph", default= "data_aldridge/v16_cholesterol_SFN/newRNAseq_rna_meta_100522_labels_ae_input.csv")

    # training parameters
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=401, type=int)

    # hyperparameters
    options.add_argument('-nz', action="store", default=60, type=int)
    return options.parse_args()

args = setup_args()

# load datasets
morph_feature_dataset = Morph_Features_Dataset(filename=args.filename_morph, labels=args.labels_morph)
morph_feature_loader = DataLoader(morph_feature_dataset, drop_last=True, shuffle=True)
morph_feature_loader = DataLoader(morph_feature_dataset, drop_last=True, shuffle=True, batch_size = len(morph_feature_loader.dataset))

rna_dataset = RNAseq_Dataset(filename=args.filename_rna, labels=args.labels_rna) 
rna_loader = DataLoader(rna_dataset, drop_last=True, shuffle=True)
rna_loader = DataLoader(rna_dataset, drop_last=True, shuffle=True, batch_size = len(rna_loader.dataset))

# initialize model before loading
netRNA = FC_VAE(nz=args.nz, n_input=rna_dataset.nfeatures)
netMorph = FC_VAE(nz=args.nz, n_input=morph_feature_dataset.nfeatures)
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
    netMet.cuda()

# arrays to store latent var (z), class labels, and cell ids
z_array_morph = []
label_array_morph = []
cell_id_array_morph = []

z_array_rna = []
label_array_rna = []
sample_id_array_rna = []

z_array_meta = []
label_array_meta = []
sample_id_array_meta = []

# go through all of the samples, add each item to its respective array
for batch_idx, samples in enumerate(morph_feature_loader):
 
        inputs = Variable(samples['tensor'])
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        recon_inputs, latents, mu, logvar = netMorph(inputs)
        netMorph.eval()
        z = netMorph.get_latent_var(inputs)
        z_array_morph.append(z)
        cell_id = samples['cell_id']
        cell_id_array_morph.append(cell_id)
        label = samples['labels']
        label_array_morph.append(label)

for batch_idx, samples in enumerate(rna_loader):
 
        inputs = Variable(samples['tensor'])
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        recon_inputs, latents, mu, logvar = netRNA(inputs)
        netRNA.eval()
        z = netRNA.get_latent_var(inputs)
        z_array_rna.append(z)
        sample_id = samples['cell_id']
        sample_id_array_rna.append(sample_id)
        label = samples['labels']
        label_array_rna.append(label)
        

# flatten the label and cell_id arrays
label_flat_morph = list(np.concatenate(label_array_morph).flat)
cell_id_flat_morph = list(np.concatenate(cell_id_array_morph).flat)

label_flat_rna = list(np.concatenate(label_array_rna).flat)
sample_id_flat_rna = list(np.concatenate(sample_id_array_rna).flat)


# concatenate tensors for z to store in an array
latent_vis_morph = torch.cat(z_array_morph)
latent_vis_rna = torch.cat(z_array_rna)

# turn concatenated tensors into a numpy array
latent_vis_np_morph = latent_vis_morph.cpu().detach().numpy()
latent_vis_np_rna = latent_vis_rna.cpu().detach().numpy()

# turn np array into a dataframe
df_morph = pd.DataFrame(latent_vis_np_morph)
df_rna = pd.DataFrame(latent_vis_np_rna)


# add labels and cell_ids to dataframe
df_morph.insert(0, "Cell_id", cell_id_flat_morph)
df_morph.insert(1, "Label", label_flat_morph)

df_rna.insert(0, "Cell_id", sample_id_flat_rna)
df_rna.insert(1, "Label", label_flat_rna)

# df['Label'] = label_flat
# df['Cell_id'] = cell_id_flat

# save latent_vis csv for further processing
df_morph.to_csv(str(args.save_dir + "/" + "latent_vis_morph_" + str(args.save_type) +".csv"), index=False)
df_rna.to_csv(str(args.save_dir+ "/" + "latent_vis_rna_" + str(args.save_type) +".csv"), index=False)
