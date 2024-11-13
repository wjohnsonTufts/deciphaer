import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from skimage import io

import os

# Dataset class that reads in a set of morpheus features
class Morph_Features_Dataset(Dataset):
    def __init__(self, filename, labels):
        self.filename = filename
        # name of label file
        self.labels = labels
        self.morph_features_data, self.cell_id, self.labels, self.nfeatures, self.nclass = self._load_morph_features_data()

    def __len__(self):
        return len(self.morph_features_data)

    def __getitem__(self, idx):
        morph_features_sample = self.morph_features_data[idx]
        cell_id = self.cell_id[idx]
        labels = self.labels[idx]
        return {'tensor': torch.from_numpy(morph_features_sample).float(), 'cell_id': cell_id, "labels": labels}

    def _load_morph_features_data(self):
        data = pd.read_csv(self.filename, index_col = 0)
        data = data.transpose()
        # find number of features (before merging with cluster labels)
        # 2nd dim of numpy array (morph_features_data) = number of features per cell
        nfeatures = data.shape[1]
        labels = pd.read_csv(self.labels, index_col = 0)
        data = data.merge(labels, how='outer', left_index=True, right_index=True)
        cluster_label = data.iloc[:, nfeatures]
        nclass = len(cluster_label.unique())
        features = data.iloc[:, 0:nfeatures]
        return features.values, data.index.values, cluster_label.values, nfeatures, nclass # REPLACE WITH LABEL
    
class RNAseq_Dataset(Dataset):
    def __init__(self,filename, labels):
        self.filename = filename
        # name of label file
        self.labels = labels
        self.rnaseq_data, self.cell_id, self.labels, self.nfeatures, self.nclass = self._load_rnaseq_data()

    def __len__(self):
        return len(self.rnaseq_data)

    def __getitem__(self, idx):
        rnaseq_sample = self.rnaseq_data[idx]
        cell_id = self.cell_id[idx]
        labels = self.labels[idx]
        return {'tensor': torch.from_numpy(rnaseq_sample).float(), 'cell_id': cell_id, "labels": labels}

    def _load_rnaseq_data(self):
        data = pd.read_csv(self.filename, index_col = 0)
        data = data.transpose()
        nfeatures = data.shape[1]
        labels = pd.read_csv(self.labels, index_col = 0)
        data = data.merge(labels, how='outer', left_index=True, right_index=True)
        cluster_label = data.iloc[:, nfeatures]
        nclass = len(cluster_label.unique())
        features = data.iloc[:, 0:nfeatures]
        return features.values, data.index.values, cluster_label.values, nfeatures, nclass

class Metabolomics_Dataset(Dataset):
    def __init__(self, filename, labels):
        self.filename = filename
        # name of label file
        self.labels = labels
        self.meta_data, self.cell_id, self.labels, self.nfeatures, self.nclass = self._load_meta_data()

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        met_sample = self.meta_data[idx]
        cell_id = self.cell_id[idx]
        labels = self.labels[idx]
        return {'tensor': torch.from_numpy(met_sample).float(), 'cell_id': cell_id, "labels": labels}

    def _load_meta_data(self):
        data = pd.read_csv(self.filename, index_col = 0)
        data = data.transpose()
        # find number of features (before merging with cluster labels)
        # 2nd dim of numpy array (morph_features_data) = number of features per cell
        nfeatures = data.shape[1]
        labels = pd.read_csv(self.labels, index_col = 0)
        data = data.merge(labels, how='outer', left_index=True, right_index=True)
        cluster_label = data.iloc[:, nfeatures]
        nclass = len(cluster_label.unique())
        features = data.iloc[:, 0:nfeatures]
        return features.values, data.index.values, cluster_label.values, nfeatures, nclass# REPLACE WITH LABEL
    
def test_loaders():
    print("testing morpheus loader")
    dataset = Morph_Features_Dataset(
                                    filename = "/Users/aalivi01/git_projects/deciphaer/data/match_hd_danger_T4_T24/morph_data.csv", 
                                    labels="/Users/aalivi01/git_projects/deciphaer/data/match_hd_danger_T4_T24/morph_meta_drug_labels_ae_input.csv")

    print("length of dataset:", len(dataset))
    print("number of features:", dataset.nfeatures)
    print("number of classes:", dataset.nclass)
    for i in dataset:
        if (str((i['tensor']).dtype) != "torch.float32"):
            print(type(i['labels']))
            print("cell_id:", i['cell_id'])
            print((i['tensor']).dtype)
            print("tensor:", i['tensor'])
            print("cluster label:", i['labels'])
            print("testing rnaseq loader")
    rna = RNAseq_Dataset(
                                filename = "/Users/aalivi01/git_projects/deciphaer/data/match_hd_danger_T4_T24/rna_data.csv", 
                                labels="/Users/aalivi01/git_projects/deciphaer/data/match_hd_danger_T4_T24/rna_meta_drug_labels_ae_input.csv")

    print("length of dataset:", len(rna))
    print("number of features:", rna.nfeatures)
    print("number of classes:", rna.nclass)
    # metabolomics = Metabolomics_Dataset(datadir="data_aldridge", 
    #                             filename = "metabolomics_chol_logTrans_050922.csv", 
    #                             labels="metabolomics_chol_labels_050922.csv")

    # print("length of dataset:", len(metabolomics))
    # print("number of features:", metabolomics.nfeatures)
    # sample = metabolomics[1]
    # print("sample_id:", sample['sample_id'])
    # print("tensor:", sample['tensor'])
    # print("cluster label:", sample['labels'])
    """for i in rna:
        print("cell_id:", i['cell_id'])
        print((i['tensor']).dtype)
        print("tensor:", i['tensor'])
        print("cluster label:", i['labels'])
        print("testing rnaseq loader")
    """
# if this file is being run as the primary file, tests the dataset
if __name__ == "__main__":
    test_loaders()

