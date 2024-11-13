#!/usr/bin/env python
import scipy.stats as st
import mne
import pandas as pd
import numpy as np
import pickle
import os

#Load train and test datasets to perform the DE testing
data = pd.read_csv("/Users/wjohns07/Desktop/GitLab/deciphaer/Models_Analysis/Finalized_Models/LCD/inf_final_1/Iterated_translations/morph_to_rna_test.csv")
meta = pd.read_csv("/Users/wjohns07/Desktop/GitLab/deciphaer/data/final_safe_dataset_pt_scaled_correction_071123/PT/PT_morph_meta.csv")
train_data = pd.read_csv("/Users/wjohns07/Desktop/GitLab/deciphaer/Models_Analysis/Finalized_Models/LCD/inf_final_1/Iterated_translations/morph_to_rna_train.csv")
train_meta = pd.read_csv("/Users/wjohns07/Desktop/GitLab/deciphaer/data/final_safe_dataset_pt_scaled_correction_071123/morph_meta.csv")
scalar = pickle.load(open('/Users/wjohns07/Desktop/GitLab/deciphaer/data/final_safe_dataset_pt_scaled_correction_071123/preprocessing_files/rna_scaler.pkl', 'rb'))
annotations = pd.read_csv("/Users/wjohns07/Desktop/GitLab/deciphaer/code/erdman_to_RV_tanlab.csv")
features = pd.read_csv("/Users/wjohns07/Desktop/GitLab/deciphaer/data/final_safe_dataset_pt_scaled_correction_071123/rna_data.csv")["locus_tag"]
out_dir = "/Users/wjohns07/Desktop/GitLab/deciphaer/Models_Analysis/Finalized_Models/LCD/inf_final_1/stat_test_results_all_021224/"
#cluster_mapping
clus_map= pd.read_csv("/Users/wjohns07/Desktop/GitLab/deciphaer/Models_Analysis/Finalized_Models/LCD/inf_final_1/stat_test_results_all_021224/")
#Ref Group
ref_group = "UNT"
annotations = annotations.rename(columns = {'Rv#':'gene'})
#Define functions
def stat_tests(values1, values2):
    mann = st.mannwhitneyu(x=values1, y=values2, method="exact")
    welch = st.ttest_ind(a=values1, b=values2, equal_var=False)
    return mann[1], welch[1]

#Make out directory
if os.path.exists(out_dir) == False:
    os.mkdir(out_dir)

#Bind the data frames and unscale 
data = pd.concat([data,train_data])
Cell_ID = data.Cell_id.tolist()
inverse_transformed = pd.DataFrame(scalar.inverse_transform(data.iloc[:,2:]))

#Rename the variables to genes and add the cell_id back in. Then export the data
inverse_transformed = inverse_transformed.set_axis(features, axis=1)
inverse_transformed['Cell_ID'] = Cell_ID
data = inverse_transformed

#Save the unscaled data for later
data.to_csv(str(out_dir +'inverse_transformed_translated_data.csv'), index = False)

#Generate full metadata
meta['cluster_marker_lcd'] = max(train_meta['cluster_marker_lcd']) +meta['cluster_marker_lcd']
full_meta = pd.concat([meta,train_meta])

##REFERENCE GROUP Testing (name vs Untreated in training)
#outer loop to return get the drug of interest
#inner loop to return the values of the statistical tests
druglabel = np.unique(full_meta['cluster_marker_lcd'])
for i in druglabel:
    target_IDs = full_meta[(full_meta.cluster_marker_lcd == i)]["Cell_ID"].tolist()
    ref_IDs = train_meta[(train_meta.Drug == ref_group)]["Cell_ID"].tolist()
    mannList = []
    welchList = []
    fcList = []
    gene = []
    for genename in features:
        mann,welch = stat_tests(np.array(data[data.Cell_ID.isin(target_IDs)][genename]), np.array(data[data.Cell_ID.isin(ref_IDs)][genename]))
        fc = ((np.mean(np.array(data[data.Cell_ID.isin(target_IDs)][genename]))) / (np.mean(np.array(data[data.Cell_ID.isin(ref_IDs)][genename]))) )-1
        mannList.append(mann)
        welchList.append(welch)
        fcList.append(fc)
        gene.append(genename)
    #padj = mne.stats.fdr_correction(pvalList, method = "indep")[1]
    retdf = pd.DataFrame(list(zip(gene, fcList,mannList, welchList)),columns =['gene','FC', 'mann_pval', 'welch_pval'])
    retdf = retdf.merge(annotations, on='gene', how='left')
    retdf.to_csv(str(out_dir +str(i) +'_vs_Unt.csv'), index = False)
##CLUSTER TESTING against training
#To perform differential expression testing on clusters of compounds, do the same thing but search for clusters and only use all of the morph train data for the reference group
label = np.unique(full_meta['cluster_marker_lcd'])
for i in label:
    target_IDs = full_meta[(full_meta.cluster_marker_lcd == i)]["Cell_ID"].tolist()
    print("\n" + "Targets: \n")
    print(target_IDs)
    ref_IDs = (full_meta[(full_meta.cluster_marker_lcd != i)].merge(train_data, left_on='Cell_ID', right_on='Cell_id', how = "inner"))["Cell_ID"].tolist()
    print("\n" +"Reference: \n")
    print(ref_IDs)
    mannList = []
    welchList = []
    fcList = []
    gene = []
    for genename in features:
        mann,welch = stat_tests(np.array(data[data.Cell_ID.isin(target_IDs)][genename]), np.array(data[data.Cell_ID.isin(ref_IDs)][genename]))
        fc = ((np.mean(np.array(data[data.Cell_ID.isin(target_IDs)][genename]))) / (np.mean(np.array(data[data.Cell_ID.isin(ref_IDs)][genename]))) )-1
        mannList.append(mann)
        welchList.append(welch)
        fcList.append(fc)
        gene.append(genename)
    #padj = mne.stats.fdr_correction(pvalList, method = "indep")[1]
    retdf = pd.DataFrame(list(zip(gene, fcList,mannList, welchList)),columns =['gene','FC', 'mann_pval', 'welch_pval'])
    retdf = retdf.merge(annotations, on='gene', how='left')
    retdf.to_csv(str(out_dir +str(i) +'_vs_alltrain.csv'), index = False)
    
##Selected cluster testing against training
#To perform differential expression testing on clusters of compounds, do the same thing but search for clusters and only use all of the morph train data for the reference group
label = np.unique(full_meta['cluster_of'])
for i in label:
    target_IDs = full_meta[(full_meta.cluster_marker_lcd == i)]["Cell_ID"].tolist()
    print("\n" + "Targets: \n")
    print(target_IDs)
    ref_IDs = (full_meta[(full_meta.cluster_marker_lcd != i)].merge(train_data, left_on='Cell_ID', right_on='Cell_id', how = "inner"))["Cell_ID"].tolist()
    print("\n" +"Reference: \n")
    print(ref_IDs)
    mannList = []
    welchList = []
    fcList = []
    gene = []
    for genename in features:
        mann,welch = stat_tests(np.array(data[data.Cell_ID.isin(target_IDs)][genename]), np.array(data[data.Cell_ID.isin(ref_IDs)][genename]))
        fc = ((np.mean(np.array(data[data.Cell_ID.isin(target_IDs)][genename]))) / (np.mean(np.array(data[data.Cell_ID.isin(ref_IDs)][genename]))) )-1
        mannList.append(mann)
        welchList.append(welch)
        fcList.append(fc)
        gene.append(genename)
    #padj = mne.stats.fdr_correction(pvalList, method = "indep")[1]
    retdf = pd.DataFrame(list(zip(gene, fcList,mannList, welchList)),columns =['gene','FC', 'mann_pval', 'welch_pval'])
    retdf = retdf.merge(annotations, on='gene', how='left')
    retdf.to_csv(str(out_dir +str(i) +'_vs_alltrain.csv'), index = False)