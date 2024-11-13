#!/usr/bin/env python
import subprocess
import argparse, sys
import os
import pandas as pd
import glob
import csv
import re
import datetime
from pathlib import Path
from gooey import Gooey

# to do's 12/15:
# add comments delineating which modules are "fringe" vs. "base"

def parse_args():
    parser = argparse.ArgumentParser()
    basic = parser.add_argument_group("Standard arguments")
    data = parser.add_argument_group("Data selection")
    analysis_selection = parser.add_argument_group("Analysis selection")
    selected_models = parser.add_argument_group("Select parameters for model inference")
    exclusive = analysis_selection.add_mutually_exclusive_group()
    basic.add_argument('--config', required = True, help = 'Path to configuration .csv table with hyperparams', default = None)
    data.add_argument('--data_dir_drug', help = 'Path to the drug data directory. If no value is provided, do not use drug-labeled data.', default = None)
    data.add_argument('--data_dir_lcd', help = 'Path to the lcd data directory. If no value is provided, do not use lcd-labeled data.', default = None)
    data.add_argument('--pt_dir', help = 'Path to pretraining data directory. If no value is provided, no pre-training will be done for model selection.', default = None)
    basic.add_argument('--save_dir', help = "Path to Models_Analysis directory where deciphaer outputs should be saved.", default = None)
    basic.add_argument('--g_dir', help = "Path to directory containing the desired genesets for correlation plots.", default = None)
    exclusive.add_argument('--model_selection', action = "store_true", help = 'Train a new model using train/test splits.', default = False)
    exclusive.add_argument('--model_inference', action = "store_true", help = 'Select a model for model inference. Train on all available data, and then pass holdout drugs through the network.', default = False)
    selected_models.add_argument('--model_morph_drug', help = "If performing model inference on a drug model, input the path to your selected morph drug model.", default = None)
    selected_models.add_argument('--model_morph_lcd', help = "If performing model inference on an lcd model, input the path to your selected morph lcd model.", default = None)
    selected_models.add_argument('--model_rna_drug', help = "If performing model inference on a drug model, input the path to your selected rna drug model.", default = None)
    selected_models.add_argument('--model_rna_lcd', help = "If performing model inference on an lcd model, input the path to your selected rna lcd model.", default = None)
    selected_models.add_argument('--holdout_dir', help = "Directory of holdout data for model inference.", default = None)
    args = parser.parse_args()
    return args

def setup_log(args):
    
    logging.info("Input args: ")

class DeciphaerDataset:
    def __init__(self, hyper, data_dir, save_dir, label_type, pt_dir = None, train_test = None, holdout_dir = None):
        # load hyperparameters, store as list of dictionaries
        # each run is a dictionary, stored as a list to make these dictionaries iterable
        self.label_type = label_type
        self.data_dir = data_dir
        self.pt_dir = pt_dir
        self.holdout_dir = holdout_dir
        self.save_dir = save_dir
        self.data_dir_meta = str(Path(data_dir).parents[0]) # directory for metadata is just the parent directory of the data_dir

        with open(hyper, "r") as f:
            reader = csv.DictReader(f)
            self.hyper = list(reader)

        if train_test == True:
            self.morph_train = sorted(glob.glob(str(data_dir + "/morph_data_train*")))
            self.morph_train_labels = sorted(glob.glob(str(data_dir + "/morph_meta_" + label_type + "_labels_ae_input_train*")))
            self.morph_test = sorted(glob.glob(str(data_dir + "/morph_data_test*")))
            self.morph_test_labels = sorted(glob.glob(str(data_dir + "/morph_meta_" + label_type + "_labels_ae_input_test*")))
            self.rna_train = sorted(glob.glob(str(data_dir + "/rna_data_train*")))
            self.rna_train_labels = sorted(glob.glob(str(data_dir + "/rna_meta_" + label_type + "_labels_ae_input_train*")))
            self.rna_test = sorted(glob.glob(str(data_dir + "/rna_data_test*")))
            self.rna_test_labels = sorted(glob.glob(str(data_dir + "/rna_meta_" + label_type + "_labels_ae_input_test*")))

        elif holdout_dir != None: # designed for model inference
            self.morph_train = sorted(glob.glob(str(self.data_dir_meta + "/morph_data*")))
            self.morph_train_labels = sorted(glob.glob(str(self.data_dir_meta + "/morph_meta_" + label_type + "_labels_ae_input*")))
            self.rna_train = sorted(glob.glob(str(self.data_dir_meta + "/rna_data*")))
            self.rna_train_labels = sorted(glob.glob(str(self.data_dir_meta + "/rna_meta_" + label_type + "_labels_ae_input*")))
            self.morph_test = sorted(glob.glob(str(self.holdout_dir + "/*morph_data*")))
            self.morph_test_labels = sorted(glob.glob(str(self.holdout_dir + "/*morph_meta_" + label_type + "_labels_ae_input*")))
            self.rna_test = sorted(glob.glob(str(self.holdout_dir + "/*rna_data*")))
            self.rna_test_labels = sorted(glob.glob(str(self.holdout_dir + "/*rna_meta_" + label_type + "_labels_ae_input*")))


        if pt_dir != None:
            self.pt_morph = str(pt_dir + "/PT_morph_data.csv")
            self.pt_morph_labels = str(pt_dir + "/PT_morph_meta_" + label_type + "_labels_ae_input.csv")
            self.pt_rna = str(pt_dir + "/PT_rna_data.csv")
            self.pt_rna_labels = str(pt_dir + "/PT_rna_meta_" + label_type + "_labels_ae_input.csv")
    
    def get_latest_model(self, model_dir, type):
        if model_dir.endswith("/") == False:
            model_dir = model_dir + "/"
        # given a run or fold directory, returns a string leading to the latest morph or rna model.
        if type == "morph":
            model = sorted(glob.glob(str(model_dir + "netMorph_*"))).pop()
        if type == "pt_morph":
            print(model_dir)
            model = sorted(glob.glob(str(model_dir + "Morph/netMorph_*"))).pop()
        if type == "rna":
            model = sorted(glob.glob(str(model_dir + "netRNA_*"))).pop()
        if type == "pt_rna":
            model = sorted(glob.glob(str(model_dir + "RNA/netRNA_*"))).pop()
        return model

    def print_report(self):
        print("----------- DECIPHAER DATASET -----------")
        print("label type:")
        print(self.label_type)
        print("data_dir:")
        print(self.data_dir)
        print("pt_dir:")
        print(self.pt_dir)
        print("holdout dir:")
        print(self.holdout_dir)
        print("morph_train:")
        print(self.morph_train)
        print("morph_train_labels:")
        print(self.morph_train_labels)
        print("morph_test")
        print(self.morph_test)
        print("morph_test_labels")
        print(self.morph_test_labels)
        print("rna_train:")
        print(self.rna_train)
        print("rna_test:")
        print(self.rna_test)
        print("rna_test_labels:")
        print(self.rna_test_labels)



def ae_pre_train(dataset, save_dir_run, run):
    print("pre-training RNAseq...")
    print(dataset.pt_rna)
    pretrain_rna="train_single_modal.py"+ \
        str(" -sd " + save_dir_run + "RNA/") + \
        str(" -ds " + dataset.data_dir) + \
        str(" -fn " + dataset.pt_rna)+ \
        str(" -lb " + dataset.pt_rna_labels)+ \
        str(" -e " + run['pt_epoch_rna']) + \
        str(" -alpha " + run['pt_alpha_rna']) + \
        str(" -beta " + run['pt_beta_rna']) + \
        str(" -lamb " + run['pt_lamb_rna']) + \
        str(" -nz " + run['z_size']) + \
        str(" -bs " + run['pt_batch_size_rna']) + \
        str(" -lrAE " + run['pt_lrAE_rna'])
    print("Starting AE pre-train for RNA-seq: " + pretrain_rna)
    subprocess.call(str(os.getcwd()+'/'+ pretrain_rna), shell=True)

    print("pre-training morph...")
    pretrain_morph="train_single_modal.py"+ \
        str(" -sd " + save_dir_run + "Morph/") + \
        str(" -ds " + dataset.data_dir) + \
        str(" -fn " + dataset.pt_morph)+ \
        str(" -lb " + dataset.pt_morph_labels)+ \
        str(" -e " + run['pt_epoch_morph']) + \
        str(" -alpha " + run['pt_alpha_morph']) + \
        str(" -beta " + run['pt_beta_morph']) + \
        str(" -lamb " + run['pt_lamb_morph']) + \
        str(" -nz " + run['z_size']) + \
        str(" -bs " + run['pt_batch_size_morph']) + \
        str(" -lrAE " + run['pt_lrAE_morph'])
    print("Starting AE pre-train for Morph: " + pretrain_morph)
    subprocess.call(str(os.getcwd() + '/' + pretrain_morph), shell = True)
    print("pre-training complete.")

def ae_train_fold(dataset, save_dir_run, save_dir_fold, run, fold, pt_morph, pt_rna):
    fold_str = str(fold + 1)
    print("training fold", fold_str)
    if pt_morph != None and pt_rna != None:
        ae_train="train_bi_modal.py"+ \
            str(" -sd " + save_dir_fold) + \
            str(" -ds " + dataset.data_dir) + \
            str(" -ptr " + pt_rna) + \
            str(" -fnr " + dataset.rna_train[fold])+ \
            str(" -lbr " + dataset.rna_train_labels[fold])+ \
            str(" -ptm " + pt_morph)+ \
            str(" -fnm " + dataset.morph_train[fold])+ \
            str(" -lbm " + dataset.morph_train_labels[fold])+ \
            str(" -bs " + run['batch_size']) + \
            str(" -lrAE " + run['lrAE']) + \
            str(" -e " + run['epoch']) + \
            str(" -alpha_rna " + run['alpha_rna']) + \
            str(" -alpha_morph " + run['alpha_morph']) + \
            str(" -beta " + run['beta']) + \
            str(" -lamb " + run['lamb']) + \
            str(" -gamm ") + run['gamm'] + \
            str(" -nz " + run['z_size']) + \
            str(" --conditional" ) # + \
            ######### unused conditionals (if these are uncommented, these features are not turned on in the training script.)
            # str(" --conditional-adv " ) + \
            # str(" --use-gpu" ) + \
            ######### defaulted values (used in training script but not in config file. Can add a column to config to access here.)
            # str(" -lrD some_value")
            # str(" -wd some_value") 
            # uncomment the above to turn on these features. Must add whitespace after last option if doing this.
    else:
        ae_train="train_bi_modal.py"+ \
            str(" -sd " + save_dir_fold) + \
            str(" -ds " + dataset.data_dir) + \
            str(" -fnr " + dataset.rna_train[fold])+ \
            str(" -lbr " + dataset.rna_train_labels[fold])+ \
            str(" -fnm " + dataset.morph_train[fold])+ \
            str(" -lbm " + dataset.morph_train_labels[fold])+ \
            str(" -bs " + run['batch_size']) + \
            str(" -lrAE " + run['lrAE']) + \
            str(" -e " + run['epoch']) + \
            str(" -alpha_rna " + run['alpha_rna']) + \
            str(" -alpha_morph " + run['alpha_morph']) + \
            str(" -beta " + run['beta']) + \
            str(" -lamb " + run['lamb']) + \
            str(" -gamm ") + run['gamm'] + \
            str(" -nz " + run['z_size']) + \
            str(" --conditional" ) # + \
            ######### unused conditionals (if these are uncommented, these features are not turned on in the training script.)
            # str(" --conditional-adv " ) + \
            # str(" --use-gpu" ) + \
            ######### defaulted values (used in training script but not in config file. Can add a column to config to access here.)
            # str(" -lrD some_value")
            # str(" -wd some_value") 
            # uncomment the above to turn on these features. Must add whitespace after last option if doing this.

    subprocess.call(str(os.getcwd() + '/' + ae_train), shell = True)
    print("Autoencoder training complete.")

def latent_access_fold(dataset, save_dir_fold, run, fold):
    fold_str = str(fold + 1)
    print("accessing latent space for fold", fold_str)
    latent_train="latent.py"+ \
        str(" -st " "train") + \
        str(" -sd " + save_dir_fold) + \
        str(" -ptr " + dataset.get_latest_model(model_dir = save_dir_fold, type = "rna")) + \
        str(" -fnr " + dataset.rna_train[fold])+ \
        str(" -lbr " + dataset.rna_train_labels[fold])+ \
        str(" -ptm " + dataset.get_latest_model(model_dir = save_dir_fold, type = "morph")) + \
        str(" -fnm " + dataset.morph_train[fold])+ \
        str(" -lbm " + dataset.morph_train_labels[fold])+ \
        str(" -e " + run['epoch']) + \
        str(" -nz " + run['z_size'])

    latent_test="latent.py"+ \
        str(" -st " "test") + \
        str(" -sd " + save_dir_fold) + \
        str(" -ptr " + dataset.get_latest_model(model_dir = save_dir_fold, type = "rna")) + \
        str(" -fnr " + dataset.rna_test[fold])+ \
        str(" -lbr " + dataset.rna_test_labels[fold])+ \
        str(" -ptm " + dataset.get_latest_model(model_dir = save_dir_fold, type = "morph")) + \
        str(" -fnm " + dataset.morph_test[fold])+ \
        str(" -lbm " + dataset.morph_test_labels[fold])+ \
        str(" -e " + run['epoch']) + \
        str(" -nz " + run['z_size'])

    subprocess.call(str(os.getcwd() + '/' + latent_train), shell = True)
    subprocess.call(str(os.getcwd() + '/' + latent_test), shell = True)
    print("Latent access complete.")


def translation_fold(dataset, save_dir_fold, run, fold):
    # inputs: models from AE training, training data
    # outputs: translations of all modalities into each other
    fold_str = str(fold + 1)
    print("Translating data for fold", fold_str)
    translation_train="translator.py" + \
        str(" -st " "train") + \
        str(" -sd " + save_dir_fold) + \
        str(" -ptr " + dataset.get_latest_model(model_dir = save_dir_fold, type = "rna")) + \
        str(" -fnr " + dataset.rna_train[fold])+ \
        str(" -lbr " + dataset.rna_train_labels[fold])+ \
        str(" -ptm " + dataset.get_latest_model(model_dir = save_dir_fold, type = "morph")) + \
        str(" -fnm " + dataset.morph_train[fold])+ \
        str(" -lbm " + dataset.morph_train_labels[fold])+ \
        str(" -e " + run['epoch']) + \
        str(" -nz " + run['z_size'])
    
    translation_test="translator.py" + \
        str(" -st " "test") + \
        str(" -sd " + save_dir_fold) + \
        str(" -ptr " + dataset.get_latest_model(model_dir = save_dir_fold, type = "rna")) + \
        str(" -fnr " + dataset.rna_test[fold])+ \
        str(" -lbr " + dataset.rna_test_labels[fold])+ \
        str(" -ptm " + dataset.get_latest_model(model_dir = save_dir_fold, type = "morph")) + \
        str(" -fnm " + dataset.morph_test[fold])+ \
        str(" -lbm " + dataset.morph_test_labels[fold])+ \
        str(" -e " + run['epoch']) + \
        str(" -nz " + run['z_size'])

    subprocess.call(str(os.getcwd() + '/' + translation_train), shell = True)
    subprocess.call(str(os.getcwd() + '/' + translation_test), shell = True)
    print("Translations complete.")

def translation_holdout(dataset, save_dir_fold, run, fold):
    print("Translating holdout data...")
    translation_holdout = "translator.py" + \
        str(" -st " "holdout") + \
        str(" -sd " + save_dir) + \
        str(" -ptr " + dataset.get_latest_model(model_dir = save_dir, type = "rna")) + \
        str(" -fnr " + dataset.rna_test[fold])+ \
        str(" -lbr " + dataset.rna_test_labels[fold])+ \
        str(" -ptm " + dataset.get_latest_model(model_dir = save_dir_fold, type = "morph")) + \
        str(" -fnm " + dataset.morph_test[fold])+ \
        str(" -lbm " + dataset.morph_test_labels[fold])+ \
        str(" -e " + run['epoch']) + \
        str(" -nz " + run['z_size'])


def model_selection_analysis(dataset, save_dir_run, run, g_dir):
    # inputs: 
    # outputs:
    print("Analyzing run...")
    KNN_R_call = "/usr/local/bin/Rscript KNN_Fstat.R" + \
        str(" -d " + save_dir_run) + \
        str(" -z " + run['z_size']) + \
        str(" 2>&1 >/dev/null")

    UMAPS_corr_plots_R_call = "/usr/local/bin/Rscript translation_UMAPs_corr_plots.R" + \
        str(" -d " + save_dir_run) + \
        str(" -x " + dataset.data_dir) + \
        str(" -m " + dataset.data_dir_meta) + \
        str(" -g " + g_dir) + \
        str(" 2>&1 >/dev/null")

    subprocess.call(KNN_R_call, shell=True)
    subprocess.call(UMAPS_corr_plots_R_call, shell = True)

    print("Model analysis complete.")

def model_inference_analysis(dataset, save_dir_run, run):
    print("Generating model inference UMAPs...")
    model_inference_R_call = "/usr/local/bin/Rscript model_inference.R" + \
        str(" -d " + save_dir_run) + \
        str(" -z " + run['z_size']) + \
        str(" 2>&1 >/dev/null")
    subprocess.call(model_inference_R_call, shell = True)
    print("Model inference complete.")


def model_selection(args, dataset):
    for run in dataset.hyper:
        print("Starting AE run", run['Run'], "with", dataset.label_type, "labels.")
        save_dir_run = str(dataset.save_dir + "/" + run['Run'] + "/")
        morph_base_model = None
        rna_base_model = None
        if args.pt_dir != None:
            ae_pre_train(dataset = dataset, save_dir_run = save_dir_run, run = run)
            morph_base_model = dataset.get_latest_model(model_dir = save_dir_run, type = "pt_morph")
            rna_base_model = dataset.get_latest_model(model_dir = save_dir_run, type = "pt_rna")
        for fold in range(len(dataset.morph_train)):
            fold_str = str(fold + 1) # for file naming purposes
            save_dir_fold = save_dir_run + fold_str + "/"
            ae_train_fold(dataset = dataset, save_dir_run = save_dir_run, save_dir_fold = save_dir_fold, fold = fold, run = run, pt_morph = morph_base_model, pt_rna = rna_base_model)
            latent_access_fold(dataset = dataset, save_dir_fold = save_dir_fold, fold = fold, run = run)
            translation_fold(dataset = dataset, save_dir_fold = save_dir_fold, fold = fold, run = run)
        model_selection_analysis(dataset = dataset, save_dir_run = save_dir_run, run = run, g_dir = args.g_dir)

def model_inference(args, dataset):
    # train on the entirety of the available data
    for run in dataset.hyper:
        print("Starting AE run", run['Run'], "with", dataset.label_type, "labels.")
        save_dir_run = str(dataset.save_dir + "/" + run['Run'] + "/")
        morph_base_model = None
        rna_base_model = None
        if args.pt_dir != None:
            ae_pre_train(dataset = dataset, save_dir_run = save_dir_run, run = run)
            morph_base_model = dataset.get_latest_model(model_dir = save_dir_run, type = "pt_morph")
            rna_base_model = dataset.get_latest_model(model_dir = save_dir_run, type = "pt_rna")
        for fold in range(len(dataset.morph_train)):
            # train using base models from model selection
            ae_train_fold(dataset = dataset, save_dir_run = save_dir_run, save_dir_fold = save_dir_run, fold = fold, run = run, pt_morph = morph_base_model, pt_rna = rna_base_model)
            # test using holdout data for model inference
            latent_access_fold(dataset = dataset, save_dir_fold = save_dir_run, fold = fold, run = run)
            translation_fold(dataset = dataset, save_dir_fold = save_dir_run, fold = fold, run = run)
        model_inference_analysis(dataset = dataset, save_dir_run = save_dir_run, run = run)

# needs to be BEFORE Gooey decorator.
# only uses Gooey when no arguments are passed to the script.
if len(sys.argv) >= 2:
    if not '--ignore-gooey' in sys.argv:
        sys.argv.append('--ignore-gooey')

@Gooey
def main():
    print("Starting Deciphaer...")
    args = parse_args()
    if args.model_selection == True:
        if args.data_dir_lcd != None:
            dataset_lcd = DeciphaerDataset(hyper=args.config, data_dir=args.data_dir_lcd, save_dir = args.save_dir + "/LCD", label_type = "lcd", pt_dir = args.pt_dir, train_test = True)
            model_selection(args = args, dataset = dataset_lcd)
        if args.data_dir_drug != None:
            dataset_drug = DeciphaerDataset(hyper=args.config, data_dir=args.data_dir_drug, save_dir = args.save_dir + "/Drug", label_type = "drug", pt_dir = args.pt_dir, train_test = True)
            model_selection(args = args, dataset = dataset_drug)

    if args.model_inference == True:
        if args.data_dir_lcd != None:
            # if args.model_morph_lcd == None or args.model_rna_lcd == None:
            #     sys.exit("No LCD models provided for model inference. Exiting deciphaer.")
            dataset_lcd = DeciphaerDataset(hyper=args.config, data_dir=args.data_dir_lcd, save_dir = args.save_dir + "/Finalized_Models/LCD", label_type = "lcd", holdout_dir = args.holdout_dir, pt_dir = args.pt_dir)
            dataset_lcd.print_report()
            model_inference(args = args, dataset = dataset_lcd)
        if args.data_dir_drug != None:
            # if args.model_morph_drug == None or args.model_rna_drug == None:
            #     sys.exit("No drug models provided for model inference. Exiting deciphaer.")
            dataset_drug = DeciphaerDataset(hyper=args.config, data_dir=args.data_dir_drug, save_dir = args.save_dir + "/Finalized_Models/Drug", label_type = "drug", holdout_dir = args.holdout_dir, pt_dir = args.pt_dir)
            dataset_drug.print_report()
            model_inference(args = args, dataset = dataset_drug)
    

if __name__ == '__main__':
    # added logging functionality. Logs are added to deciphaer_logfile.txt with the arguments provided each time deciphaer.py is run.
    now = datetime.datetime.now().ctime()
    with open('deciphaer_logfile.txt', 'a') as log:
        log.write("Deciphaer started %s with args: %s\n" % (now, sys.argv))
    main()
