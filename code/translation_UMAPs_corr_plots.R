# options(warn=-1)
# sink("r_out.txt")

suppressMessages({
  library(optparse, verbose = F)
  library(pheatmap, verbose=F)
  library(umap, verbose = F)
  library(RColorBrewer,verbose = F)
  library(readr,verbose = F)
  library(class,verbose = F)
  library(tidyverse,verbose = F, quietly = T)
  library(ggplot2,verbose = F)
})


option_list = list(
  make_option(c("-d", "--dir_ls"), type="character", default="/Users/wjohns07/Google\ Drive/Other\ computers/PhD\ Years\ 3.5\ -\ 4/Desktop/GitLab/deciphaer/Models_Analysis/LCD/dirk_final_1/", 
              help="AE run directory", metavar="character"),
  make_option(c("-x", "--x_data"), type="character", default="/Users/wjohns07/Desktop/GitLab/deciphaer/data/final_safe_dataset_pt_scaled_correction_071123/LCD/", 
              help="dataset directory", metavar="character"),
  make_option(c("-m", "--meta_data"), type="character", default="/Users/wjohns07/Desktop/GitLab/deciphaer/data/final_safe_dataset_pt_scaled_correction_071123/", 
              help="Directory for metadata", metavar="character"),
  make_option(c("-g", "--geneset_dir"), type = "character", default = "/Users/wjohns07/Desktop/GitLab/deciphaer/Genesets/Broad_DESeq2_results_Drug_v_UNT_Chol_112822")
);

# directory setup
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);
path_to_latent <- opt$d
data_dir <- paste0(opt$x, "/")
meta_dir <- paste0(opt$m, "/")
geneset_dir <- paste0(opt$g, "/")
save_folder_UMAP <- paste0(path_to_latent, "UMAPs/")
save_folder_corr_plots <- paste0(path_to_latent, "corr_plots/")
dir.create(save_folder_UMAP, showWarnings = FALSE)
dir.create(save_folder_corr_plots, showWarnings = FALSE)


morph_to_rna <- list.files(path = path_to_latent, pattern = "morph_to_rna_train.csv", full.names = T, recursive = T)
morph_to_rna <- lapply(morph_to_rna, function(x){
  read_csv(x, show_col_types=F)
})

morph_to_rna_test <- list.files(path = path_to_latent, pattern = "morph_to_rna_test.csv", full.names = T, recursive = T)
morph_to_rna_test <- lapply(morph_to_rna_test, function(x){
  read_csv(x, show_col_types=F)
})

morph_to_morph <- list.files(path = path_to_latent, pattern = "morph_to_morph_train.csv", full.names = T, recursive = T)
morph_to_morph <- lapply(morph_to_morph, function(x) {
  read_csv(x, show_col_types = F)
})

morph_to_morph_test <- list.files(path = path_to_latent, pattern = "morph_to_morph_test.csv", full.names = T, recursive = T)
morph_to_morph_test <- lapply(morph_to_morph_test, function(x) {
  read_csv(x, show_col_types = F)
})

rna_to_morph <- list.files(path = path_to_latent, pattern = "rna_to_morph_train.csv", full.names = T, recursive = T)
rna_to_morph <- lapply(rna_to_morph, function(x) {
  read_csv(x, show_col_types = F)
})

rna_to_morph_test <- list.files(path = path_to_latent, pattern = "rna_to_morph_test.csv", full.names = T, recursive = T)
rna_to_morph_test <- lapply(rna_to_morph_test, function(x) {
  read_csv(x, show_col_types = F)
})

rna_to_rna <- list.files(path = path_to_latent, pattern = "rna_to_rna_train.csv", full.names = T, recursive = T)
rna_to_rna <- lapply(rna_to_rna, function(x){
  read_csv(x, show_col_types=F)
})
rna_to_rna_test <- list.files(path = path_to_latent, pattern = "rna_to_rna_test.csv", full.names = T, recursive = T)
rna_to_rna_test <- lapply(rna_to_rna_test, function(x){
  read_csv(x, show_col_types=F)
})

original <- list.files(path = data_dir, pattern = "rna_data_train_", full.names = T, recursive = T)
original <- lapply(original, function(x){
  read_csv(x, show_col_types=F)
})

original_morph <- list.files(path = data_dir, pattern = "morph_data_train_", full.names = T, recursive = T)
original_morph <- lapply(original_morph, function(x){
  read_csv(x, show_col_types = F)
})

RNA_meta <- read_csv(paste(meta_dir, "rna_meta.csv", sep=""), show_col_types = F)
morph_meta <- read_csv(paste(meta_dir, "morph_meta.csv", sep=""), show_col_types = F)
RNA_meta_subset <- RNA_meta %>% dplyr::select(Run, Drug)
morph_meta_subset <- morph_meta %>% dplyr::select(Cell_ID, Drug)

morph_to_rna <- lapply(c(1:length(morph_to_rna)), function(x){
  morph_to_rna[[x]] <- morph_to_rna[[x]] %>% dplyr::rename(id=Cell_id)
})
morph_to_rna_test <- lapply(c(1:length(morph_to_rna_test)), function(x){
  morph_to_rna_test[[x]] <- morph_to_rna_test[[x]] %>% dplyr::rename(id=Cell_id)
})
rna_to_rna <- lapply(c(1:length(rna_to_rna)), function(x){
  rna_to_rna[[x]] <- rna_to_rna[[x]] %>% dplyr::rename(id=Cell_id)
})
rna_to_rna_test <- lapply(c(1:length(rna_to_rna_test)), function(x){
  rna_to_rna_test[[x]] <- rna_to_rna_test[[x]] %>% dplyr::rename(id=Cell_id)
})

morph_to_morph <- lapply(c(1:length(morph_to_morph)), function(x){
  morph_to_morph[[x]] <- morph_to_morph[[x]] %>% dplyr::rename(id=Cell_id)
})
morph_to_morph_test <- lapply(c(1:length(morph_to_morph_test)), function(x){
  morph_to_morph_test[[x]] <- morph_to_morph_test[[x]] %>% dplyr::rename(id=Cell_id)
})

rna_to_morph <- lapply(c(1:length(rna_to_morph)), function(x){
  rna_to_morph[[x]] <- rna_to_morph[[x]] %>% dplyr::rename(id=Cell_id)
})
rna_to_morph_test <- lapply(c(1:length(rna_to_morph_test)), function(x){
  rna_to_morph_test[[x]] <- rna_to_morph_test[[x]] %>% dplyr::rename(id=Cell_id)
})


tags <- list()
# tags <- original$locus_tag
tags <- lapply(c(1:length(original)), function(x){
  tags[[x]] <- original[[x]]$locus_tag
})

# original <- original %>% dplyr::select(-locus_tag)
original <- lapply(original, function(x){
  df <- as.data.frame(t(x[,-1]))
  colnames(df) <- unlist(x[,1])
  df$id <- rownames(df)
  dplyr::left_join(df, RNA_meta_subset, by=c("id"="Run")) %>% relocate(Drug) %>% relocate(id)
})

original_morph <- lapply(original_morph, function(x){
  df <- as.data.frame(t(x[,-1]))
  colnames(df) <- unlist(x[,1])
  df$id <- rownames(df)
  dplyr::left_join(df, morph_meta_subset, by=c("id"="Cell_ID")) %>% relocate(Drug) %>% relocate(id)
})

columns <- list()
columns <- lapply(c(1:length(original)), function(x){
  columns[[x]] <- colnames(original[[x]])
})

columns_morph <- list()
columns_morph <- lapply(c(1:length(original_morph)), function(x){
  columns_morph[[x]] <- colnames(original_morph[[x]])
})

# morph_to_rna <- left_join(morph_to_rna, RNA_meta_subset, by=c("Cell_id"="Run")) %>% relocate(Drug) %>% relocate(Cell_id) %>% rename(id=Cell_id) %>% select(-Label)
morph_to_rna <- lapply(c(1:length(morph_to_rna)), function(x){
  morph_to_rna[[x]] <- left_join(morph_to_rna[[x]], morph_meta_subset, by=c("id"="Cell_ID")) %>% relocate(Drug) %>% relocate(id) %>% select(-Label)
})
morph_to_rna_test <- lapply(c(1:length(morph_to_rna_test)), function(x){
  morph_to_rna_test[[x]] <- left_join(morph_to_rna_test[[x]], morph_meta_subset, by=c("id"="Cell_ID")) %>% relocate(Drug) %>% relocate(id) %>% select(-Label)
})

# same for rna to rna
rna_to_rna <- lapply(c(1:length(rna_to_rna)), function(x){
  rna_to_rna[[x]] <- left_join(rna_to_rna[[x]], RNA_meta_subset, by=c("id"="Run")) %>% relocate(Drug) %>% relocate(id) %>% select(-Label)
})
rna_to_rna_test <- lapply(c(1:length(rna_to_rna_test)), function(x){
  rna_to_rna_test[[x]] <- left_join(rna_to_rna_test[[x]], RNA_meta_subset, by=c("id"="Run")) %>% relocate(Drug) %>% relocate(id) %>% select(-Label)
})

# same for morph to morph and morph to rna
morph_to_morph <- lapply(c(1:length(morph_to_morph)), function(x){
  morph_to_morph[[x]] <- left_join(morph_to_morph[[x]], morph_meta_subset, by=c("id"="Cell_ID")) %>% relocate(Drug) %>% relocate(id) %>% select(-Label)
})
morph_to_morph_test <- lapply(c(1:length(morph_to_morph_test)), function(x){
  morph_to_morph_test[[x]] <- left_join(morph_to_morph_test[[x]], morph_meta_subset, by=c("id"="Cell_ID")) %>% relocate(Drug) %>% relocate(id) %>% select(-Label)
})

rna_to_morph <- lapply(c(1:length(rna_to_morph)), function(x){
  rna_to_morph[[x]] <- left_join(rna_to_morph[[x]], RNA_meta_subset, by=c("id"="Run")) %>% relocate(Drug) %>% relocate(id) %>% select(-Label)
})
rna_to_morph_test <- lapply(c(1:length(rna_to_morph_test)), function(x){
  rna_to_morph_test[[x]] <- left_join(rna_to_morph_test[[x]], RNA_meta_subset, by=c("id"="Run")) %>% relocate(Drug) %>% relocate(id) %>% select(-Label)
})


# colnames(morph_to_rna) <- columns
morph_to_rna <- lapply(c(1:length(morph_to_rna)), function(x){
  colnames(morph_to_rna[[x]]) <- columns[[x]]
  return(morph_to_rna[[x]])
})

morph_to_rna_test <- lapply(c(1:length(morph_to_rna_test)), function(x){
  colnames(morph_to_rna_test[[x]]) <- columns[[x]]
  return(morph_to_rna_test[[x]])
})

# colnames(rna_to_rna) <- columns
rna_to_rna <- lapply(c(1:length(rna_to_rna)), function(x){
  colnames(rna_to_rna[[x]]) <- columns[[x]]
  return(rna_to_rna[[x]])
})
rna_to_rna_test <- lapply(c(1:length(rna_to_rna_test)), function(x){
  colnames(rna_to_rna_test[[x]]) <- columns[[x]]
  return(rna_to_rna_test[[x]])
})

rna_to_morph <- lapply(c(1:length(rna_to_morph)), function(x){
  colnames(rna_to_morph[[x]]) <- columns_morph[[x]]
  return(rna_to_morph[[x]])
})
rna_to_morph_test <- lapply(c(1:length(rna_to_morph_test)), function(x){
  colnames(rna_to_morph_test[[x]]) <- columns_morph[[x]]
  return(rna_to_morph_test[[x]])
})

morph_to_morph <- lapply(c(1:length(morph_to_morph)), function(x){
  colnames(morph_to_morph[[x]]) <- columns_morph[[x]]
  return(morph_to_morph[[x]])
})
morph_to_morph_test <- lapply(c(1:length(morph_to_morph_test)), function(x){
  colnames(morph_to_morph_test[[x]]) <- columns_morph[[x]]
  return(morph_to_morph_test[[x]])
})

getUmapTable <- function(Df) {
  Df <- as.data.frame(Df)
  DfPlot <- umap::umap(Df)
  DfPlot <- as.data.frame(DfPlot$layout)
  DfPlot <- cbind(DfPlot,Df)
  DfPlot
}
getUmapTableClose <- function(Df) {
  Df <- as.data.frame(Df)
  DfPlot <- umap::umap(Df, n_neighbors =50)
  DfPlot <- as.data.frame(DfPlot$layout)
  DfPlot <- cbind(DfPlot,Df)
  DfPlot
}
getUmapTableDisperse <- function(Df) {
  Df <- as.data.frame(Df)
  DfPlot <- umap::umap(Df, n_neighbors =15, min_dist = 0.99)
  DfPlot <- as.data.frame(DfPlot$layout)
  DfPlot <- cbind(DfPlot,Df)
  DfPlot
}
getUmapTablelowNeigh <- function(Df) {
  Df <- as.data.frame(Df)
  DfPlot <- umap::umap(Df, n_neighbors =8)
  DfPlot <- as.data.frame(DfPlot$layout)
  DfPlot <- cbind(DfPlot,Df)
  DfPlot
}
getUmapTablehighNeigh <- function(Df) {
  Df <- as.data.frame(Df)
  DfPlot <- umap::umap(Df, n_neighbors =120)
  DfPlot <- as.data.frame(DfPlot$layout)
  DfPlot <- cbind(DfPlot,Df)
  DfPlot
}
getUmapTablelowDist <- function(Df) {
  Df <- as.data.frame(Df)
  DfPlot <- umap::umap(Df, n_neighbors =100, min_dist = 0.1)
  DfPlot <- as.data.frame(DfPlot$layout)
  DfPlot <- cbind(DfPlot,Df)
  DfPlot
}

RNA_meta_umap <- RNA_meta %>% select(Run, cluster_marker_lcd)
morph_meta_umap <- morph_meta %>% select(Cell_ID, cluster_marker_lcd)

umap_rna_to_rna <- list()
umap_morph_to_rna <- list()
umap_morph_to_morph <- list()
umap_rna_to_morph <- list()
rna <- list()
morph_to_rna_test1 <- list()
Drug_umap_list_rna <- list()
Drug_umap_list_morph <- list()
Drug_umap_list_morph_to_morph <- list()
Drug_umap_list_rna_to_morph <- list()
Drug_umap_list_rna_true <- list()
Broad_Cellular_Target_list_rna <- list()
Broad_Cellular_Target_list_morph <- list()
Broad_Cellular_Target_list_morph_to_morph <- list()
Broad_Cellular_Target_list_rna_to_morph <- list()
Broad_Cellular_Target_list_rna_true <- list()
Dataset_list_rna <- list()
Dataset_list_morph <- list()
Dataset_list_morph_to_morph <- list()
Dataset_list_rna_to_morph <- list()
Dataset_list_rna_true <- list()
Modality_list_rna_true <- list()

for (x in c(1:length(rna_to_rna))) {
  rna_to_rna[[x]]$Dataset <- "Train"
  rna_to_rna_test[[x]]$Dataset <- "Test"
  umap_rna_to_rna[[x]]<- bind_rows(rna_to_rna_test[[x]],rna_to_rna[[x]])
  umap_rna_to_rna[[x]] <- umap_rna_to_rna[[x]] %>% left_join(RNA_meta_umap, by=c("id"="Run")) %>% dplyr::select(id, Drug, cluster_marker_lcd,Dataset, everything())
  Drug_umap_list_rna[[x]] <- umap_rna_to_rna[[x]]$Drug
  Broad_Cellular_Target_list_rna[[x]] <- umap_rna_to_rna[[x]]$cluster_marker_lcd
  Dataset_list_rna[[x]] <- umap_rna_to_rna[[x]]$Dataset
  
  morph_to_rna[[x]]$Dataset <- "Train"
  morph_to_rna_test[[x]]$Dataset <- "Test"
  morph_to_rna[[x]]<- bind_rows(morph_to_rna_test[[x]],morph_to_rna[[x]])
  umap_morph_to_rna[[x]] <- morph_to_rna[[x]] %>% left_join(morph_meta_umap, by=c("id"="Cell_ID")) %>% dplyr::select(id, Drug, cluster_marker_lcd,Dataset, everything())
  Drug_umap_list_morph[[x]] <- umap_morph_to_rna[[x]]$Drug
  Broad_Cellular_Target_list_morph[[x]] <- umap_morph_to_rna[[x]]$cluster_marker_lcd
  Dataset_list_morph[[x]] <- umap_morph_to_rna[[x]]$Dataset
  
  rna_to_morph[[x]]$Dataset <- "Train"
  rna_to_morph_test[[x]]$Dataset <- "Test"
  rna_to_morph[[x]]<- bind_rows(rna_to_morph_test[[x]],rna_to_morph[[x]])
  umap_rna_to_morph[[x]] <- rna_to_morph[[x]] %>% left_join(RNA_meta_umap, by=c("id"="Run")) %>% dplyr::select(id, Drug, cluster_marker_lcd,Dataset, everything())
  Drug_umap_list_rna_to_morph[[x]] <- umap_rna_to_morph[[x]]$Drug
  Broad_Cellular_Target_list_rna_to_morph[[x]] <- umap_rna_to_morph[[x]]$cluster_marker_lcd
  Dataset_list_rna_to_morph[[x]] <- umap_rna_to_morph[[x]]$Dataset
  
  morph_to_morph[[x]]$Dataset <- "Train"
  morph_to_morph_test[[x]]$Dataset <- "Test"
  morph_to_morph[[x]]<- bind_rows(morph_to_morph_test[[x]],morph_to_morph[[x]])
  umap_morph_to_morph[[x]] <- morph_to_morph[[x]] %>% left_join(morph_meta_umap, by=c("id"="Cell_ID")) %>% dplyr::select(id, Drug, cluster_marker_lcd,Dataset, everything())
  Drug_umap_list_morph_to_morph[[x]] <- umap_morph_to_morph[[x]]$Drug
  Broad_Cellular_Target_list_morph_to_morph[[x]] <- umap_morph_to_morph[[x]]$cluster_marker_lcd
  Dataset_list_morph_to_morph[[x]] <- umap_morph_to_morph[[x]]$Dataset
  # 
  # rna[[x]] <- original[[x]]
  # rna[[x]]$Dataset <- "Train"
  # rna[[x]]$Modality <- "RNA-seq Profile"
  # morph_to_rna_test1[[x]] <- morph_to_rna_test[[x]]
  # morph_to_rna_test1[[x]]$Dataset <- "Test"
  # morph_to_rna_test1[[x]]$Modality <- "RNA-seq Profile Translated from Morphological Profile"
  # rna[[x]] <- bind_rows(morph_to_morph_test1[[x]],rna[[x]])
  # rna[[x]] <- rna[[x]] %>% left_join(morph_meta_umap, by=c("id"="Cell_ID")) %>% left_join(RNA_meta_subset, by=c("id"="Run")) %>% dplyr::select(id, Drug, cluster_marker_lcd,Dataset,Modality, everything())
  # Drug_umap_list_rna_true[[x]] <- rna[[x]]$Drug
  # Broad_Cellular_Target_list_rna_true[[x]] <- rna[[x]]$cluster_marker_lcd
  # Dataset_list_rna_true[[x]] <- rna[[x]]$Dataset
  # Modality_list_rna_true[[x]] <- rna[[x]]$Modality
}

full_reg_rna <- list()
full_reg_morph <- list()
full_reg_morph_to_morph <- list()
full_reg_rna_to_morph <- list()
full_reg_rna_true <- list()

for (x in c(1:length(umap_rna_to_rna))) {
  full_reg_rna[[x]] <- getUmapTableDisperse(umap_rna_to_rna[[x]][,c(-1,-2,-3, -4)])
  full_reg_rna[[x]]$Drug <- Drug_umap_list_rna[[x]]
  full_reg_rna[[x]]$Morph_knn_consen <- Broad_Cellular_Target_list_rna[[x]]
  full_reg_rna[[x]]$Dataset <- factor(Dataset_list_rna[[x]], levels = c("Train", "Test"))
  full_reg_morph[[x]] <- getUmapTableDisperse(umap_morph_to_rna[[x]][,c(-1,-2,-3, -4)])
  full_reg_morph[[x]]$Drug <- Drug_umap_list_morph[[x]]
  full_reg_morph[[x]]$Morph_knn_consen <- Broad_Cellular_Target_list_morph[[x]]
  full_reg_morph[[x]]$Dataset <- factor(Dataset_list_morph[[x]], levels = c("Train", "Test"))
  full_reg_morph_to_morph[[x]] <- getUmapTableDisperse(umap_morph_to_morph[[x]][,c(-1,-2,-3, -4)])
  full_reg_morph_to_morph[[x]]$Drug <- Drug_umap_list_morph_to_morph[[x]]
  full_reg_morph_to_morph[[x]]$Morph_knn_consen <- Broad_Cellular_Target_list_morph_to_morph[[x]]
  full_reg_morph_to_morph[[x]]$Dataset <- factor(Dataset_list_morph_to_morph[[x]], levels = c("Train", "Test"))
  full_reg_rna_to_morph[[x]] <- getUmapTableDisperse(umap_rna_to_morph[[x]][,c(-1,-2,-3, -4)])
  full_reg_rna_to_morph[[x]]$Drug <- Drug_umap_list_rna_to_morph[[x]]
  full_reg_rna_to_morph[[x]]$Morph_knn_consen <- Broad_Cellular_Target_list_rna_to_morph[[x]]
  full_reg_rna_to_morph[[x]]$Dataset <- factor(Dataset_list_rna_to_morph[[x]], levels = c("Train", "Test"))
  # full_reg_rna_true[[x]] <- getUmapTableDisperse(rna[[x]][,c(-1,-2,-3, -4, -5)])
  # full_reg_rna_true[[x]]$Drug <- Drug_umap_list_rna_true[[x]]
  # full_reg_rna_true[[x]]$Morph_knn_consen <- Broad_Cellular_Target_list_rna_true[[x]]
  # full_reg_rna_true[[x]]$Dataset <- factor(Dataset_list_rna_true[[x]], levels = c("Train", "Test"))
  # full_reg_rna_true[[x]]$Modality <- factor(Modality[[x]], levels = c("RNA-seq Profile", "RNA-seq Profile Translated from Morphological Profile"))
  
}

for (x in c(1:length(full_reg_rna))) {
  ggplot(full_reg_rna[[x]], aes(x=V1, y=V2, color=Drug, alpha = Dataset)) + geom_point()+scale_alpha_discrete(range=c(0.25, 1))+theme_classic()
  ggsave(filename=paste(save_folder_UMAP, "UMAP_rna_to_rna_", as.character(x), ".svg", sep=""), width=6, height=5)
  ggplot(full_reg_morph[[x]], aes(x=V1, y=V2, color=Drug, alpha = Dataset)) + geom_point()+scale_alpha_discrete(range=c(0.25, 1))+theme_classic()
  ggsave(filename=paste(save_folder_UMAP, "UMAP_morph_to_rna_", as.character(x), ".svg", sep=""), width=6, height=5)
  ggplot(full_reg_morph_to_morph[[x]], aes(x=V1, y=V2, color=Drug, alpha = Dataset)) + geom_point()+scale_alpha_discrete(range=c(0.25, 1))+theme_classic()
  ggsave(filename=paste(save_folder_UMAP, "UMAP_morph_to_morph_", as.character(x), ".svg", sep=""), width=6, height=5)
  ggplot(full_reg_rna_to_morph[[x]], aes(x=V1, y=V2, color=Drug, alpha = Dataset)) + geom_point()+scale_alpha_discrete(range=c(0.25, 1))+theme_classic()
  ggsave(filename=paste(save_folder_UMAP, "UMAP_rna_to_morph_", as.character(x), ".svg", sep=""), width=6, height=5)
  ggplot(full_reg_morph_to_morph[[x]], aes(x=V1, y=V2, color=Drug, alpha = Dataset)) + geom_point()+scale_alpha_discrete(range=c(0.25, 1))+theme_classic()
  ggsave(filename=paste(save_folder_UMAP, "UMAP_rna_to_morph_", as.character(x), ".svg", sep=""), width=6, height=5)
  # ggplot(full_reg_rna_true[[x]], aes(x=V1, y=V2, color=Drug, alpha = Dataset, shape = Modality)) + geom_point()+scale_alpha_discrete(range=c(0.25, 1))+theme_classic()
  # ggsave(filename=paste(save_folder_UMAP, "UMAP_rna_true_translated_morph", as.character(x), ".svg", sep=""), width=6, height=5)
}


# drugs <- unique(original$Drug)
drugs_morph <- list()
drugs_morph <- lapply(c(1:length(morph_to_rna)), function(x){
  drugs_morph[[x]] <- unique(morph_to_rna[[x]]$Drug)
  # remove untreated from drug lists
  drugs_morph[[x]] <- drugs_morph[[x]][! drugs_morph[[x]] %in% c('UNT')]
})

drugs_rna <- list()
drugs_rna <- lapply(c(1:length(rna_to_rna)), function(x){
  drugs_rna[[x]] <- unique(rna_to_rna[[x]]$Drug)
  # remove untreated from drug lists
  drugs_rna[[x]] <- drugs_rna[[x]][! drugs_rna[[x]] %in% c('UNT')]
})

drugs_original <- list()
drugs_original <- lapply(c(1:length(original)), function(x){
  drugs_original[[x]] <- unique(original[[x]]$Drug)
  # remove untreated from drug lists
  drugs_original[[x]] <- drugs_original[[x]][! drugs_original[[x]] %in% c('UNT')]
})

########

get_gene_set <- function(doi) { 
  deseq_result <- list.files(path = geneset_dir, pattern = doi, full.names = T) %>% read_csv(show_col_types = F)
  print(head(deseq_result))
  gene_set_df <- deseq_result %>% dplyr::filter(locus_tag %in% tags[[1]]) %>% slice_min(padj, n=300, with_ties=FALSE) %>% slice_max(abs(log2FoldChange), n=100, with_ties=FALSE)
  gene_set <- gene_set_df$locus_tag
  return(gene_set)
}

rna_genesets <- list()
rna_genesets <- lapply(c(1:length(drugs_rna)), function(x){
  rna_genesets[[x]] <- lapply(drugs_rna[[x]], get_gene_set)
  drugnames <- drugs_rna[[x]]
  names(rna_genesets[[x]]) <- drugnames
  return(rna_genesets[[x]])
})

morph_genesets <- list()
morph_genesets <- lapply(c(1:length(drugs_morph)), function(x){
  morph_genesets[[x]] <- lapply(drugs_morph[[x]], get_gene_set)
  drugnames <- drugs_morph[[x]]
  names(morph_genesets[[x]]) <- drugnames
  return(morph_genesets[[x]])
})

generate_correlations <- function(drugs, og_data, trans_data, geneset) {
  results_list <- list()
  results <- numeric()
  #drugs <- drugs[drugs %in% trans_data$Drug]
  for (g in drugs) {
    results <- numeric()
    og_filter <- og_data %>% dplyr::filter(Drug %in% drugs) %>% select(Drug, any_of(unlist(geneset[g]))) %>% group_by(Drug) %>% summarise(across(where(is.numeric), ~ median(.x, na.rm = TRUE)))
    trans_filter <- trans_data %>% dplyr::filter(Drug %in% drugs) %>% select(Drug, any_of(unlist(geneset[g]))) %>% group_by(Drug) %>% summarise(across(where(is.numeric), ~ median(.x, na.rm = TRUE)))
    trans_filter <- as.data.frame(trans_filter)
    if (nrow(og_filter) != nrow(trans_filter)) {
      trans_rows <- nrow(trans_filter)
      og_rows <- nrow(og_filter)
      for (i in c(1:(og_rows - trans_rows))) {
        trans_filter[trans_rows+i,] <- NA
      }
    }
    trans_filter <- trans_filter[match(og_filter$Drug, trans_filter$Drug),]
    rownames(trans_filter) <- rownames(og_filter)
    trans_filter$Drug <- og_filter$Drug
    trans_filter.t <- t(trans_filter)
    og_filter.t <- t(og_filter)
    colnames(trans_filter.t) <- trans_filter$Drug
    colnames(og_filter.t) <- og_filter$Drug
    for (d in drugs){
      current_result <- cor(x=as.numeric(trans_filter.t[-1,d]), y=as.numeric(og_filter.t[-1,g]), use = "everything")
      results <- c(results, current_result)
    }
    results_list[[g]] <- results
  }
  return(results_list)
}

generate_cor_matrix <- function(cors, drugs, meta) {
  cor_matrix <- matrix(unlist(cors), ncol=length(drugs), nrow=length(drugs))
  rownames(cor_matrix) <- drugs
  colnames(cor_matrix) <- drugs
  mechanism_df <- meta %>% select(Drug, cluster_marker_lcd) %>% distinct(Drug, cluster_marker_lcd) %>% dplyr::filter(Drug %in% drugs) %>% column_to_rownames(var="Drug") %>% rename(Mechanism = cluster_marker_lcd) %>% arrange(Mechanism)
  cor_matrix_ordered <- cor_matrix[rownames(mechanism_df), rownames(mechanism_df)]
  return(cor_matrix_ordered)
}
harmonic_mean <- function(cor_vector) {
  hmean <- 1/mean(1/cor_vector, na.rm = T)
  return(hmean)
}
arithmetic_mean <- function(cor_vector) {
  amean <- mean(cor_vector, na.rm = T)
  return(amean)
}

generate_plots <- function(cor_matrix, meta, name) {
  transposed <- data.table::transpose(cor_matrix)
  hmean_list <- lapply(transposed, arithmetic_mean)
  output <- matrix(unlist(hmean_list), ncol = length(rownames(cor_matrix[[1]])), byrow = FALSE)
  rownames(output) <- rownames(cor_matrix[[1]])
  colnames(output) <- colnames(cor_matrix[[1]])
  mechanism_df <- meta %>% select(Drug, cluster_marker_lcd) %>% distinct(Drug, cluster_marker_lcd) %>% column_to_rownames(var="Drug") %>% arrange(cluster_marker_lcd)
  mechanism_df$cluster_marker_lcd <-  as.character(mechanism_df$cluster_marker_lcd)
  output %>% pheatmap(scale="column", cluster_rows = FALSE, cluster_cols = FALSE, height=5, width=9, annotation_col = mechanism_df, filename=paste(save_folder_corr_plots, name, "_cor_heatmap.pdf", sep=""),  color = colorRampPalette(c("navy", "white", "firebrick3"))(50))
}
  
rna_correlations <- list()
morph_correlations <- list()
for (x in c(1:length(rna_to_rna))) {
  rna_correlations[[x]] <- generate_correlations(drugs=drugs_rna[[x]], og_data=original[[x]], trans_data=rna_to_rna[[x]], geneset=rna_genesets[[x]])
  morph_correlations[[x]] <- generate_correlations(drugs=drugs_morph[[x]], og_data=original[[x]], trans_data=morph_to_rna[[x]], geneset=morph_genesets[[x]])
}
rna_cor_matrix <- list()
morph_cor_matrix <- list()
for (x in c(1:length(rna_correlations))) {
  rna_cor_matrix[[x]] <- generate_cor_matrix(cors=rna_correlations[[x]], drugs=drugs_rna[[x]], meta=RNA_meta)
  morph_cor_matrix[[x]] <- generate_cor_matrix(cors=morph_correlations[[x]], drugs=drugs_morph[[x]], meta=morph_meta)
}
generate_plots(cor_matrix=rna_cor_matrix, meta=RNA_meta, name="rna_to_rna_train")
generate_plots(cor_matrix=morph_cor_matrix, meta=morph_meta, name="morph_to_rna_train")

rna_correlations <- list()
morph_correlations <- list()
for (x in c(1:length(rna_to_rna_test))) {
  rna_correlations[[x]] <- generate_correlations(drugs=drugs_rna[[x]], og_data=original[[x]], trans_data=rna_to_rna_test[[x]], geneset=rna_genesets[[x]])
  morph_correlations[[x]] <- generate_correlations(drugs=drugs_morph[[x]], og_data=original[[x]], trans_data=morph_to_rna_test[[x]], geneset=morph_genesets[[x]])
}
rna_cor_matrix <- list()
morph_cor_matrix <- list()
for (x in c(1:length(rna_correlations))) {
  rna_cor_matrix[[x]] <- generate_cor_matrix(cors=rna_correlations[[x]], drugs=drugs_rna[[x]], meta=RNA_meta)
  morph_cor_matrix[[x]] <- generate_cor_matrix(cors=morph_correlations[[x]], drugs=drugs_morph[[x]], meta=morph_meta)
}
generate_plots(cor_matrix=rna_cor_matrix, meta=RNA_meta, name="rna_to_rna_test")
generate_plots(cor_matrix=morph_cor_matrix, meta=morph_meta, name="morph_to_rna_test")
