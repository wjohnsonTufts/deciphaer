options(warn=-1)
sink("r_out.txt")

suppressMessages({
  library(optparse, verbose = F)
  library(readr,verbose = F)
  library(class,verbose = F)
  library(tidyverse,verbose = F, quietly = T)
  library(ggplot2,verbose = F)
  library(rstatix)
  library(plotly)
  library(ggpubr)
  library(htmlwidgets)
  library(htmltools)
  library(gplots)
  library(RColorBrewer)
  library(gridExtra)
  library(grid)
})

option_list = list(
  make_option(c("-d", "--dir_ls"), type="character", default="./dirk_final_1/", 
              help="run directory", metavar="character"),
  make_option(c("-z", "--z_size"), type="character", default="50", 
              help="latent dimension number", metavar="character")
  # make_option(c("-m", "--meta_data"), type="character", default="/Users/aalivi01/Projects/multiomics-moa/deciphaer/data/v19", 
              # help="Directory for metadata", metavar="character")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);
path_to_latent <- opt$d
latent_dims <- as.integer(opt$z)
# meta_dir <- paste0(opt$m, "/")
save_folder <- paste0(path_to_latent, "model_inference/")
dir.create(save_folder, showWarnings = FALSE)

m_latent <- read_csv(paste0(path_to_latent, "latent_vis_morph_train.csv"), show_col_types = F)
r_latent <- read_csv(paste0(path_to_latent, "latent_vis_rna_train.csv"), show_col_types = F)
m_validation <- read_csv(paste0(path_to_latent, "latent_vis_morph_test.csv"), show_col_types = F)
m_tran_validation <- read_csv(paste0(path_to_latent, "morph_to_rna_test.csv"), show_col_types = F)
m_tran <- read_csv(paste0(path_to_latent, "morph_to_rna_train.csv"), show_col_types = F)
# rnaseq_og <- read_csv(paste0(meta_dir, "rna_data.csv"), show_col_types = F)

r_latent <- r_latent%>%
  mutate(Modality = "RNA", split = "Train")
m_latent <- m_latent%>%
  mutate(Modality = "Morph", split = "Train")
m_validation <- m_validation%>%
  mutate(Modality = "Morph", split = "Test")
m_tran_validation<- m_tran_validation%>%
  mutate(Modality = "Morph", split = "Test")
m_tran<- m_tran%>%
  mutate(Modality = "Morph", split = "Train")

# generate latent embedding UMAP
completeData <- m_latent%>%
  dplyr::bind_rows(r_latent, m_validation)
completeData$Label <- as.character(completeData$Label)
completeData$Drug <- gsub(completeData$Cell_id, pattern  = ".*M70.*", replacement = "M70")
completeData$Drug <- gsub(completeData$Drug, pattern  = "R.\\-(...)\\-T.*", replacement = "\\1")
completeData$Drug <- gsub(completeData$Drug, pattern  = ".*_(...)_.*", replacement = "\\1")

# subset the untreated to avoid overwhelming the visualization
Untreated <- completeData %>%
  dplyr::filter(Drug == "UNT")%>%
  sample_n(10)
completeData <- completeData %>%
  filter(Drug != "UNT")%>%
  bind_rows(Untreated)

umapedData <- umap::umap((completeData%>%dplyr::select(as.character(c(0:(latent_dims-1))))))
id <- completeData$Cell_id
umapedData <- as.data.frame(umapedData$layout)
umapedData$Cell_id <- id
umapedData <- umapedData %>%
  dplyr::full_join(completeData, by = "Cell_id")
umapedData$split <-relevel(as.factor(umapedData$split), ref = "Train")
p <- ggplotly(ggplot(umapedData, aes(x = V1, y=V2, color = Drug, shape = Modality))+
                geom_point(aes(alpha = split, id = Drug))+
                scale_alpha_discrete(c(1,0.75))+
                labs(color = "Integration Label"))
paste0(save_folder, "VAE_Latent_Embedding")
saveWidget(widget = p, file = paste0(save_folder, "VAE_Latent_Embedding.html"), selfcontained = F, title = "Integrated Latent Space")

# generate decodings UMAP
completeData <- m_tran%>%
  dplyr::bind_rows(m_tran_validation)
completeData$Label <- as.character(completeData$Label)
completeData$Drug <- gsub(completeData$Cell_id, pattern  = ".*M70.*", replacement = "M70")
completeData$Drug <- gsub(completeData$Drug, pattern  = "R.\\-(...)\\-T.*", replacement = "\\1")
completeData$Drug <- gsub(completeData$Drug, pattern  = ".*_(...)_.*", replacement = "\\1")
# subset the untreated to avoid overwhelming the visualization
Untreated <- completeData %>%
  dplyr::filter(Drug == "UNT")%>%
  sample_n(10)
completeData <- completeData %>%
  filter(Drug != "UNT")%>%
  bind_rows(Untreated)
umapedData <- umap::umap((completeData%>%dplyr::select(-Label, -Cell_id, -Drug,-split)%>%dplyr::select_if(is.numeric)))
id <- completeData$Cell_id
umapedData <- as.data.frame(umapedData$layout)
umapedData$Cell_id <- id
umapedData <- umapedData %>%
  dplyr::full_join(completeData, by = "Cell_id")
umapedData$split <-relevel(as.factor(umapedData$split), ref = "Train")
p <- ggplotly(ggplot(umapedData, aes(x = V1, y=V2, color = Drug, shape = Modality))+
                geom_point(aes(alpha = split, id = Drug))+
                scale_alpha_discrete(c(1,0.75))+
                labs(color = "Integration Label"))
saveWidget(widget = p, file = paste0(save_folder, "VAE_Morph_to_RNA_Decodings.html"), selfcontained = F, title = "Decoded Morph to RNA space")

# T tests etc can go here or in a separate file.
