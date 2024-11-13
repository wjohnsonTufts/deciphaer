options(warn=-1)
sink("r_out.txt")

suppressMessages({
  library(optparse, verbose = F)
  library(umap, verbose = F)
  library(RColorBrewer,verbose = F)
  library(readr,verbose = F)
  library(class,verbose = F)
  library(tidyverse,verbose = F, quietly = T)
  library(ggplot2,verbose = F)
})

option_list = list(
  make_option(c("-d", "--dir_ls"), type="character", default="/Users/wjohns07/Desktop/GitLab/deciphaer/Models_Analysis/LCD/dirk_final_1/", 
              help="run directory", metavar="character"),
  make_option(c("-z", "--z_size"), type="character", default="50", 
              help="latent dimension number", metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);
#Here are the arguments for the R script. This will need to be run once for each hyperparameter setting. It will go through all the training and test
knns_morph_apply <- c(1,2,3,4,5)
knns_rna_apply <- c(1,2,3,4,5)
pc_dim <- 5
path_to_latent <- opt$d
latent_dims <- as.integer(opt$z)
save_folder <- paste0(path_to_latent, "KNN_F_stat/")
dir.create(save_folder, showWarnings = FALSE)

#The input it expects is the latent_vis_morph.csv and latent_vis_rna.csv and it will output a graph for the KNN accuracy at different Ks
morph_train <- list.files(path = path_to_latent, pattern = "latent_vis_morph_train", full.names = T, recursive = T)
morph_train <- lapply(morph_train, function(x){
  read_csv(x, show_col_types =F )
})
rna_train <- list.files(path = path_to_latent, pattern = "latent_vis_rna_train", full.names = T, recursive = T)
rna_train <- lapply(rna_train, function(x){
  read_csv(x, show_col_types =F )
})
morph_test <- list.files(path = path_to_latent, pattern = "latent_vis_morph_test", full.names = T, recursive = T)
morph_test <- lapply(morph_test, function(x){
  read_csv(x, show_col_types =F )
})
rna_test <- list.files(path = path_to_latent, pattern = "latent_vis_rna_test", full.names = T, recursive = T)
rna_test <- lapply(rna_test, function(x){
  read_csv(x, show_col_types =F )
})
merged_data_train <- lapply(c(1:length(rna_train)), function(x){
  morph_train[[x]] <- morph_train[[x]] %>% mutate(Source="morph") %>% dplyr::rename(id=Cell_id)
  rna_train[[x]] <- rna_train[[x]] %>% mutate(Source="rna") %>% dplyr::rename(id=Cell_id)
  rbind(morph_train[[x]], rna_train[[x]]) %>%dplyr::select(Source, everything())
})
merged_data_test <- lapply(c(1:length(rna_test)), function(x){
  morph_test[[x]] <- morph_test[[x]] %>% mutate(Source="morph") %>% dplyr::rename(id=Cell_id)
  rna_test[[x]] <- rna_test[[x]] %>% mutate(Source="rna") %>% dplyr::rename(id=Cell_id)
  rbind(morph_test[[x]], rna_test[[x]]) %>%dplyr::select(Source, everything())
})
set.seed(123)
train_rna <- lapply(c(1:length(merged_data_train)), function(x){
  classMin <- min(table((merged_data_train[[x]]%>% dplyr::filter(Source == "rna"))$Label)) 
  balTrain <- lapply(c(unique((merged_data_train[[x]]%>% dplyr::filter(Source == "rna"))$Label)), function(lab){
    sample_n((merged_data_train[[x]] %>%dplyr::filter(Source == "rna", Label == lab)), size = classMin,replace = F)
  })
  reduce(balTrain, bind_rows)
})
test_morph <- lapply(c(1:length(merged_data_test)), function(x){
  Test <- lapply(c(unique((merged_data_test[[x]]%>% dplyr::filter(Source == "morph"))$Label)), function(lab){
    (merged_data_test[[x]] %>%dplyr::filter(Source == "morph", Label == lab))
  })
  reduce(Test, bind_rows)
})
print(length(test_morph))
knnTab <- data.frame(k_num = knns_morph_apply, accuracy = c(1:length(knns_morph_apply)))
nested_accuracy <- lapply(c(1:length(train_rna)), function(data_set_num){
  test_morph[[data_set_num]]$Split <- "Test"
  train_rna[[data_set_num]]$Split <- "Train"
  completeData <- bind_rows(test_morph[[data_set_num]], train_rna[[data_set_num]])
  umapedData <- umap::umap(completeData%>%dplyr::select(as.character(c(0:(latent_dims-1)))))
  id <- completeData$id
  umapedData <- as.data.frame(umapedData$layout)
  umapedData$id <- id
  umapedData <- umapedData %>%
    dplyr::full_join(completeData, by = "id")
  train_cl <- scale((umapedData%>%filter(Split == "Train")%>%dplyr::select(V1,V2)), center = T, scale = T)
  test_cl <- scale((umapedData%>%filter(Split == "Test")%>%dplyr::select(V1,V2)), center = T, scale = T)
  test_cat <- as.character((umapedData%>%filter(Split == "Test"))$Label)
  target_cat <- as.character((umapedData%>%filter(Split == "Train"))$Label)
  accuracy <- lapply(c(1:nrow(knnTab)), function(x){
    pr <- knn(train_cl,test_cl,cl=target_cat,k=as.integer(knnTab$k_num[x]))
    pr <- as.character(pr)
    test_cat <-as.character(test_cat)
    sum(test_cat==pr)/length(pr)*100
  })
  unlist(accuracy)
})
PlotTable <- lapply(c(1:length(nested_accuracy)), function(x){
  ret <- data.frame(accuracy = unlist(nested_accuracy[[x]]), k_num =  c(1:length(unlist(nested_accuracy[[x]]))))
  ret$split <-x
  ret
})
PlotTable <- as.data.frame(reduce(PlotTable, bind_rows))
write_csv(paste0(save_folder,"Knn_morph_against_rna_table.csv"), x = PlotTable)
set.seed(123)
train_rna <- lapply(c(1:length(merged_data_train)), function(x){
  classMin <- min(table((merged_data_train[[x]]%>% dplyr::filter(Source == "morph"))$Label)) 
  balTrain <- lapply(c(unique((merged_data_train[[x]]%>% dplyr::filter(Source == "morph"))$Label)), function(lab){
    sample_n((merged_data_train[[x]] %>%dplyr::filter(Source == "morph", Label == lab)), size = classMin,replace = F)
  })
  reduce(balTrain, bind_rows)
})
test_morph <- lapply(c(1:length(merged_data_test)), function(x){
  Test <- lapply(c(unique((merged_data_test[[x]]%>% dplyr::filter(Source == "rna"))$Label)), function(lab){
    (merged_data_test[[x]] %>%dplyr::filter(Source == "rna", Label == lab))
  })
  reduce(Test, bind_rows)
})
knnTab <- data.frame(k_num = knns_morph_apply, accuracy = c(1:length(knns_morph_apply)))
nested_accuracy <- lapply(c(1:length(train_rna)), function(data_set_num){
  test_morph[[data_set_num]]$Split <- "Test"
  train_rna[[data_set_num]]$Split <- "Train"
  completeData <- bind_rows(test_morph[[data_set_num]], train_rna[[data_set_num]])
  umapedData <- umap::umap(completeData%>%dplyr::select(as.character(c(0:(latent_dims-1)))))
  id <- completeData$id
  umapedData <- as.data.frame(umapedData$layout)
  umapedData$id <- id
  umapedData <- umapedData %>%
    dplyr::full_join(completeData, by = "id")
  train_cl <- scale((umapedData%>%filter(Split == "Train")%>%dplyr::select(V1,V2)), center = T, scale = T)
  test_cl <- scale((umapedData%>%filter(Split == "Test")%>%dplyr::select(V1,V2)), center = T, scale = T)
  test_cat <- as.character((umapedData%>%filter(Split == "Test"))$Label)
  target_cat <- as.character((umapedData%>%filter(Split == "Train"))$Label)
  accuracy <- lapply(c(1:nrow(knnTab)), function(x){
    pr <- knn(train_cl,test_cl,cl=target_cat,k=as.integer(knnTab$k_num[x]))
    pr <- as.character(pr)
    test_cat <-as.character(test_cat)
    sum(test_cat==pr)/length(pr)*100
  })
  unlist(accuracy)
})
PlotTable <- lapply(c(1:length(nested_accuracy)), function(x){
  ret <- data.frame(accuracy = unlist(nested_accuracy[[x]]), k_num =  c(1:length(unlist(nested_accuracy[[x]]))))
  ret$split <-x
  ret
})
PlotTable <- as.data.frame(reduce(PlotTable, bind_rows))
write_csv(paste0(save_folder,"Knn_rna_against_morph_table.csv"), x = PlotTable)
#Now get the F statistic for each train test umapped latent embeddings. We need to combine the train and test data together in order to calculate the F
#F-test = F=S2A/S2B
nest_fstat <- lapply(c(1:length(merged_data_test)), function(data_set_num){
  merged_data_test[[data_set_num]]$Split <- "Test"
  merged_data_train[[data_set_num]]$Split <- "Train"
  completeData <- bind_rows(merged_data_test[[data_set_num]], merged_data_train[[data_set_num]])
  completeData%>%dplyr::select(as.character(c(0:(latent_dims-1))))
  umapedData <- umap::umap((completeData%>%dplyr::select(as.character(c(0:(latent_dims-1))))))
  id <- completeData$id
  umapedData <- as.data.frame(umapedData$layout)
  umapedData$id <- id
  umapedData <- umapedData %>%
    dplyr::full_join(completeData, by = "id")
  f_train_1 <- summary(aov(V1 ~Label, data = (umapedData%>%filter(Split == "Train"))))
  f_train_2 <- summary(aov(V2 ~Label, data = (umapedData%>%filter(Split == "Train"))))
  f_test_1 <- summary(aov(V1 ~Label, data = (umapedData%>%filter(Split == "Test"))))
  f_test_2 <- summary(aov(V2 ~Label, data = (umapedData%>%filter(Split == "Test"))))
  f_all_1 <- summary(aov(V1 ~Label, data = umapedData))
  f_all_2 <- summary(aov(V2 ~Label, data = umapedData))
  data.frame(f_train = mean(f_train_1[[1]]$`F value`[1], f_train_2[[1]]$`F value`[1]),f_test =mean(f_test_1[[1]]$`F value`[1], f_test_2[[1]]$`F value`[1]), f_all = mean(f_all_1[[1]]$`F value`[1], f_all_2[[1]]$`F value`[1]), Split = data_set_num)
})
nest_fstat <- reduce(nest_fstat, bind_rows)
write_csv(paste0(save_folder,"F_stat_table.csv"), x = nest_fstat)
sink()