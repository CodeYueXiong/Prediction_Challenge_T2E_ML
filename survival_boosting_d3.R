##################################################
##################################################
########## Boosting in COX Regression ############
##################################################
##################################################

#==========================================================================================

# clear all workspace
rm(list=ls()) 
# install the necessary packages
library(mlr3)
library(mlr3proba)
library(mlr3learners)
library(mlr3extralearners)
library(data.table)
library(mlr3viz)
library(mlr3tuning)
library(mlr3pipelines)
library(purrr)
library(mboost)
library(CoxBoost)
library(survival)
## library(xgboost)

# set or change R working directory
setwd("/Users/echo/Desktop/1_WS20/1_T2E_ML/2_Prediction_Challenge/")

wd = dirname(rstudioapi::getActiveDocumentContext()$path)

##############################################################
##########      Step 1: Default training set      ############
##############################################################

# ----- train without the d2 dataframe ------- #
# read in the raw data
train_data_original <- readRDS("Data-20210107/train_data.Rds")
test_data <- readRDS("Data-20210107/test_list_x.Rds")

length(train_data_original)
# split the train data 
# especially with the 2nd dataset
train_data_d2 <- train_data_original[2] # store the d2 dataframe
# str(train_data_d2)
# train_data_original[2] <- NULL
# train_data_wo_d2 <- train_data_original

# check the data without dataframe 2
str(train_data_original)
length(train_data_original) # no dataframe 6


dim(train_data_original$d7)
# check again for NA values
any(is.na(train_data_original$d7))

head(train_data_original[[6]])

# tsks_train_list <- list(
#   TaskSurv$new("df1", backend = train_data_original[[1]], time = "time", event = "status"),
#   TaskSurv$new("df2", backend = train_data_original[[2]], time = "time", event = "status"),
#   TaskSurv$new("df3", backend = train_data_original[[3]], time = "time", event = "status"),
#   TaskSurv$new("df4", backend = train_data_original[[4]], time = "time", event = "status"),
#   TaskSurv$new("df5", backend = train_data_original[[5]], time = "time", event = "status"),
#   TaskSurv$new("df7", backend = train_data_original[[6]], time = "time", event = "status"),
#   TaskSurv$new("df8", backend = train_data_original[[7]], time = "time", event = "status"),
#   TaskSurv$new("df9", backend = train_data_original[[8]], time = "time", event = "status")
# )

############### missing data processing ##########
## --------------- delete the NA values ---------
# trying to drop the NA columns, with na.omit()
train_data_original$d7 <- na.omit(train_data_original$d7)
head(train_data_original$d7)

dim(train_data_original$d7)
# check for missing values again
is.na(train_data_original[[6]])

tsks_train_list = imap(
  train_data_original, # list of tasks
  ~TaskSurv$new(
    id = .y,
    backend = .x,
    time = "time",
    event = "status")
)

#-----------------------------------------------------------------------------------------------------
## Step 1_1: Boosting with AFT

## load the learner for the aft boosting
# ?mboost
# tsks_train_list[[6]] <- NULL
length(tsks_train_list)

design <- benchmark_grid(
  tasks = tsks_train_list,
  learners = lrn("surv.mboost", baselearner = 'bols'), # cannot compute ‘bbs’ for non-numeric variables; used ‘bols’ instead.
  resampling = rsmp("cv", folds = 3L)
)

# set global measure
all_measures <- msr("surv.graf")


# define function to start benchmark with fixed seed
run_benchmark <- function(design){
  set.seed(2021)
  bmr <- benchmark(design, store_models = TRUE)
  run_benchmark <- bmr
}

## run benchmark and save the results
aft_bmr <- run_benchmark(design)
aft_results <- aft_bmr$aggregate(measures = all_measures)

library(mlr3viz) 
library(precrec)
library(ggplot2)

head(fortify(aft_bmr))
fortify(aft_bmr)
autoplot(aft_bmr)




model_df1 <- aft_bmr$clone()$filter(task_id = "df1")
model_df1
cindex <- round(model_df1$aggregate(msr("surv.cindex"))[[7]], 4)
cindex
autoplot(model_df1, type = "boxplot") + ggtitle(paste("df1:", cindex))

model_df2 <- aft_bmr$clone()$filter(task_id = "df2")
cindex <- round(model_df2$aggregate(msr("surv.cindex"))[[7]], 4)
autoplot(model_df2, type = "boxplot") + ggtitle(paste("df2:", cindex))

# plot for the 6 data tasks
multiplot_cindex <- function(models, type = "boxplot") {
  # set a null list to colect all plots
  plots <- list()
  
  # filter model with task id, and the corresponding AUC value
  # then plot the ROC curve
  model <- models$clone()$filter(task_id = "df1")
  cindex <- round(model$aggregate(msr("surv.cindex"))[[7]], 4)
  plots[[1]] <- autoplot(model, type = type) + ggtitle(paste("df1:", cindex))
  
  model <- models$clone()$filter(task_id = "df2")
  cindex <- round(model$aggregate(msr("surv.cindex"))[[7]], 4)
  plots[[2]] <- autoplot(model) + ggtitle(paste("df2:", cindex))

  # model <- models$clone()$filter(task_id = "df3")
  # cindex <- round(model$aggregate(msr("surv.cindex"))[[7]], 4)
  # plots[[3]] <- autoplot(model) + ggtitle(paste("df3:", cindex))
  # 
  # model <- models$clone()$filter(task_id = "df4")
  # cindex <- round(model$aggregate(msr("surv.cindex"))[[7]], 4)
  # plots[[4]] <- autoplot(model) + ggtitle(paste("df4:", cindex))
  # 
  # model <- models$clone()$filter(task_id = "df5")
  # cindex <- round(model$aggregate(msr("surv.cindex"))[[7]], 4)
  # plots[[5]] <- autoplot(model) + ggtitle(paste("df5:", cindex))
  # 
  # model <- models$clone()$filter(task_id = "df7")
  # cindex <- round(model$aggregate(msr("surv.cindex"))[[7]], 4)
  # plots[[6]] <- autoplot(model) + ggtitle(paste("df7", cindex))
  # 
  # model <- models$clone()$filter(task_id = "df8")
  # cindex <- round(model$aggregate(msr("surv.cindex"))[[7]], 4)
  # plots[[6]] <- autoplot(model) + ggtitle(paste("df8", cindex))
  
  # merge all plots in the list
  do.call("grid.arrange", plots)
}

# plot roc curve of all tasks
multiplot_cindex(aft_bmr)


