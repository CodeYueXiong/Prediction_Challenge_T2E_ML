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
train_data_original[2] <- NULL
train_data_wo_d2 <- train_data_original

# check the data without dataframe 2
str(train_data_wo_d2)
length(train_data_wo_d2)

# create the corresponding tasks regarding the training datasets
tsks_train_list_wo_d2 = imap(
  train_data_wo_d2, # competing risks case
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
lrn_aft_wo_d2 = lrn("surv.mboost", baselearner = 'bols')

lrn_aft_wo_d2$param_set$values(list(importance = "impurity"))
library(filter)
filter  
# train the aft boosting learner
aft_list_wo_d2<- map(
  tsks_train_list_wo_d2,
  ~lrn_aft_wo_d2$train(.x)
)



# save submission for semi-parametric aft boosting
saveRDS(aft_list_wo_d2, "submissions/aft_wo_d2_list_2021-02-15.Rds")

# sub_aft <- readRDS("submissions/semi_aft_list_2021-01-11.Rds")
# str(aft_list)

# evaluate the performance score

