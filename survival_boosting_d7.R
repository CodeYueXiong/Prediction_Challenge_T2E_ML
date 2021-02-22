##################################################
##################################################
########## Boosting in COX Regression ############
################## dataframe1 ####################
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
library("paradox")
library("ggrepel")
library(precrec)
library(ggplot2)
library("ggpubr")
## library(xgboost)

# set or change R working directory
setwd("/Users/echo/Desktop/1_WS20/1_T2E_ML/2_Prediction_Challenge/")

wd = dirname(rstudioapi::getActiveDocumentContext()$path)

##############################################################
##########    Step 1: Default training set as d1    ##########
##############################################################

# read in the raw data
train_data_original <- readRDS("Data-20210107/train_data.Rds")
test_data <- readRDS("Data-20210107/test_list_x.Rds")

length(train_data_original)

## get the corresponding dataframe1
train_data_d7 <- train_data_original$d7
train_data_d7 <- na.omit(train_data_d7)
head(train_data_d7)
summary(train_data_d7$V10)
summary(train_data_d7$V1)
summary(train_data_d7$V2) # problem with V2, more than 53 categories
summary(train_data_d7$V5)
summary(train_data_d7$V7)
######### missing value: data imputation ##########
### ---- use missForest ---- #####
## Nonparametric missing value imputation on mixed-type data:
## Take a look at iris definitely has a variable that is a factor 
library(missForest)

## Impute missing values providing the complete matrix for
## illustration. Use 'verbose' to see what happens between iterations:
d7.imp <- missForest(train_data_d7, verbose = TRUE) # Can not handle categorical predictors with more than 53 categories

## Here are the final results
iris.imp

## As can be seen here it still has the factor column
str(iris.imp$ximp)


## create the corresponding task as for dataframe 1
tsks_train_d7 <- TaskSurv$new("df7", backend = train_data_d7, time = "time", event = "status")

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

## check the task
tsks_train_d7

#---------------------------Fine Tuning--------------------------------------------------------------------------
## Step 1: Boosting with AFT


## load the learner for the aft boosting and design the benchmark
## Step 1_1: Try with a default setting
design <- benchmark_grid(
  tasks = tsks_train_d7,
  learners = lrn("surv.mboost", baselearner = 'bols', family = "weibull"), # cannot compute ‘bbs’ for non-numeric variables; used ‘bols’ instead.
  resampling = rsmp("cv", folds = 3L)
)


# define function to start benchmark with fixed seed
run_benchmark <- function(design){
  set.seed(2021)
  bmr <- benchmark(design, store_models = TRUE)
  run_benchmark <- bmr
}

## run benchmark and save the results
aft_bmr <- run_benchmark(design)

# set the measure
time = train_data_d1$time[train_data_d1$status == 1]
quantile = quantile(time, probs = 0.5)

# set the global evaluation metric
all_measures <- msr("surv.cindex")

aft_results <- aft_bmr$aggregate(measures = all_measures)
aft_results
# plot the corresponding the performance
ggplot(data = aft_results, aes(x = learner_id, y = surv.harrell_c, label = round(surv.harrell_c, 4))) +
                            geom_point() + geom_text_repel() +
                            ggtitle("step1_1 aft with a default parameter setting")
  
## Step_1_2: model tuning with the "baselearner"


# load the learner with aft
lrn_aft <- lrn("surv.mboost", baselearner = 'bols', family = "weibull")
lrn("surv.mboost", baselearner = 'bols')$param_set
# train with "baselearner"
base_learner_type <- c("bols", "btree")

# set the search space
param_aft_bl <- ParamSet$new(params = list(
  ParamFct$new("baselearner", levels = base_learner_type)
))

# inner resampling set
inner_rsmp <- rsmp("cv", folds = 4L)

# create the AutoTuner
aft_bl <- AutoTuner$new(
  learner = lrn_aft, resampling =  inner_rsmp,
  measure = msr("surv.cindex"), search_space = param_aft_bl,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 5)
)
# set the outer resampling
outer_rsmp <- rsmp("cv", folds = 3L)
# design the benchmark with bf
design_aft_bl <- benchmark_grid(
  tasks = tsks_train_d1,
  learners = aft_bl,
  resamplings = outer_rsmp
)

bmr_aft_bl <- run_benchmark(design_aft_bl)

# aggregate to get results of model when tuning with bf
bmr_aft_bl_results <- bmr_aft_bl$aggregate(measures = msr("surv.cindex"))
# load install.packages("ggrepel")


# plot the corresponding performances
aft_bl_path1 <- bmr_aft_bl$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
aft_bl_ggp1 <- ggplot(aft_bl_path1, aes(
  x = baselearner,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()

aft_bl_path2 <- bmr_aft_bl$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
aft_bl_ggp2 <- ggplot(aft_bl_path2, aes(
  x = baselearner,
  y = surv.harrell_c, #, col = factor(family)
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()
  # geom_text(aes(label = round(surv.harrell_c, 4)), size = 3, position = position_dodge(width=0.9))

aft_bl_path3 <- bmr_aft_bl$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
aft_bl_ggp3 <- ggplot(aft_bl_path3, aes(
  x = baselearner,
  y = surv.harrell_c,#, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


# arrange all the plots together
ggarrange(aft_bl_ggp1, aft_bl_ggp2, aft_bl_ggp3, common.legend = TRUE, legend = "bottom")

## Step_1_3: model tuning with the "family"

# load the learner with aft
lrn_aft <- lrn("surv.mboost", baselearner = 'bols', family = "weibull")

# train with "family"
family_type <- c("weibull", "loglog", "lognormal")

# set the search space
param_aft_fam <- ParamSet$new(params = list(
  ParamFct$new("family", levels = family_type)
))


# create the AutoTuner
aft_fam <- AutoTuner$new(
  learner = lrn_aft, resampling =  inner_rsmp,
  measure = msr("surv.cindex"), search_space = param_aft_fam,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 5)
)

# design the benchmark with bf
design_aft_fam <- benchmark_grid(
  tasks = tsks_train_d1,
  learners = aft_fam,
  resamplings = outer_rsmp
)

bmr_aft_fam <- run_benchmark(design_aft_fam)

# aggregate to get results of model when tuning with bf
bmr_aft_fam_results <- bmr_aft_fam$aggregate(measures = msr("surv.cindex"))
# plot the corresponding performances
aft_fam_path1 <- bmr_aft_fam$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
aft_fam_ggp1 <- ggplot(aft_fam_path1, aes(
  x = family,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()
  

aft_fam_path2 <- bmr_aft_fam$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
aft_fam_ggp2 <- ggplot(aft_fam_path2, aes(
  x = family,
  y = surv.harrell_c,#, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()

aft_fam_path3 <- bmr_aft_fam$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
aft_fam_ggp3 <- ggplot(aft_fam_path3, aes(
  x = family,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()

# arrange all the plots together
ggarrange(aft_fam_ggp1, aft_fam_ggp2, aft_fam_ggp3, common.legend = TRUE, legend = "bottom")

## Step_1_3_2: exclude "lognormal",and train between "weibull" and "loglog"
# load the learner with aft
lrn_aft <- lrn("surv.mboost", baselearner = 'bols', family = "weibull")

# train with "family"
family_type <- c("weibull", "loglog")

# set the search space
param_aft_fam <- ParamSet$new(params = list(
  ParamFct$new("family", levels = family_type)
))


# create the AutoTuner
aft_fam <- AutoTuner$new(
  learner = lrn_aft, resampling =  inner_rsmp,
  measure = msr("surv.cindex"), search_space = param_aft_fam,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 5)
)

# design the benchmark with fam
design_aft_fam <- benchmark_grid(
  tasks = tsks_train_d1,
  learners = aft_fam,
  resamplings = outer_rsmp
)

bmr_aft_fam <- run_benchmark(design_aft_fam)

# aggregate to get results of model when tuning with fam
bmr_aft_fam_results <- bmr_aft_fam$aggregate(measures = msr("surv.cindex"))
# plot the corresponding performances
aft_fam_path1 <- bmr_aft_fam$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
aft_fam_ggp1 <- ggplot(aft_fam_path1, aes(
  x = family,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


aft_fam_path2 <- bmr_aft_fam$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
aft_fam_ggp2 <- ggplot(aft_fam_path2, aes(
  x = family,
  y = surv.harrell_c,#, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()

aft_fam_path3 <- bmr_aft_fam$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
aft_fam_ggp3 <- ggplot(aft_fam_path3, aes(
  x = family,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


# arrange all the plots together
ggarrange(aft_fam_ggp1, aft_fam_ggp2, aft_fam_ggp3, common.legend = TRUE, legend = "bottom")

# conclusion: go with "weibull", as it is reaching the highest performance "0.847", compared with the highest "loglog" ones, 0.844..


## Step_1_4: model tuning with the "mstop", early stopping, 
# from 10 to 250, 50-250, 50-125, 50-90, 65-85, 65-75

# load the learner with aft
lrn_aft <- lrn("surv.mboost", baselearner = 'bols', family = "weibull")


# set the search space
param_aft_mstop <- ParamSet$new(params = list(
  ParamInt$new("mstop", lower = 65, upper = 75)
))


# create the AutoTuner
aft_mstop <- AutoTuner$new(
  learner = lrn_aft, resampling =  inner_rsmp,
  measure = msr("surv.cindex"), search_space = param_aft_mstop,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 10)
)

# design the benchmark with bf
design_aft_mstop <- benchmark_grid(
  tasks = tsks_train_d1,
  learners = aft_mstop,
  resamplings = outer_rsmp
)

bmr_aft_mstop <- run_benchmark(design_aft_mstop)

# aggregate to get results of model when tuning with bf
bmr_aft_mstop_results <- bmr_aft_mstop$aggregate(measures = msr("surv.cindex"))
# plot the corresponding performances
aft_mstop_path1 <- bmr_aft_mstop$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
aft_mstop_ggp1 <- ggplot(aft_mstop_path1, aes(
  x = mstop,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


aft_mstop_path2 <- bmr_aft_mstop$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
aft_mstop_ggp2 <- ggplot(aft_mstop_path2, aes(
  x = mstop,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


aft_mstop_path3 <- bmr_aft_mstop$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
aft_mstop_ggp3 <- ggplot(aft_mstop_path3, aes(
  x = mstop,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()



# arrange all the plots together
ggarrange(aft_mstop_ggp1, aft_mstop_ggp2, aft_mstop_ggp3, common.legend = TRUE, legend = "bottom")

# conclusion: set mstop to be 65

## Step_1_5_1: model tuning with the "nu", 
# from 0.00 to 1.00, 0.1-1.0, 0.1-0.4, 0.1-0.2

# load the learner with aft
lrn_aft <- lrn("surv.mboost", baselearner = 'bols', family = "weibull", mstop = 65)

# set the search space
param_aft_nu <- ParamSet$new(params = list(
  ParamDbl$new("nu", lower = 0.1, upper = 0.2)
))


# create the AutoTuner
aft_nu <- AutoTuner$new(
  learner = lrn_aft, resampling = inner_rsmp,
  measure = msr("surv.cindex"), search_space = param_aft_nu,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 5)
)

# design the benchmark with bf
design_aft_nu <- benchmark_grid(
  tasks = tsks_train_d1,
  learners = aft_nu,
  resamplings = outer_rsmp
)

bmr_aft_nu <- run_benchmark(design_aft_nu)

# aggregate to get results of model when tuning with bf
bmr_aft_nu_results <- bmr_aft_nu$aggregate(measures = msr("surv.cindex"))
# plot the corresponding performances
aft_nu_path1 <- bmr_aft_nu$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
aft_nu_ggp1 <- ggplot(aft_nu_path1, aes(
  x = nu,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


aft_nu_path2 <- bmr_aft_nu$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
aft_nu_ggp2 <- ggplot(aft_nu_path2, aes(
  x = nu,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


aft_nu_path3 <- bmr_aft_nu$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
aft_nu_ggp3 <- ggplot(aft_nu_path3, aes(
  x = nu,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


# arrange all the plots together
ggarrange(aft_nu_ggp1, aft_nu_ggp2, aft_nu_ggp3, common.legend = TRUE, legend = "bottom")

# conclusion: nu 0.2

## final learner with aft model
lrn_aft <- lrn("surv.mboost", baselearner = 'bols', family = "weibull", mstop = 65, nu = 0.2)

design_aft <- benchmark_grid(
  tasks = tsks_train_d1,
  learners = lrn_aft,
  resampling = rsmp("cv", folds = 3L)
)


# define function to start benchmark with fixed seed
run_benchmark <- function(design_aft){
  set.seed(2021)
  bmr <- benchmark(design_aft, store_models = TRUE)
  run_benchmark <- bmr
}

## run benchmark and save the results
aft_bmr <- run_benchmark(design_aft)

# set the global evaluation metric
all_measures <- msr("surv.cindex")

aft_results <- aft_bmr$aggregate(measures = all_measures)
aft_results$surv.harrell_c # 0.839

## --- load the learner for the mboost and design the benchmark
## Step 2_1: Try with a default setting
design <- benchmark_grid(
  tasks = tsks_train_d1,
  learners = lrn("surv.mboost", baselearner = 'bols', family = "coxph"), # cannot compute ‘bbs’ for non-numeric variables; used ‘bols’ instead.
  resampling = rsmp("cv", folds = 3L)
)


# define function to start benchmark with fixed seed
run_benchmark <- function(design){
  set.seed(2021)
  bmr <- benchmark(design, store_models = TRUE)
  run_benchmark <- bmr
}

## run benchmark and save the results
aft_bmr <- run_benchmark(design)

# set the measure
time = train_data_d1$time[train_data_d1$status == 1]
quantile = quantile(time, probs = 0.5)

# set the global evaluation metric
all_measures <- msr("surv.cindex")

aft_results <- aft_bmr$aggregate(measures = all_measures)
aft_results$surv.harrell_c ## 0.839

## Step_2_2: model tuning with the "baselearner" and "family"

# load the learner with aft
lrn_mb <- lrn("surv.mboost", baselearner = 'bols', family = "coxph")
# train with "baselearner"
base_learner_type <- c("bols", "btree")
family_type <- c("coxph", "cindex")
# set the search space
param_mb_bl <- ParamSet$new(params = list(
  ParamFct$new("baselearner", levels = base_learner_type),
  ParamFct$new("family", levels = family_type)
))

# inner resampling set
inner_rsmp <- rsmp("cv", folds = 4L)

# create the AutoTuner
mb_bl <- AutoTuner$new(
  learner = lrn_mb, resampling =  inner_rsmp,
  measure = msr("surv.cindex"), search_space = param_mb_bl,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 5)
)

# set the outer resampling
outer_rsmp <- rsmp("cv", folds = 3L)
# design the benchmark with bf
design_mb_bl <- benchmark_grid(
  tasks = tsks_train_d1,
  learners = mb_bl,
  resamplings = outer_rsmp
)

bmr_mb_bl <- run_benchmark(design_mb_bl)

# aggregate to get results of model when tuning with bf
bmr_mb_bl_results <- bmr_mb_bl$aggregate(measures = msr("surv.cindex"))


# plot the corresponding performances
mb_bl_path1 <- bmr_mb_bl$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
mb_bl_ggp1 <- ggplot(mb_bl_path1, aes(
  x = baselearner,
  y = surv.harrell_c, col = factor(family),
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()

mb_bl_path2 <- bmr_mb_bl$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
mb_bl_ggp2 <- ggplot(mb_bl_path2, aes(
  x = baselearner,
  y = surv.harrell_c, col = factor(family),
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()

mb_bl_path3 <- bmr_mb_bl$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
mb_bl_ggp3 <- ggplot(mb_bl_path3, aes(
  x = baselearner,
  y = surv.harrell_c, col = factor(family),
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()

# arrange all the plots together
ggarrange(mb_bl_ggp1, mb_bl_ggp2, mb_bl_ggp3, common.legend = TRUE, legend = "bottom")


# conclusion: go with "coxph", and "bols".


## Step_2_3_1: model tuning with the "mstop", early stopping, 
# from 10 to 300(default), 100-300, 150-300, 150-200, 155-185
# 160-180

# load the learner with aft
lrn_aft <- lrn("surv.mboost", baselearner = 'bols', family = "coxph")


# set the search space
param_aft_mstop <- ParamSet$new(params = list(
  ParamInt$new("mstop", lower = 160, upper = 180)
))


# create the AutoTuner
aft_mstop <- AutoTuner$new(
  learner = lrn_aft, resampling =  inner_rsmp,
  measure = msr("surv.cindex"), search_space = param_aft_mstop,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 10)
)

# design the benchmark with bf
design_aft_mstop <- benchmark_grid(
  tasks = tsks_train_d1,
  learners = aft_mstop,
  resamplings = outer_rsmp
)

bmr_aft_mstop <- run_benchmark(design_aft_mstop)

# aggregate to get results of model when tuning with bf
bmr_aft_mstop_results <- bmr_aft_mstop$aggregate(measures = msr("surv.cindex"))
# plot the corresponding performances
aft_mstop_path1 <- bmr_aft_mstop$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
aft_mstop_ggp1 <- ggplot(aft_mstop_path1, aes(
  x = mstop,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


aft_mstop_path2 <- bmr_aft_mstop$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
aft_mstop_ggp2 <- ggplot(aft_mstop_path2, aes(
  x = mstop,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


aft_mstop_path3 <- bmr_aft_mstop$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
aft_mstop_ggp3 <- ggplot(aft_mstop_path3, aes(
  x = mstop,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


# arrange all the plots together
ggarrange(aft_mstop_ggp1, aft_mstop_ggp2, aft_mstop_ggp3, common.legend = TRUE, legend = "bottom")

# conclusion: mstop = 164

## Step_1_5_1: model tuning with the "nu", 
# from 0.1 to 1, 0.1-0.4, 0.15-0.3

# load the learner with aft
lrn_aft <- lrn("surv.mboost", baselearner = 'bols', family = "coxph", mstop = 164)

# set the search space
param_aft_nu <- ParamSet$new(params = list(
  ParamDbl$new("nu", lower = 0.16, upper = 0.3)
))

# create the AutoTuner
aft_nu <- AutoTuner$new(
  learner = lrn_aft, resampling = inner_rsmp,
  measure = msr("surv.cindex"), search_space = param_aft_nu,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 15)
)

# design the benchmark with bf
design_aft_nu <- benchmark_grid(
  tasks = tsks_train_d1,
  learners = aft_nu,
  resamplings = outer_rsmp
)

bmr_aft_nu <- run_benchmark(design_aft_nu)

# aggregate to get results of model when tuning with bf
bmr_aft_nu_results <- bmr_aft_nu$aggregate(measures = msr("surv.cindex"))
# plot the corresponding performances
aft_nu_path1 <- bmr_aft_nu$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
aft_nu_ggp1 <- ggplot(aft_nu_path1, aes(
  x = nu,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


aft_nu_path2 <- bmr_aft_nu$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
aft_nu_ggp2 <- ggplot(aft_nu_path2, aes(
  x = nu,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


aft_nu_path3 <- bmr_aft_nu$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
aft_nu_ggp3 <- ggplot(aft_nu_path3, aes(
  x = nu,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()




# arrange all the plots together
ggarrange(aft_nu_ggp1, aft_nu_ggp2, aft_nu_ggp3, common.legend = TRUE, legend = "bottom")

# conclusion nu = 0.25

## final learner with aft model
lrn_aft <- lrn("surv.mboost", baselearner = 'bols', family = "coxph", mstop = 164, nu = 0.25)

design_aft <- benchmark_grid(
  tasks = tsks_train_d1,
  learners = lrn_aft,
  resampling = rsmp("cv", folds = 3L)
)


# define function to start benchmark with fixed seed
run_benchmark <- function(design_aft){
  set.seed(2021)
  bmr <- benchmark(design_aft, store_models = TRUE)
  run_benchmark <- bmr
}

## run benchmark and save the results
aft_bmr <- run_benchmark(design_aft)

# set the global evaluation metric
all_measures <- msr("surv.cindex")

aft_results <- aft_bmr$aggregate(measures = all_measures)
aft_results$surv.harrell_c # 0.8321


## ---- Step 3: Boosting with CoxBoost
## load the learner for the CoxBoost boosting and design the benchmark
# read in the raw data
train_data_original <- readRDS("Data-20210107/train_data.Rds")
test_data <- readRDS("Data-20210107/test_list_x.Rds")

## get the corresponding dataframe1
train_data_d1 <- train_data_original$d1
# only V4 needs to be one-hot-encoded

## Step 3_1: change the factor type to one-hot encoded
## Data preparation using one hot encoder
library("dataPreparation")
# Compute encoding
train_onehot_d1 <- train_data_d1
encoding <- build_encoding(train_onehot_d1, cols = c("V4"), verbose = TRUE)
# Apply one hot encoding
train_onehot_d1 <- one_hot_encoder(train_onehot_d1, encoding = encoding, drop = TRUE)
str(train_onehot_d1)

## create the corresponding task as for dataframe 1 after one-hot-encoding
tsks_oh_d1 <- TaskSurv$new("df1", backend = train_onehot_d1, time = "time", event = "status")
tsks_oh_d1


## Step 3_1: Try with a default setting
design <- benchmark_grid(
  tasks = tsks_oh_d1,
  learners = lrn("surv.coxboost", stepno=100, penalty=100, criterion="hpscore"),
  resampling = rsmp("cv", folds = 3L)
)

# lrn("surv.coxboost")$param_set

# define function to start benchmark with fixed seed
run_benchmark <- function(design){
  set.seed(2021)
  bmr <- benchmark(design, store_models = TRUE)
  run_benchmark <- bmr
}

## run benchmark and save the results
coxboost_bmr <- run_benchmark(design)


# set the global evaluation metric
all_measures <- msr("surv.cindex")

coxboost_results <- coxboost_bmr$aggregate(measures = all_measures)
coxboost_results # 0.827
# plot the corresponding the performance
autoplot(coxboost_bmr)

## Step_3_2: model tuning with the "stepno"

# load the learner with CoxBoost
lrn_cb <- lrn("surv.coxboost", criterion="hpscore")
# from 50-250, 100-300, 150-250

# set the search space
param_cb_stepno <- ParamSet$new(params = list(
  ParamInt$new("stepno", lower = 150, upper = 250)
))

# inner resampling set
inner_rsmp <- rsmp("cv", folds = 4L)

# create the AutoTuner
cb_stepno <- AutoTuner$new(
  learner = lrn_cb, resampling =  inner_rsmp,
  measure = msr("surv.cindex"), search_space = param_cb_stepno,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 11)
)
# set the outer resampling
outer_rsmp <- rsmp("cv", folds = 3L)
# design the benchmark with stepno
design_cb_stepno <- benchmark_grid(
  tasks = tsks_oh_d1,
  learners = cb_stepno,
  resamplings = outer_rsmp
)

bmr_cb_stepno <- run_benchmark(design_cb_stepno)

# aggregate to get results of model when tuning with bf
bmr_cb_stepno_results <- bmr_cb_stepno$aggregate(measures = msr("surv.cindex"))
# load install.packages("ggrepel")


# plot the corresponding performances
cb_stepno_path1 <- bmr_cb_stepno$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
cb_stepno_ggp1 <- ggplot(cb_stepno_path1, aes(
  x = stepno,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()

cb_stepno_path2 <- bmr_cb_stepno$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
cb_stepno_ggp2 <- ggplot(cb_stepno_path2, aes(
  x = stepno,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()

cb_stepno_path3 <- bmr_cb_stepno$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
cb_stepno_ggp3 <- ggplot(cb_stepno_path3, aes(
  x = stepno,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


# arrange all the plots together
ggarrange(cb_stepno_ggp1, cb_stepno_ggp2, cb_stepno_ggp3, common.legend = TRUE, legend = "bottom")

# conclusion: set stepno to be 150

## ## Step_3_3: model tuning with the "penalty"

# load the learner with CoxBoost
lrn_cb <- lrn("surv.coxboost", criterion = "hpscore", stepno = 150)
# from 100-500, 450-500, 450-490, 450-480

# set the search space
param_cb_penal <- ParamSet$new(params = list(
  ParamDbl$new("penalty", lower = 450, upper = 480)
))

# inner resampling set
inner_rsmp <- rsmp("cv", folds = 4L)

# create the AutoTuner
cb_penal <- AutoTuner$new(
  learner = lrn_cb, resampling =  inner_rsmp,
  measure = msr("surv.cindex"), search_space = param_cb_penal,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 7)
)
# set the outer resampling
outer_rsmp <- rsmp("cv", folds = 3L)
# design the benchmark with stepno
design_cb_penal <- benchmark_grid(
  tasks = tsks_oh_d1,
  learners = cb_penal,
  resamplings = outer_rsmp
)

bmr_cb_penal <- run_benchmark(design_cb_penal)

# aggregate to get results of model when tuning with bf
bmr_cb_penal_results <- bmr_cb_penal$aggregate(measures = msr("surv.cindex"))
# load install.packages("ggrepel")


# plot the corresponding performances
cb_penal_path1 <- bmr_cb_penal$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
cb_penal_ggp1 <- ggplot(cb_penal_path1, aes(
  x = penalty,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()

cb_penal_path2 <- bmr_cb_penal$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
cb_penal_ggp2 <- ggplot(cb_penal_path2, aes(
  x = penalty,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()

cb_penal_path3 <- bmr_cb_penal$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
cb_penal_ggp3 <- ggplot(cb_penal_path3, aes(
  x = penalty,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


# arrange all the plots together
ggarrange(cb_penal_ggp1, cb_penal_ggp2, cb_penal_ggp3, common.legend = TRUE, legend = "bottom")
# conclusion: go with penalty = 470

## ## Step_3_4: model tuning with the "stepsize.factor"

# load the learner with CoxBoost
lrn_cb <- lrn("surv.coxboost", criterion = "hpscore", stepno = 150, penalty = 470)
# from 0.1-1, 1-10

# set the search space
param_cb_sf <- ParamSet$new(params = list(
  ParamDbl$new("stepsize.factor", lower = 0.1, upper = 1)
))

# inner resampling set
inner_rsmp <- rsmp("cv", folds = 4L)

# create the AutoTuner
cb_sf <- AutoTuner$new(
  learner = lrn_cb, resampling =  inner_rsmp,
  measure = msr("surv.cindex"), search_space = param_cb_sf,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 10)
)
# set the outer resampling
outer_rsmp <- rsmp("cv", folds = 3L)
# design the benchmark with stepno
design_cb_sf <- benchmark_grid(
  tasks = tsks_oh_d1,
  learners = cb_sf,
  resamplings = outer_rsmp
)

bmr_cb_sf <- run_benchmark(design_cb_sf)

# aggregate to get results of model when tuning with bf
bmr_cb_sf_results <- bmr_cb_sf$aggregate(measures = msr("surv.cindex"))
# load install.packages("ggrepel")


# plot the corresponding performances
cb_sf_path1 <- bmr_cb_sf$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
cb_sf_ggp1 <- ggplot(cb_sf_path1, aes(
  x = stepsize.factor,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()

cb_sf_path2 <- bmr_cb_sf$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
cb_sf_ggp2 <- ggplot(cb_sf_path2, aes(
  x = stepsize.factor,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()

cb_sf_path3 <- bmr_cb_sf$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
cb_sf_ggp3 <- ggplot(cb_sf_path3, aes(
  x = stepsize.factor,
  y = surv.harrell_c, #, col = factor(family) 
  label = round(surv.harrell_c, 4)
)) +
  geom_point(size = 3, color = "red") + 
  geom_line() +
  geom_text_repel()


# arrange all the plots together
ggarrange(cb_sf_ggp1, cb_sf_ggp2, cb_sf_ggp3, common.legend = TRUE, legend = "bottom")
# conclusion: go with sf=1

## final learner with CoxBoost model
lrn_cb <- lrn("surv.coxboost", criterion = "hpscore", stepno = 150, penalty = 470, stepsize.factor = 1)

design_cb <- benchmark_grid(
  tasks = tsks_oh_d1,
  learners = lrn_cb,
  resampling = rsmp("cv", folds = 3L)
)


# define function to start benchmark with fixed seed
run_benchmark <- function(design_cb){
  set.seed(2021)
  bmr <- benchmark(design_cb, store_models = TRUE)
  run_benchmark <- bmr
}

## run benchmark and save the results
cb_bmr <- run_benchmark(design_cb)

# set the global evaluation metric
all_measures <- msr("surv.cindex")

cb_results <- cb_bmr$aggregate(measures = all_measures)
cb_results$surv.harrell_c # 0.839

