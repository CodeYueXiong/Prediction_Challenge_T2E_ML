##################################################
##################################################
##########       Master Thesis        ############
##################################################
##################################################


#==========================================================================================

# install the necessary packages
library(readr)
library(dplyr)
library(mlr3)
library(tidyverse)
library(ggplot2)
library(mlr3learners)
library(data.table)
library(mlr3viz)
library(mlr3tuning)
library(mlr3pipelines)
library(paradox)
library(skimr)
library(smotefamily)
library(purrr)
library(mice)
library(missForest)
library(Amelia)
library(caret)
library(mltools)
library(scorecard)
library(xlsx)
library(checkmate)
library(dataPreparation)
library(gridExtra)
library(precrec)
library(xgboost)


# packages <- installed.packages()
# lapply(packages[1:200], require, character.only = TRUE)

# Set or change R working directory
setwd("/Users/apple/Desktop/Master_Thesis/Data")

wd = dirname(rstudioapi::getActiveDocumentContext()$path)

##############################################################
##########       Step 1: Data preparation        #############
##############################################################

# Read in raw data
application_data <- read_csv("application_record.csv") 
record <- read_csv("credit_record.csv")


#------------------------------------------------------------------------------------------
# Group all data from the credit record by "ID"
# then extract the minimum snapshot date indictating when an account is opened
opentime_data <- record %>%
  group_by(ID) %>%
  filter(MONTHS_BALANCE == min(MONTHS_BALANCE)) %>% select(ID, MONTHS_BALANCE) %>%
  rename(opentime = MONTHS_BALANCE)
# merge two dataset based on the common variable "ID"
data_with_day <- left_join(application_data, opentime_data, by = "ID")

# Check the overdue status
unique(record$STATUS)

# Create a function to define the target variable 
create_target <- function(x) {
  if (x == "2" | x == "3" | x == "4" | x == "5") {
    return(TRUE) } else {
      return(FALSE) }
}

# "TRUE" indicates that the applicant has a record of more than 60 days overdue
new_record <- record %>% mutate(target = map_dbl(record$STATUS, create_target))

# Sum the value of variable "target" for each applicant (grouped by ID).
data_target <- new_record %>% group_by(ID) %>% summarise(y = sum(target))

# for each applicant, if the target value > 0,
# means there is at least one TRUE under the ID number,
# if an ID has one TURE, means this person has been overdue at least 60 days.
# Consequently, mark this ID as 1, which indicates that credit card application will not be approved. 


# Credit card application will not be approved for each applicant who is more than 60 days overdue
data_target$y <- map_dbl(data_target$y, function(x) ifelse(x > 0, 1, 0))

# Check the distribution of the target variable
round(prop.table(table(data_target$y)), digits = 3)
#    0     1 
#  0.985 0.015 
# Only 1.5% of applications will be rejected. Extremely imbalanced data


# merge two data with method inner_join.
data <- inner_join(data_with_day, data_target, by = "ID")
# convert all character data into facotr datatype.
final_data <- data %>% mutate_if(is.character, as.factor) %>% mutate(y = as.factor(y))
# check the final data
str(final_data)

# remove irrelevant variable "ID" and "FLAG_MOBIL"
final_data <- final_data %>% select(-ID, -FLAG_MOBIL)
# print dimension of the data (number of entries, number of variables)
dim(final_data)

# convert to the data frame
to_imp_data <- final_data %>% as.data.frame()
# list all variables (column name)
names(final_data)


#-------------------------------------------------------------------------------------

length(data_target$y)

df <- data.frame(Target_Variable =c("Approval", "Not Approval"),
                 len=c(0.985, 0.015))

p<-ggplot(data=df, aes(x=Target_Variable, y=len)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_bw()+
  geom_text(aes(label=len), vjust=-0.3, size=5)+
  xlab("Target Variable") + ylab("Percentage (%)")+
  ggtitle("Approval Rate")
p



#######################################################################
##########       Step 2: Missing value processing         #############
#######################################################################


# check the missing pattern by using md.pattern function
mice::md.pattern(to_imp_data, plot = FALSE)

#------------------------------------------------------------------------------------------
# Step 2.1: Listwise Deletion
# remove rows with missing values
completeData_delet <- na.omit(final_data)
# check the dataset after deleting
str(completeData_delet)

# Export data from R
# write_csv2(completeData_delet, "./dl_na_data_3.csv")
 
#------------------------------------------------------------------------------------------
# Step 2.2: Multiple imputation with MICE package
# imputed data with mice function
tempData_mice <- mice(to_imp_data, m = 1, meth = "polyreg", seed = 2020)
#to extract the imputed data with the assigned column name "OCCUPATION_TYPE"
summary(tempData_mice$imp$OCCUPATION_TYPE)
# Use function complete() to return the complete dataset
complete_mice <- complete(tempData_mice)

# Export data from R
# write_csv2(complete_mice, "./mice_na_data.csv")

#------------------------------------------------------------------------------------------
# Step 2.3: Multiple imputation with missForest package
# Missing value imputation with missForest
imputed_Data <- missForest(to_imp_data)
# save data and check imputation with OCCUPATION_TYPE
completeData_missForest <- imputed_Data$ximp

# Export data from R
# write_csv2(completeData_missForest, "./mf_na_data.csv")












##############################################################
##########       Step 3: Variable encoding        ############
##############################################################


dl_data <- read.csv2("./dl_na_data.csv") 
binary_data <- dl_data %>% select_if(is.factor) 
glimpse(binary_data)


# Function 3.1: "convert_binary_variable" is used to convert the binary variables
convert_binary_variable <- function(feature) {
  # save condition for succinct expression
  # check for defensive programm
  if (check_class(feature, "factor") & nlevels(feature) == 2) {
    levels(feature) <- seq(nlevels(feature))
    feature <- as.numeric(feature)
    return(feature)
  }
}


## Function 3.2: "calc_iv" is used to compute the "information value (iv)" of a variable.
calc_iv <- function(feature) {
  # how many rows we need.
  number_row <- length(unique(final_data[[feature]]))
  # initialize the tibble
  inner_iv_table <- tibble(feature = 0, val = 0, all = 0, good = 0, bad = 0)
  # for each level, we calculate their information value
  for (i in seq(number_row)) {
    # mark which level is computed
    val <- unique(final_data[[feature]])[[i]]
    # conver data as "data.table" for the convenience of select.
    final_data <- data.table(final_data)
    # compute how many simples within this level
    all <- nrow(final_data[final_data[[feature]] == val])
    # the number of good people within this level
    good <- nrow(final_data[final_data[[feature]] == val & final_data$y == 0])
    # the number of bad people within this level
    bad <- nrow(final_data[final_data[[feature]] == val & final_data$y == 1])
    # store as a tibble
    inner_tibble <- tibble(feature, val, all, good, bad)
    # rbind "inner_tibble" with "inner_iv_table
    inner_iv_table <- rbind(inner_iv_table, inner_tibble)
  }
  # delet our initial rows.
  inner_iv_table <- inner_iv_table[-1, ]
  
  # compute IV
  inner_iv_table <- inner_iv_table %>%
    # compute the proportion for each level,
    mutate(share = all / sum(all)) %>%
    # compute bad rate
    mutate(bad_rate = bad / all) %>%
    # compute good distribution
    mutate(good_dis = (all - bad) / (sum(all) - sum(bad))) %>%
    # compute bad distribution
    mutate(bad_dis = bad / sum(bad)) %>%
    # compute woe
    mutate(woe = log(good_dis / bad_dis))
  # deal with the extreme situation
  inner_iv_table$woe[is.infinite(inner_iv_table$woe)] <- 0
  # compute IV
  iv_table <- inner_iv_table %>% mutate(iv = woe * (good_dis - bad_dis))
  # return value of IV
  print(sum(iv_table$iv))
  # return "iv_table"
  return(iv_table)
}


#------------------------------------------------------------------------------------------
## Step 3.1 WOE Encoding & Listwise Deletion

##change final_data
dl_data <- read.csv2("./dl_na_data.csv")
final_data <- dl_data
# convert all binary variable into binary number
converted_data <- final_data %>% mutate_if(is.factor, convert_binary_variable)

# find which variables have several values
dif_variable <- setdiff(names(final_data), names(converted_data))

# analyse the details of first multi-factors variable
# found the variable "NAME_INCOME_TYPE" has 5 types of values
final_data["less_factor_income"] <- final_data %>%
  pull(dif_variable[[1]])

# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[1])
calc_iv("less_factor_income")

# integrate "student" , "pensioner" with "state servant" seems not good.
# try integrate "student" "pensioner" with "Commercial associate"
final_data["less_factor_income"] <- final_data %>%
  pull(dif_variable[[1]]) %>%
  recode("Student" = "Commercial associate", "Pensioner" = "Commercial associate")


# Iv increases from 0.010 to 0.013, seems better.
calc_iv(dif_variable[1])
calc_iv("less_factor_income")

#analyse the details of second multi-factors variable
calc_iv(dif_variable[2])
# found the variable "NAME_EDUCSTION_TYPE" has 5 types of values. we integrate
# "Incomplete higher" with "Higher education"
final_data["less_factor_edu"] <- final_data %>%
  pull(dif_variable[[2]]) %>%
  recode("Higher education" = "Incomplete higher")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[2])
calc_iv("less_factor_edu")
# analyse the details of third multi-factors variable
calc_iv(dif_variable[3])
final_data["less_factor_status"] <- final_data %>% pull(dif_variable[3])
# found this variable seems relatively balanced, no need to change.


# analyse the details of fourth multi-factors variable
calc_iv(dif_variable[4])
# found the variable "NAME_HOUSING_TYPE" has 6 types of values. we integrate
# "Co-op apartment" with "Office apartment"
final_data["less_factor_house"] <- final_data %>%
  pull(dif_variable[[4]]) %>%
  recode("Co-op apartment" = "Office apartment")
# analyse how the IV of the variable has been changed.
calc_iv("less_factor_house")

# analyse the details of fifth multi-factors variable
calc_iv(dif_variable[5])
# found the variable "NAME_HOUSING_TYPE" has too manny types of values. we integrate

final_data["less_factor_work"] <- final_data %>%
  pull(dif_variable[[5]]) %>%
  recode("Cleaning staff" = "Labor",
         "Cooking staff" = "Labor",
         "Drivers" = "Labor",
         "Laborers" = "Labor",
         "Low-skill Laborers" = "Labor",
         "Security staff" = "Labor",
         "Waiters/barmen staff" = "Labor") %>% 
  recode("Accountants" = "Office",
         "Core staff" = "Office",
         "HR staff" = "Office",
         "Medicine staff" = "Office",
         "Private service staff" = "Office",
         "Realty agents" = "Office",
         "Sales staff" = "Office",
         "Secretaries" = "Office") %>% 
  recode("Managers" = "higher",
         "High skill tech staff" = "higher",
         "IT staff" = "higher")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[5])
calc_iv("less_factor_work")


########
#encode multi-factors variables into one-hot 
factor_data <- final_data %>% select(starts_with("less")) %>% as.data.table()
oh_data <- one_hot(factor_data,dropCols = T)
#cbind oh_data with converted_data
converted_data <- final_data %>% mutate_if(is.factor, convert_binary_variable)
converted_data <- converted_data %>% add_column(oh_data)


#Binning continuous variable to improve performance.
dl_bin <- woebin(converted_data,y= "y",
                 x=c("CNT_CHILDREN","AMT_INCOME_TOTAL","DAYS_BIRTH","DAYS_EMPLOYED","CNT_FAM_MEMBERS"), 
                 method = "chimerge")
#plot binning variable
plotlist = woebin_plot(dl_bin)
#cbind dl_bin with dl_iv_data
dl_iv_data = woebin_ply(converted_data, rbindlist(dl_bin))
#save dl_iv_data
dl_iv_data$y <- as.factor(dl_iv_data$y)
colnames(dl_iv_data) <- make.names(colnames(dl_iv_data),unique=T)

# Export data from R
# write_csv2(dl_iv_data,"./dl_iv_data.csv")


#------------------------------------------------------------------------------------------
## Step 3.2: WOE Encoding & MICE 

#change final_data
mice_data <- read.csv2("./mice_na_data.csv")
final_data <- mice_data
# convert all binary variable into binary number
converted_data <- final_data %>% mutate_if(is.factor, convert_binary_variable)

# find which variables have several values
dif_variable <- setdiff(names(final_data), names(converted_data))

# analyse the details of first multi-factors variable
# found the variable "NAME_INCOME_TYPE" has 5 types of values
final_data["less_factor_income"] <- final_data %>%
  pull(dif_variable[[1]])

# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[1])
calc_iv("less_factor_income")

# integrate "student" , "pensioner" with "state servant" seems not good.
# try integrate "student" "pensioner" with "Commercial associate"
final_data["less_factor_income"] <- final_data %>%
  pull(dif_variable[[1]]) %>%
  recode("Student" = "Commercial associate", "Pensioner" = "Commercial associate")


# Iv increases from 0.01 to 0.013, seems better.
calc_iv(dif_variable[1])
calc_iv("less_factor_income")

#analyse the details of second multi-factors variable
calc_iv(dif_variable[2])
# found the variable "NAME_EDUCSTION_TYPE" has 5 types of values. we integrate
# "Incomplete higher" with "Higher education"
final_data["less_factor_edu"] <- final_data %>%
  pull(dif_variable[[2]]) %>%
  recode("Higher education" = "Incomplete higher")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[2])
calc_iv("less_factor_edu")
# analyse the details of third multi-factors variable
calc_iv(dif_variable[3])
final_data["less_factor_status"] <- final_data %>% pull(dif_variable[3])
# found this variable seems relatively balanced, no need to change.


# analyse the details of fourth multi-factors variable
calc_iv(dif_variable[4])
# found the variable "NAME_HOUSING_TYPE" has 6 types of values. we integrate
# "Co-op apartment" with "Office apartment"
final_data["less_factor_house"] <- final_data %>%
  pull(dif_variable[[4]]) %>%
  recode("Co-op apartment" = "Office apartment")
# analyse how the IV of the variable has been changed.
calc_iv("less_factor_house")

# analyse the details of fifth multi-factors variable
calc_iv(dif_variable[5])
# found the variable "NAME_HOUSING_TYPE" has too manny types of values. we integrate

final_data["less_factor_work"] <- final_data %>%
  pull(dif_variable[[5]]) %>%
  recode("Cleaning staff" = "Labor",
         "Cooking staff" = "Labor",
         "Drivers" = "Labor",
         "Laborers" = "Labor",
         "Low-skill Laborers" = "Labor",
         "Security staff" = "Labor",
         "Waiters/barmen staff" = "Labor") %>% 
  recode("Accountants" = "Office",
         "Core staff" = "Office",
         "HR staff" = "Office",
         "Medicine staff" = "Office",
         "Private service staff" = "Office",
         "Realty agents" = "Office",
         "Sales staff" = "Office",
         "Secretaries" = "Office") %>% 
  recode("Managers" = "higher",
         "High skill tech staff" = "higher",
         "IT staff" = "higher")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[5])
calc_iv("less_factor_work")
#########
#encode multi-factors variables into one-hot 
factor_data <- final_data %>% select(starts_with("less")) %>% as.data.table()
oh_data <- one_hot(factor_data,dropCols = T)
#cbind oh_data with converted_data
converted_data <- final_data %>% mutate_if(is.factor, convert_binary_variable)
converted_data <- converted_data %>% add_column(oh_data)
#Binning continuous variable to improve performance.
dl_bin <- woebin(converted_data,y= "y",
                 x=c("CNT_CHILDREN","AMT_INCOME_TOTAL","DAYS_BIRTH","DAYS_EMPLOYED","CNT_FAM_MEMBERS"), 
                 method = "chimerge")
#plot binning variable
plotlist = woebin_plot(dl_bin)
#cbind dl_bin with dl_iv_data
mice_iv_data = woebin_ply(converted_data, rbindlist(dl_bin))
#save mice_iv_data
mice_iv_data$y <- as.factor(mice_iv_data$y)
colnames(mice_iv_data) <- make.names(colnames(mice_iv_data),unique=T) 

# Export data from R
# write_csv2(mice_iv_data,"./mice_iv_data.csv")


#------------------------------------------------------------------------------------------
## Step 3.3: WOE Encoding & missForest

# read in mf_na_data
mf_data <- read.csv2("./mf_na_data.csv")
final_data <- mf_data


# convert all binary variable into binary number
converted_data <- final_data %>% mutate_if(is.factor, convert_binary_variable)

# find which variables have several values
dif_variable <- setdiff(names(final_data), names(converted_data))

# analyse the details of first multi-factors variable
# found the variable "NAME_INCOME_TYPE" has 5 types of values
final_data["less_factor_income"] <- final_data %>%
  pull(dif_variable[[1]])

# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[1])
calc_iv("less_factor_income")

# integrate "student" , "pensioner" with "state servant" seems not good.
# try integrate "student" "pensioner" with "Commercial associate"
final_data["less_factor_income"] <- final_data %>%
  pull(dif_variable[[1]]) %>%
  recode("Student" = "Commercial associate", "Pensioner" = "Commercial associate")


# Iv increases from 0.01 to 0.013, seems better.
calc_iv(dif_variable[1])
calc_iv("less_factor_income")

#analyse the details of second multi-factors variable
calc_iv(dif_variable[2])
# found the variable "NAME_EDUCSTION_TYPE" has 5 types of values. we integrate
# "Incomplete higher" with "Higher education"
final_data["less_factor_edu"] <- final_data %>%
  pull(dif_variable[[2]]) %>%
  recode("Higher education" = "Incomplete higher")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[2])
calc_iv("less_factor_edu")
# analyse the details of third multi-factors variable
calc_iv(dif_variable[3])
final_data["less_factor_status"] <- final_data %>% pull(dif_variable[3])
# found this variable seems relatively balanced, no need to change.


# analyse the details of fourth multi-factors variable
calc_iv(dif_variable[4])
# found the variable "NAME_HOUSING_TYPE" has 6 types of values. we integrate
# "Co-op apartment" with "Office apartment"
final_data["less_factor_house"] <- final_data %>%
  pull(dif_variable[[4]]) %>%
  recode("Co-op apartment" = "Office apartment")
# analyse how the IV of the variable has been changed.
calc_iv("less_factor_house")

# analyse the details of fifth multi-factors variable
calc_iv(dif_variable[5])
# found the variable "NAME_HOUSING_TYPE" has too manny types of values. we integrate

final_data["less_factor_work"] <- final_data %>%
  pull(dif_variable[[5]]) %>%
  recode("Cleaning staff" = "Labor",
         "Cooking staff" = "Labor",
         "Drivers" = "Labor",
         "Laborers" = "Labor",
         "Low-skill Laborers" = "Labor",
         "Security staff" = "Labor",
         "Waiters/barmen staff" = "Labor") %>% 
  recode("Accountants" = "Office",
         "Core staff" = "Office",
         "HR staff" = "Office",
         "Medicine staff" = "Office",
         "Private service staff" = "Office",
         "Realty agents" = "Office",
         "Sales staff" = "Office",
         "Secretaries" = "Office") %>% 
  recode("Managers" = "higher",
         "High skill tech staff" = "higher",
         "IT staff" = "higher")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[5])
calc_iv("less_factor_work")
#########
# encode multi-factors variables into one-hot 
factor_data <- final_data %>% select(starts_with("less")) %>% as.data.table()
oh_data <- one_hot(factor_data,dropCols = T)
# bind oh_data with converted_data
converted_data <- final_data %>% mutate_if(is.factor, convert_binary_variable)
converted_data <- converted_data %>% add_column(oh_data)
# binning continuous variable to improve performance.
dl_bin <- woebin(converted_data,y= "y",
                 x=c("CNT_CHILDREN","AMT_INCOME_TOTAL","DAYS_BIRTH","DAYS_EMPLOYED","CNT_FAM_MEMBERS"), 
                 method = "chimerge")
# plot binning variable
plotlist = woebin_plot(dl_bin)
# cbind dl_bin with dl_iv_data
mf_iv_data = woebin_ply(converted_data, rbindlist(dl_bin))
# save mf_iv_data
mf_iv_data$y <- as.factor(mf_iv_data$y)
colnames(mf_iv_data) <- make.names(colnames(mf_iv_data),unique=T)

# Export data from R
# write_csv2(mf_iv_data,"./mf_iv_data.csv")


#------------------------------------------------------------------------------------------
## Step 3.4: One-hot Encoding & Listwise Deletion

# read in dl_na_data
data_onehot <- read.csv2("./dl_na_data.csv")

# compute encoding
encoding <- build_encoding(data_onehot, cols = c("CODE_GENDER", "FLAG_OWN_CAR","FLAG_OWN_REALTY",
                                                 "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS",
                                                 "NAME_HOUSING_TYPE","OCCUPATION_TYPE"), verbose = TRUE)

# apply one hot encoding
dl_oh_data <- one_hot_encoder(data_onehot, encoding = encoding, drop = TRUE)

# Export data from R
# write_csv2(dl_oh_data,"./dl_oh_data.csv")

#------------------------------------------------------------------------------------------
## Step 3.5: One-hot Encoding & MICE

# read in mice_na_data
data_onehot <- read.csv2("./mice_na_data.csv")

# compute encoding
encoding <- build_encoding(data_onehot, cols = c("CODE_GENDER", "FLAG_OWN_CAR","FLAG_OWN_REALTY",
                                                 "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS",
                                                 "NAME_HOUSING_TYPE","OCCUPATION_TYPE"), verbose = TRUE)

# apply one hot encoding
mice_oh_data <- one_hot_encoder(data_onehot, encoding = encoding, drop = TRUE)

# Export data from R
# write_csv2(mice_oh_data,"./mice_oh_data.csv")

#------------------------------------------------------------------------------------------
## Step 3.6: One-hot Encoding & missForest

# read in mf_na_data
data_onehot <- read.csv2("./mf_na_data.csv")

# compute encoding
encoding <- build_encoding(data_onehot, cols = c("CODE_GENDER", "FLAG_OWN_CAR","FLAG_OWN_REALTY",
                                                 "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS",
                                                 "NAME_HOUSING_TYPE","OCCUPATION_TYPE"), verbose = TRUE)

# apply one hot encoding
mf_oh_data <- one_hot_encoder(data_onehot, encoding = encoding, drop = TRUE)

# Export data from R
# write_csv2(mf_oh_data,"./mf_oh_data.csv")















#################################################################
##########       Step 4: Data set selection         #############
#################################################################

setwd("/Users/apple/Desktop/Master_Thesis/Data")

# load all data sets
dl_iv_data <- read.csv2("./dl_iv_data.csv") %>% mutate_if(is.integer, as.numeric) %>%
  mutate(y = as.factor(y))
mice_iv_data <- read.csv2("./mice_iv_data.csv") %>% mutate_if(is.integer, as.numeric) %>%
  mutate(y = as.factor(y))
mf_iv_data <- read.csv2("./mf_iv_data.csv") %>% mutate_if(is.integer, as.numeric) %>%
  mutate(y = as.factor(y))
dl_oh_data <- read.csv("./dl_oh_data.csv") %>% mutate_if(is.integer, as.numeric) %>%
  mutate(y = as.factor(y))
mice_oh_data <- read.csv("./mice_oh_data.csv") %>% mutate_if(is.integer, as.numeric) %>%
  mutate(y = as.factor(y))
mf_oh_data <- read.csv("./mf_oh_data.csv") %>% mutate_if(is.integer, as.numeric) %>%
  mutate(y = as.factor(y))


library(mlr3)
# create tasks from all the data
task_all <- list(
  TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y"), 
  TaskClassif$new("mf_iv", backend = mf_iv_data, target = "y"), 
  TaskClassif$new("mice_iv", backend = mice_iv_data, target = "y"), 
  TaskClassif$new("dl_oh", backend = dl_oh_data, target = "y"), 
  TaskClassif$new("mf_oh", backend = mf_oh_data, target = "y"), 
  TaskClassif$new("mice_oh", backend = mice_oh_data, target = "y")
)

# plot for the 6 data tasks
multiplot_roc <- function(models, type = "roc") {
  # set a null list to colect all plots
  plots <- list()
  
  # filter model with task id, and the corresponding AUC value
  # then plot the ROC curve
  model <- models$clone(deep = TRUE)$filter(task_id = "dl_iv")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[1]] <- autoplot(model, type = type) + ggtitle(paste("dl_iv:", auc)) 
  
  model <- models$clone(deep = TRUE)$filter(task_id = "mf_iv")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[2]] <- autoplot(model, type = type) + ggtitle(paste("mf_iv:", auc)) 
  
  model <- models$clone(deep = TRUE)$filter(task_id = "mice_iv")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[3]] <- autoplot(model, type = type) + ggtitle(paste("mice_iv:", auc)) 
  
  model <- models$clone(deep = TRUE)$filter(task_id = "dl_oh")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[4]] <- autoplot(model, type = type) + ggtitle(paste("dl_oh:", auc)) 
  
  model <- models$clone(deep = TRUE)$filter(task_id = "mf_oh")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[5]] <- autoplot(model, type = type) + ggtitle(paste("mf_oh:", auc)) 
  
  model <- models$clone(deep = TRUE)$filter(task_id = "mice_oh")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[6]] <- autoplot(model, type = type) + ggtitle(paste("mice_oh:", auc)) 
  
  # merge all plots in the list
  do.call("grid.arrange", plots)
}


# set global measure
all_measures <- msr("classif.auc")
# define function to start benchmark with fixed seed
# save the result, and print training time
# tictoc is used to measure the duration of the training library(tictoc)

library(tictoc)
run_benchmark <- function(design){
  set.seed(2020)
  bmr <- benchmark(design, store_models = FALSE) 
  run_benchmark <- bmr
}


#=======================================================================================
## Step 4.1: Logistic Regression

# create a benchmark
design <- benchmark_grid(
  tasks = task_all,
  learners = lrn("classif.log_reg", predict_type = "prob"), 
  resamplings = rsmp("cv", folds = 5L)
)


# run benchmark and save the results
lg_bmr <- run_benchmark(design)
# lg_bmr <- benchmark(design, store_models = TRUE)
lg_results <- lg_bmr$aggregate(measures = all_measures) # 0.7228495

# plot roc curve of all tasks
lg_roc <- multiplot_roc(lg_bmr)


#=======================================================================================
## Step 4.2: KNN

# create a benchmark
design_knn <- benchmark_grid(
  tasks = task_all,
  learners = lrn("classif.kknn", predict_type = "prob"), 
  resampling = rsmp("cv", folds = 5L)
)
# run the benchmark, and save the results
knn_bmr <- run_benchmark(design_knn)
knn_results <- knn_bmr$aggregate(measures = all_measures)
# plot roc curve of all tasks
knn_roc <- multiplot_roc(knn_bmr)

#=======================================================================================
## Step 4.3: Random Forest

# create a benchmark to train a list of tasks with default hyperparameter settings
design_rf <- benchmark_grid(
  tasks = task_all,
  learners = lrn("classif.ranger", predict_type = "prob"), 
  resampling = rsmp("cv", folds = 5L)
)
# run the benchmark, and save the result
rf_bmr <- run_benchmark(design_rf)
rf_results <- rf_bmr$aggregate(all_measures)
# plot roc curve of all tasks
rf_roc <- multiplot_roc(rf_bmr)

#=======================================================================================
## Step 4.4: xgboost

# The runtime of SVM is longer than other methods

design_xgboost <- benchmark_grid(
  tasks = task_all,
  learners = lrn("classif.xgboost", predict_type = "prob"),
  resampling = rsmp("cv", folds = 5L)
)
# run the benchmark, and save the result
xgboost_bmr <- run_benchmark(design_xgboost)

xgboost_results <- xgboost_bmr$aggregate(all_measures)
# plot roc curve of all tasks
xgboost_roc <- multiplot_roc(xgboost_bmr)



#=======================================================================================
## Step 4.5: SVM

# The runtime of SVM is longer than other methods

design_svm <- benchmark_grid(
  tasks = task_all,
  learners = lrn("classif.svm", predict_type = "prob"), 
  resampling = rsmp("cv", folds = 5L)
)
# run the benchmark, and save the result
svm_bmr <- run_benchmark(design_svm)

svm_results <- svm_bmr$aggregate(all_measures)
# plot roc curve of all tasks
svm_roc <- multiplot_roc(svm_bmr)






##############################################################
##############################################################
##########        Step 5: Data balancing         #############
##############################################################
##############################################################

#==========================================================================================

####################################################################
##########        Step 5.1: Logistic Regression        #############
####################################################################

#------------------------------------------------------------------------------------------
#### Oversample

# load library
library(mlr3learners)
library(mlr3tuning)
library(mlr3pipelines)
library(paradox)

task <- TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y")

# logistic learner
lg_learner <- lrn("classif.log_reg", predict_type = "prob")

po_over <- po("classbalancing",
              id = "oversample", adjust = "minor",
              reference = "minor", shuffle = FALSE, ratio = 6
)

# create oversample
lg_over_learner <- GraphLearner$new(po_over %>>% lg_learner, predict_type = "prob")

# define parameter
lg_over_param_set <- ParamSet$new(list(ParamDbl$new("oversample.ratio", 
                                                    lower = 10, upper = 70)))

# set inner resampling
inner_rsmp <- rsmp("cv", folds = 10L)

# create autotuner
lg_over_auto <- AutoTuner$new(
  learner = lg_over_learner, resampling = inner_rsmp,
  measure = msr("classif.auc"), 
  search_space = lg_over_param_set,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 12)
)

# set outer_resampling, and create a design with it
outer_rsmp <- rsmp("cv", folds = 10L)
lg_over_design <- benchmark_grid(
  tasks = task,
  learners = lg_over_auto,
  resamplings = outer_rsmp
)

#-----

set.seed(2020)

# create benchmark
lg_over_bmr <- benchmark(lg_over_design, store_models = TRUE)

# aggregate to get results of model
# run the benchmark, and save the result
# lg_over_bmr <- run_benchmark(lg_over_design)

# aggregate to get results of model
lg_over_results <- lg_over_bmr$aggregate(measures = msr("classif.auc"))

# inspect auc value.
lg_over_results$classif.auc


# plot oversample ratio vs AUC
library(ggplot2)

over_path1 <- lg_over_bmr$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
over_gg1 <- ggplot(over_path1, aes(
  x = oversample.ratio,
  y = classif.auc
)) +
  geom_point(size = 3) +
  geom_line()

over_path2 <- lg_over_bmr$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
over_gg2 <- ggplot(over_path2, aes(
  x = oversample.ratio,
  y = classif.auc
)) +
  geom_point(size = 3) +
  geom_line()

over_path3 <- lg_over_bmr$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
over_gg3 <- ggplot(over_path3, aes(
  x = oversample.ratio,
  y = classif.auc
)) +
  geom_point(size = 3) +
  geom_line()



# combine all plots in one
library(ggpubr)
ggarrange(over_gg1, over_gg2, over_gg3, common.legend = TRUE, legend = "bottom")


#---------------------------------------------------------------------------------------------------
# Narrow search area

# define new parameter
lg_over_param_set <- ParamSet$new(list(ParamDbl$new("oversample.ratio", 
                                                    lower = 20, upper = 45)))

# set inner resampling
inner_rsmp <- rsmp("cv", folds = 10L)

# create autotuner
lg_over_auto <- AutoTuner$new(
  learner = lg_over_learner, resampling = inner_rsmp,
  measure = msr("classif.auc"), 
  search_space = lg_over_param_set,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 12)
)

# set outer_resampling, and create a design with it
outer_rsmp <- rsmp("cv", folds = 10L)
lg_over_design <- benchmark_grid(
  tasks = task,
  learners = lg_over_auto,
  resamplings = outer_rsmp
)

#-----

set.seed(2020)

# create benchmark
lg_over_bmr <- benchmark(lg_over_design, store_models = TRUE)

# aggregate to get results of model
# run the benchmark, and save the result
# lg_over_bmr <- run_benchmark(lg_over_design)

# aggregate to get results of model
lg_over_results <- lg_over_bmr$aggregate(measures = msr("classif.auc"))

# inspect auc value.
lg_over_results$classif.auc


# plot oversample ratio vs AUC
library(ggplot2)

over_path1 <- lg_over_bmr$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
over_path2 <- lg_over_bmr$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
over_path3 <- lg_over_bmr$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
over_path4 <- lg_over_bmr$data$learners()[[2]][[4]]$model$tuning_instance$archive$data()
over_path5 <- lg_over_bmr$data$learners()[[2]][[5]]$model$tuning_instance$archive$data()
over_path6 <- lg_over_bmr$data$learners()[[2]][[6]]$model$tuning_instance$archive$data()
over_path7 <- lg_over_bmr$data$learners()[[2]][[7]]$model$tuning_instance$archive$data()
over_path8 <- lg_over_bmr$data$learners()[[2]][[8]]$model$tuning_instance$archive$data()
over_path9 <- lg_over_bmr$data$learners()[[2]][[9]]$model$tuning_instance$archive$data()
over_path10 <- lg_over_bmr$data$learners()[[2]][[10]]$model$tuning_instance$archive$data()

# Order the AUC value by dup size
df_1 <- over_path1[order(over_path1$oversample.ratio)]
df_2 <- over_path2[order(over_path2$oversample.ratio)]
df_3 <- over_path3[order(over_path3$oversample.ratio)]
df_4 <- over_path4[order(over_path4$oversample.ratio)]
df_5 <- over_path5[order(over_path5$oversample.ratio)]
df_6 <- over_path6[order(over_path6$oversample.ratio)]
df_7 <- over_path7[order(over_path7$oversample.ratio)]
df_8 <- over_path8[order(over_path8$oversample.ratio)]
df_9 <- over_path8[order(over_path9$oversample.ratio)]
df_10 <- over_path10[order(over_path10$oversample.ratio)]

df <- data_frame(
  oversample.ratio = df_1$oversample.ratio,
  average_classif.auc = c(mean(c(df_1$classif.auc[1],df_2$classif.auc[1],df_3$classif.auc[1],df_4$classif.auc[1],df_5$classif.auc[1],
                                 df_6$classif.auc[1],df_7$classif.auc[1],df_8$classif.auc[1],df_9$classif.auc[1],df_10$classif.auc[1])),
                          mean(c(df_1$classif.auc[2],df_2$classif.auc[2],df_3$classif.auc[2],df_4$classif.auc[2],df_5$classif.auc[2],
                                 df_6$classif.auc[2],df_7$classif.auc[2],df_8$classif.auc[2],df_9$classif.auc[2],df_10$classif.auc[2])),
                          mean(c(df_1$classif.auc[3],df_2$classif.auc[3],df_3$classif.auc[3],df_4$classif.auc[3],df_5$classif.auc[3],
                                 df_6$classif.auc[3],df_7$classif.auc[3],df_8$classif.auc[3],df_9$classif.auc[3],df_10$classif.auc[3])),
                          mean(c(df_1$classif.auc[4],df_2$classif.auc[4],df_3$classif.auc[4],df_4$classif.auc[4],df_5$classif.auc[4],
                                 df_6$classif.auc[4],df_7$classif.auc[4],df_8$classif.auc[4],df_9$classif.auc[4],df_10$classif.auc[4])),
                          mean(c(df_1$classif.auc[5],df_2$classif.auc[5],df_3$classif.auc[5],df_4$classif.auc[5],df_5$classif.auc[5],
                                 df_6$classif.auc[5],df_7$classif.auc[5],df_8$classif.auc[5],df_9$classif.auc[5],df_10$classif.auc[5])),
                          mean(c(df_1$classif.auc[6],df_2$classif.auc[6],df_3$classif.auc[6],df_4$classif.auc[6],df_5$classif.auc[6],
                                 df_6$classif.auc[6],df_7$classif.auc[6],df_8$classif.auc[6],df_9$classif.auc[6],df_10$classif.auc[6])),
                          mean(c(df_1$classif.auc[7],df_2$classif.auc[7],df_3$classif.auc[7],df_4$classif.auc[7],df_5$classif.auc[7],
                                 df_6$classif.auc[7],df_7$classif.auc[7],df_8$classif.auc[7],df_9$classif.auc[7],df_10$classif.auc[7])),
                          mean(c(df_1$classif.auc[8],df_2$classif.auc[8],df_3$classif.auc[8],df_4$classif.auc[8],df_5$classif.auc[8],
                                 df_6$classif.auc[8],df_7$classif.auc[8],df_8$classif.auc[8],df_9$classif.auc[8],df_10$classif.auc[8])),
                          mean(c(df_1$classif.auc[9],df_2$classif.auc[9],df_3$classif.auc[9],df_4$classif.auc[9],df_5$classif.auc[9],
                                 df_6$classif.auc[9],df_7$classif.auc[9],df_8$classif.auc[9],df_9$classif.auc[9],df_10$classif.auc[9])),
                          mean(c(df_1$classif.auc[10],df_2$classif.auc[10],df_3$classif.auc[10],df_4$classif.auc[10],df_5$classif.auc[10],
                                 df_6$classif.auc[10],df_7$classif.auc[10],df_8$classif.auc[10],df_9$classif.auc[10],df_10$classif.auc[10])),
                          mean(c(df_1$classif.auc[11],df_2$classif.auc[11],df_3$classif.auc[11],df_4$classif.auc[11],df_5$classif.auc[11],
                                 df_6$classif.auc[11],df_7$classif.auc[11],df_8$classif.auc[11],df_9$classif.auc[11],df_10$classif.auc[11])),
                          mean(c(df_1$classif.auc[12],df_2$classif.auc[12],df_3$classif.auc[12],df_4$classif.auc[12],df_5$classif.auc[12],
                                 df_6$classif.auc[12],df_7$classif.auc[12],df_8$classif.auc[12],df_9$classif.auc[12],df_10$classif.auc[12]))
  )
)
# Return plot with size 5*10
oversample_lg <- ggplot(df, aes(
  x = oversample.ratio,
  y = average_classif.auc
  # col = factor(smote.K)
)) +
  theme(legend.position = "none")+
  geom_point(size = 3) +
  geom_line(aes(color = "red"),linetype="dashed")+
  geom_text(aes(label = round(average_classif.auc, 4)),
            vjust = "inward", hjust = "inward",
            show.legend = FALSE) # best smote.dup_size = 22 and average_classif.auc = 0.7236211





#---------------------------------------------------------------------------------------------------
#### SMOTE

library(mlr3learners)
library(mlr3tuning)
library(mlr3pipelines)
library(paradox)


set.seed(2020)

task <- TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y") # logistic learner
lg_learner <- lrn("classif.log_reg", predict_type = "prob")
# create smote
po_smote <- po("smote", dup_size = 50)
lg_smote_learner <- GraphLearner$new(po_smote %>>% lg_learner, predict_type = "prob")

lg_smote_param_set <- ParamSet$new(params = list( ParamInt$new("smote.dup_size", lower = 10, upper = 60), 
                                                  ParamInt$new("smote.K", lower = 10, upper = 25)
))
# set outer_resampling, and create a design with it
inner_rsmp <- rsmp("cv", folds = 10L) 

lg_smote_auto <- AutoTuner$new(
  learner = lg_smote_learner, resampling = inner_rsmp,
  measure = msr("classif.auc"), 
  search_space = lg_smote_param_set, 
  terminator = trm("none"), 
  tuner = tnr("grid_search", resolution = 5)
)
# set outer_resampling, and create a design with it
outer_rsmp <- rsmp("cv", folds = 10L) 

lg_smote_design <- benchmark_grid(
  tasks = task,
  learners = lg_smote_auto, 
  resamplings = outer_rsmp
)



lg_smote_bmr <- benchmark(lg_smote_design, store_models = TRUE)
lg_smote_results <- lg_smote_bmr$aggregate(measures = msr("classif.auc"))
lg_smote_results$classif.auc


smote_path1 <- lg_smote_bmr$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
smote_gg1 <- ggplot(smote_path1, aes(
  x = smote.dup_size,
  y = classif.auc,
  col = factor(smote.K)
)) +
  geom_point(size = 3) +
  geom_line()
# scale_y_continuous(limits = c(0.695,0.715))


smote_path2 <- lg_smote_bmr$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
smote_gg2 <- ggplot(smote_path2, aes(
  x = smote.dup_size,
  y = classif.auc,
  col = factor(smote.K)
)) +
  geom_point(size = 3) +
  geom_line()

smote_path3 <- lg_smote_bmr$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
smote_gg3 <- ggplot(smote_path3, aes(
  x = smote.dup_size,
  y = classif.auc,
  col = factor(smote.K)
)) +
  geom_point(size = 3) +
  geom_line()


# combine all plots in one
library(ggpubr)
ggarrange(smote_gg1, smote_gg2, smote_gg3, common.legend = TRUE, legend = "bottom") # best 33

#-----------------------------------------------------------------------------------------------------------------------------------

set.seed(2020)

# create smote
po_smote <- po("smote", K = 25)
lg_smote_learner <- GraphLearner$new(po_smote %>>% lg_learner, predict_type = "prob")

lg_smote_param_set <- ParamSet$new(params = list( ParamInt$new("smote.dup_size", lower = 10, upper = 60)))

# set outer_resampling, and create a design with it
inner_rsmp <- rsmp("cv", folds = 10L) 

lg_smote_auto <- AutoTuner$new(
  learner = lg_smote_learner, resampling = inner_rsmp,
  measure = msr("classif.auc"), 
  search_space = lg_smote_param_set, 
  terminator = trm("none"), 
  tuner = tnr("grid_search", resolution = 5)
)
# set outer_resampling, and create a design with it
outer_rsmp <- rsmp("cv", folds = 10L) 

lg_smote_design <- benchmark_grid(
  tasks = task,
  learners = lg_smote_auto, 
  resamplings = outer_rsmp
)


lg_smote_bmr <- benchmark(lg_smote_design, store_models = TRUE) # Runtime: 32 min
lg_smote_results <- lg_smote_bmr$aggregate(measures = msr("classif.auc"))
lg_smote_results$classif.auc

# Return the AUC value
smote_path1 <- lg_smote_bmr$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
smote_path2 <- lg_smote_bmr$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
smote_path3 <- lg_smote_bmr$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
smote_path4 <- lg_smote_bmr$data$learners()[[2]][[4]]$model$tuning_instance$archive$data()
smote_path5 <- lg_smote_bmr$data$learners()[[2]][[5]]$model$tuning_instance$archive$data()
smote_path6 <- lg_smote_bmr$data$learners()[[2]][[6]]$model$tuning_instance$archive$data()
smote_path7 <- lg_smote_bmr$data$learners()[[2]][[7]]$model$tuning_instance$archive$data()
smote_path8 <- lg_smote_bmr$data$learners()[[2]][[8]]$model$tuning_instance$archive$data()
smote_path9 <- lg_smote_bmr$data$learners()[[2]][[9]]$model$tuning_instance$archive$data()
smote_path10 <- lg_smote_bmr$data$learners()[[2]][[10]]$model$tuning_instance$archive$data()

# Order the AUC value by dup size
df_1 <- smote_path1[order(smote_path1$smote.dup_size)]
df_2 <- smote_path2[order(smote_path2$smote.dup_size)]
df_3 <- smote_path3[order(smote_path3$smote.dup_size)]
df_4 <- smote_path4[order(smote_path4$smote.dup_size)]
df_5 <- smote_path5[order(smote_path5$smote.dup_size)]
df_6 <- smote_path6[order(smote_path6$smote.dup_size)]
df_7 <- smote_path7[order(smote_path7$smote.dup_size)]
df_8 <- smote_path8[order(smote_path8$smote.dup_size)]
df_9 <- smote_path9[order(smote_path9$smote.dup_size)]
df_10 <- smote_path10[order(smote_path10$smote.dup_size)]

df <- data_frame(
  smote.dup_size = df_1$smote.dup_size,
  average_classif.auc = c(mean(c(df_1$classif.auc[1],df_2$classif.auc[1],df_3$classif.auc[1],df_4$classif.auc[1],df_5$classif.auc[1],
                                 df_6$classif.auc[1],df_7$classif.auc[1],df_8$classif.auc[1],df_9$classif.auc[1],df_10$classif.auc[1])),
                          mean(c(df_1$classif.auc[2],df_2$classif.auc[2],df_3$classif.auc[2],df_4$classif.auc[2],df_5$classif.auc[2],
                                 df_6$classif.auc[2],df_7$classif.auc[2],df_8$classif.auc[2],df_9$classif.auc[2],df_10$classif.auc[2])),
                          mean(c(df_1$classif.auc[3],df_2$classif.auc[3],df_3$classif.auc[3],df_4$classif.auc[3],df_5$classif.auc[3],
                                 df_6$classif.auc[3],df_7$classif.auc[3],df_8$classif.auc[3],df_9$classif.auc[3],df_10$classif.auc[3])),
                          mean(c(df_1$classif.auc[4],df_2$classif.auc[4],df_3$classif.auc[4],df_4$classif.auc[4],df_5$classif.auc[4],
                                 df_6$classif.auc[4],df_7$classif.auc[4],df_8$classif.auc[4],df_9$classif.auc[4],df_10$classif.auc[4])),
                          mean(c(df_1$classif.auc[5],df_2$classif.auc[5],df_3$classif.auc[5],df_4$classif.auc[5],df_5$classif.auc[5],
                                 df_6$classif.auc[5],df_7$classif.auc[5],df_8$classif.auc[5],df_9$classif.auc[5],df_10$classif.auc[5]))
  )
)
# size 
smote_lg <- ggplot(df, aes(
  x = smote.dup_size,
  y = average_classif.auc
  # col = factor(smote.K)
)) +
  geom_point(size = 3) +
  geom_line(aes(color = "red"),linetype="dashed")+
  theme(legend.position = "none")+
  geom_text(aes(label = round(average_classif.auc, 4)),
            vjust = "inward", hjust = "inward",
            show.legend = FALSE) # best smote.dup_size = 22 and average_classif.auc = 0.7236211




####################################################################
##########                Step 5.2: KNN               ##############
####################################################################

#------------------------------------------------------------------------------------------
#### Oversample: Parameter: oversample.ratio

task <- TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y") 

# logistic learner
knn_learner <- lrn("classif.kknn", predict_type = "prob")

po_over <- po("classbalancing",
              id = "oversample", adjust = "minor",
              reference = "minor", shuffle = FALSE, ratio = 5)

# create oversample
knn_over_learner <- GraphLearner$new(po_over %>>% knn_learner, predict_type = "prob")

# define parameter
knn_over_param_set <- ParamSet$new(list(ParamDbl$new("oversample.ratio", 
                                                    lower = 30, upper = 90)))

# set inner resampling
inner_rsmp <- rsmp("cv", folds = 10L)

# create autotuner
knn_over_auto <- AutoTuner$new(
  learner = knn_over_learner, resampling = inner_rsmp,
  measure = msr("classif.auc"), 
  search_space = knn_over_param_set,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 10)
)

# set outer_resampling, and create a design with it
outer_rsmp <- rsmp("cv", folds = 10L)

knn_over_design <- benchmark_grid(
  tasks = task,
  learners = knn_over_auto,
  resamplings = outer_rsmp
)

#-----

set.seed(2020)

# create benchmark
knn_over_bmr <- benchmark(knn_over_design, store_models = TRUE) # Runtime:1h

# aggregate to get results of model
knn_over_results <- knn_over_bmr$aggregate(measure = msr("classif.auc"))

# inspect auc value.
knn_over_results$classif.auc


# plot oversample ratio vs AUC
library(ggplot2)

knn_over_path1 <- knn_over_bmr$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
knn_over_path2 <- knn_over_bmr$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
knn_over_path3 <- knn_over_bmr$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
knn_over_path4 <- knn_over_bmr$data$learners()[[2]][[4]]$model$tuning_instance$archive$data()
knn_over_path5 <- knn_over_bmr$data$learners()[[2]][[5]]$model$tuning_instance$archive$data()
knn_over_path6 <- knn_over_bmr$data$learners()[[2]][[6]]$model$tuning_instance$archive$data()
knn_over_path7 <- knn_over_bmr$data$learners()[[2]][[7]]$model$tuning_instance$archive$data()
knn_over_path8 <- knn_over_bmr$data$learners()[[2]][[8]]$model$tuning_instance$archive$data()
knn_over_path9 <- knn_over_bmr$data$learners()[[2]][[9]]$model$tuning_instance$archive$data()
knn_over_path10 <- knn_over_bmr$data$learners()[[2]][[10]]$model$tuning_instance$archive$data()

# Order the AUC value by dup size
df_1 <- knn_over_path1[order(knn_over_path1$oversample.ratio)]
df_2 <- knn_over_path2[order(knn_over_path2$oversample.ratio)]
df_3 <- knn_over_path3[order(knn_over_path3$oversample.ratio)]
df_4 <- knn_over_path4[order(knn_over_path4$oversample.ratio)]
df_5 <- knn_over_path5[order(knn_over_path5$oversample.ratio)]
df_6 <- knn_over_path6[order(knn_over_path6$oversample.ratio)]
df_7 <- knn_over_path7[order(knn_over_path7$oversample.ratio)]
df_8 <- knn_over_path8[order(knn_over_path8$oversample.ratio)]
df_9 <- knn_over_path8[order(knn_over_path9$oversample.ratio)]
df_10 <- knn_over_path10[order(knn_over_path10$oversample.ratio)]

knn_df <- data_frame(
  oversample.ratio = df_1$oversample.ratio,
  average_classif.auc = c(mean(c(df_1$classif.auc[1],df_2$classif.auc[1],df_3$classif.auc[1],df_4$classif.auc[1],df_5$classif.auc[1],
                                 df_6$classif.auc[1],df_7$classif.auc[1],df_8$classif.auc[1],df_9$classif.auc[1],df_10$classif.auc[1])),
                          mean(c(df_1$classif.auc[2],df_2$classif.auc[2],df_3$classif.auc[2],df_4$classif.auc[2],df_5$classif.auc[2],
                                 df_6$classif.auc[2],df_7$classif.auc[2],df_8$classif.auc[2],df_9$classif.auc[2],df_10$classif.auc[2])),
                          mean(c(df_1$classif.auc[3],df_2$classif.auc[3],df_3$classif.auc[3],df_4$classif.auc[3],df_5$classif.auc[3],
                                 df_6$classif.auc[3],df_7$classif.auc[3],df_8$classif.auc[3],df_9$classif.auc[3],df_10$classif.auc[3])),
                          mean(c(df_1$classif.auc[4],df_2$classif.auc[4],df_3$classif.auc[4],df_4$classif.auc[4],df_5$classif.auc[4],
                                 df_6$classif.auc[4],df_7$classif.auc[4],df_8$classif.auc[4],df_9$classif.auc[4],df_10$classif.auc[4])),
                          mean(c(df_1$classif.auc[5],df_2$classif.auc[5],df_3$classif.auc[5],df_4$classif.auc[5],df_5$classif.auc[5],
                                 df_6$classif.auc[5],df_7$classif.auc[5],df_8$classif.auc[5],df_9$classif.auc[5],df_10$classif.auc[5])),
                          mean(c(df_1$classif.auc[6],df_2$classif.auc[6],df_3$classif.auc[6],df_4$classif.auc[6],df_5$classif.auc[6],
                                 df_6$classif.auc[6],df_7$classif.auc[6],df_8$classif.auc[6],df_9$classif.auc[6],df_10$classif.auc[6])),
                          mean(c(df_1$classif.auc[7],df_2$classif.auc[7],df_3$classif.auc[7],df_4$classif.auc[7],df_5$classif.auc[7],
                                 df_6$classif.auc[7],df_7$classif.auc[7],df_8$classif.auc[7],df_9$classif.auc[7],df_10$classif.auc[7])),
                          mean(c(df_1$classif.auc[8],df_2$classif.auc[8],df_3$classif.auc[8],df_4$classif.auc[8],df_5$classif.auc[8],
                                 df_6$classif.auc[8],df_7$classif.auc[8],df_8$classif.auc[8],df_9$classif.auc[8],df_10$classif.auc[8])),
                          mean(c(df_1$classif.auc[9],df_2$classif.auc[9],df_3$classif.auc[9],df_4$classif.auc[9],df_5$classif.auc[9],
                                 df_6$classif.auc[9],df_7$classif.auc[9],df_8$classif.auc[9],df_9$classif.auc[9],df_10$classif.auc[9])),
                          mean(c(df_1$classif.auc[10],df_2$classif.auc[10],df_3$classif.auc[10],df_4$classif.auc[10],df_5$classif.auc[10],
                                 df_6$classif.auc[10],df_7$classif.auc[10],df_8$classif.auc[10],df_9$classif.auc[10],df_10$classif.auc[10]))
  )
)
# Return plot with size 5*10
oversample_knn <- ggplot(knn_df, aes(
  x = oversample.ratio,
  y = average_classif.auc
  # col = factor(smote.K)
)) +
  theme(legend.position = "none")+
  geom_point(size = 3) +
  geom_line(aes(color = "red"),linetype="dashed")+
  geom_text(aes(label = round(average_classif.auc, 4)),
            vjust = "inward", hjust = "inward",
            show.legend = FALSE) # best oversample.ratio =  85 and average_classif.auc = 0.7214

oversample_knn

#------------------------------------------------------------------------------------------
#### Oversample: Parameter: distance

# create oversampling with previously tuned value (e.g. oversample.ratio =  85)
po_over_tuned <- po("classbalancing",
                    id = "oversample", adjust = "minor",
                    reference = "minor", shuffle = FALSE, ratio = 85) 

knn_over_lrn <- GraphLearner$new(po_over_tuned %>>% lrn("classif.kknn",
                                                        predict_type = "prob"), predict_type = "prob")

# set distance range for tuning
para_dist <- ParamSet$new(params = list(ParamInt$new("classif.kknn.distance", lower = 1, upper = 5)))

# set inner resampling
inner_rsmp <- rsmp("cv", folds = 10L)

knn_over_dist <- AutoTuner$new(
  learner = knn_over_lrn, resampling = inner_rsmp,
  measure = msr("classif.auc"), search_space = para_dist,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 10))

# set outer_resampling, and create a design with it
outer_rsmp <- rsmp("cv", folds = 10L)

knn_over_design <- benchmark_grid(
  tasks = task,
  learners = knn_over_dist,
  resamplings = outer_rsmp
)

set.seed(2020)


# create benchmark
knn_over_bmr_dist <- benchmark(knn_over_design, store_models = TRUE) # Runtime: 4h

# aggregate to get results of model
knn_over_results <- knn_over_bmr_dist$aggregate(measure = msr("classif.auc"))

# inspect auc value.
knn_over_results$classif.auc

# plot oversample ratio vs AUC
library(ggplot2)

knn_over_path1 <- knn_over_bmr_dist$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
knn_over_path2 <- knn_over_bmr_dist$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
knn_over_path3 <- knn_over_bmr_dist$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
knn_over_path4 <- knn_over_bmr_dist$data$learners()[[2]][[4]]$model$tuning_instance$archive$data()
knn_over_path5 <- knn_over_bmr_dist$data$learners()[[2]][[5]]$model$tuning_instance$archive$data()
knn_over_path6 <- knn_over_bmr_dist$data$learners()[[2]][[6]]$model$tuning_instance$archive$data()
knn_over_path7 <- knn_over_bmr_dist$data$learners()[[2]][[7]]$model$tuning_instance$archive$data()
knn_over_path8 <- knn_over_bmr_dist$data$learners()[[2]][[8]]$model$tuning_instance$archive$data()
knn_over_path9 <- knn_over_bmr_dist$data$learners()[[2]][[9]]$model$tuning_instance$archive$data()
knn_over_path10 <- knn_over_bmr_dist$data$learners()[[2]][[10]]$model$tuning_instance$archive$data()

# Order the AUC value by dup size
df_1 <- knn_over_path1[order(knn_over_path1$classif.kknn.distance)]
df_2 <- knn_over_path2[order(knn_over_path2$classif.kknn.distance)]
df_3 <- knn_over_path3[order(knn_over_path3$classif.kknn.distance)]
df_4 <- knn_over_path4[order(knn_over_path4$classif.kknn.distance)]
df_5 <- knn_over_path5[order(knn_over_path5$classif.kknn.distance)]
df_6 <- knn_over_path6[order(knn_over_path6$classif.kknn.distance)]
df_7 <- knn_over_path7[order(knn_over_path7$classif.kknn.distance)]
df_8 <- knn_over_path8[order(knn_over_path8$classif.kknn.distance)]
df_9 <- knn_over_path8[order(knn_over_path9$classif.kknn.distance)]
df_10 <- knn_over_path10[order(knn_over_path10$classif.kknn.distance)]

knn_df <- data_frame(
  classif.kknn.distance = df_1$classif.kknn.distance,
  average_classif.auc = c(mean(c(df_1$classif.auc[1],df_2$classif.auc[1],df_3$classif.auc[1],df_4$classif.auc[1],df_5$classif.auc[1],
                                 df_6$classif.auc[1],df_7$classif.auc[1],df_8$classif.auc[1],df_9$classif.auc[1],df_10$classif.auc[1])),
                          mean(c(df_1$classif.auc[2],df_2$classif.auc[2],df_3$classif.auc[2],df_4$classif.auc[2],df_5$classif.auc[2],
                                 df_6$classif.auc[2],df_7$classif.auc[2],df_8$classif.auc[2],df_9$classif.auc[2],df_10$classif.auc[2])),
                          mean(c(df_1$classif.auc[3],df_2$classif.auc[3],df_3$classif.auc[3],df_4$classif.auc[3],df_5$classif.auc[3],
                                 df_6$classif.auc[3],df_7$classif.auc[3],df_8$classif.auc[3],df_9$classif.auc[3],df_10$classif.auc[3])),
                          mean(c(df_1$classif.auc[4],df_2$classif.auc[4],df_3$classif.auc[4],df_4$classif.auc[4],df_5$classif.auc[4],
                                 df_6$classif.auc[4],df_7$classif.auc[4],df_8$classif.auc[4],df_9$classif.auc[4],df_10$classif.auc[4])),
                          mean(c(df_1$classif.auc[5],df_2$classif.auc[5],df_3$classif.auc[5],df_4$classif.auc[5],df_5$classif.auc[5],
                                 df_6$classif.auc[5],df_7$classif.auc[5],df_8$classif.auc[5],df_9$classif.auc[5],df_10$classif.auc[5]))

  )
)
# Return plot with size 5*10
oversample_knn <- ggplot(knn_df, aes(
  x = classif.kknn.distance,
  y = average_classif.auc
  # col = factor(smote.K)
)) +
  theme(legend.position = "none")+
  geom_point(size = 3) +
  geom_line(aes(color = "red"),linetype="dashed")+
  geom_text(aes(label = round(average_classif.auc, 4)),
            vjust = "inward", hjust = "inward",
            show.legend = FALSE) # best oversample.ratio =  85 and average_classif.auc = 0.7214

oversample_knn



#------------------------------------------------------------------------------------------
#### Oversample: Parameter: kernel

kernel_type <- c("rectangular", "triangular", "epanechnikov", "biweight",
                 "triweight", "cos", "inv", "gaussian", "rank", "optimal")

knn_over_lrn <- GraphLearner$new(po_over_tuned %>>% lrn("classif.kknn", predict_type = "prob", distance=1),
                                 predict_type = "prob")

para_kernel <- ParamSet$new(params = list(ParamFct$new("classif.kknn.kernel",
                                                       levels = kernel_type)))

# set inner resampling
inner_rsmp <- rsmp("cv", folds = 5L)


knn_over_kernel <- AutoTuner$new(
  learner = knn_over_lrn, resampling = inner_rsmp,
  measure = msr("classif.auc"), search_space = para_kernel,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 5))

# set outer_resampling, and create a design with it
outer_rsmp <- rsmp("cv", folds = 5L)

design_over_kernel <- benchmark_grid( tasks = task,
                                      learners = knn_over_kernel, resamplings = outer_rsmp)

bmr_over_kern <- benchmark(design_over_kernel, store_models = TRUE) # Runtime: 

over_kern_para <- bmr_over_kern$score() %>%
  pull(learner) %>%
  map(pluck(c(function(x) x$tuning_result)))

# extract top kernels from 5-fold cv
selected_kernel <- c(over_kern_para[[1]]$classif.kknn.kernel, 
                     over_kern_para[[2]]$classif.kknn.kernel, 
                     over_kern_para[[3]]$classif.kknn.kernel,
                     over_kern_para[[4]]$classif.kknn.kernel,
                     over_kern_para[[5]]$classif.kknn.kernel)

# list selected top kernel
selected_kernel



df_knn <- data.frame(classif.kknn.kernel =c("gaussian", "inv","triangular","epanechnikov","biweight "),
                 len=c(0.71089, 0.71789,0.72775,0.71946,0.7325))

p<-ggplot(data=df_knn, aes(x=classif.kknn.kernel, y=len, fill=classif.kknn.kernel)) +
  geom_bar(stat="identity")+
  theme_bw()+
  geom_text(aes(label=len), vjust=-0.3, size=5)+
  xlab("classif.kknn.kernel") + ylab("average_classif.auc")+
  ggtitle("Approval Rate")+
  theme(legend.position="none")+
  #theme(axis.text.x=element_text(size=rel(3)))+
  theme(text = element_text(size=15))+
  ggtitle("Comparison of the average AUC level for different kernel") 
p  



#------------------------------------------------------------------------------------------
#### Smote

knn_learner <- lrn("classif.kknn", predict_type = "prob")
# create smote, and combine it with the current learner
po_smote <- po("smote", dup_size = 50)

lrn_smote <- GraphLearner$new(po_smote %>>% knn_learner, predict_type = "prob")

# setting smote parameters' tuning range
knn_smote_param_set <- ParamSet$new(params = list(ParamInt$new("smote.dup_size", lower = 10, upper = 60),
                                             ParamInt$new("smote.K", lower = 10, upper = 25)))

# create autotuner, using the inner sampling and tuning parameter with random search
inner_rsmp <- rsmp("cv",folds = 5L)

knn_smote_auto <- AutoTuner$new(learner = lrn_smote, resampling = inner_rsmp,
                                measure = msr("classif.auc"),
                                search_space = knn_smote_param_set,
                                terminator = trm("none"),
                                tuner = tnr("grid_search", resolution = 5))


# set outer_resampling, and create a design with it
outer_rsmp <- rsmp("cv", folds = 5L) 

design_smote = benchmark_grid(
  tasks = task,
  learners = knn_smote_auto, resamplings = outer_rsmp
)

set.seed(2020)


# run the benchmark, save the results afterwards
# knn_smote_bmr <- run_benchmark(design_smote)
knn_smote_bmr <- benchmark(design_smote, store_models = TRUE) # Runtime: 2 h

# plot oversample ratio vs AUC
library(ggplot2)

knn_over_path1 <- knn_smote_bmr$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
knn_over_path2 <- knn_smote_bmr$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
knn_over_path3 <- knn_smote_bmr$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
knn_over_path4 <- knn_smote_bmr$data$learners()[[2]][[4]]$model$tuning_instance$archive$data()
knn_over_path5 <- knn_smote_bmr$data$learners()[[2]][[5]]$model$tuning_instance$archive$data()

# Order the AUC value by dup size
df_1 <- knn_over_path1[order(knn_over_path1$smote.dup_size, knn_over_path1$smote.K)]
df_2 <- knn_over_path2[order(knn_over_path2$smote.dup_size, knn_over_path1$smote.K)]
df_3 <- knn_over_path3[order(knn_over_path3$smote.dup_size, knn_over_path1$smote.K)]
df_4 <- knn_over_path4[order(knn_over_path4$smote.dup_size, knn_over_path1$smote.K)]
df_5 <- knn_over_path5[order(knn_over_path5$smote.dup_size, knn_over_path1$smote.K)]

# df_1 <- knn_over_path1[order(knn_over_path1$classif.auc)] %>% filter(smote.K == c(30,33,35,40,45))



knn_df <- data_frame(
  smote.dup_size = df_1$smote.dup_size,
  smote.K = df_1$smote.K,
  average_classif.auc = c(mean(c(df_1$classif.auc[1],df_2$classif.auc[1],df_3$classif.auc[1],df_4$classif.auc[1],df_5$classif.auc[1])),
                          mean(c(df_1$classif.auc[2],df_2$classif.auc[2],df_3$classif.auc[2],df_4$classif.auc[2],df_5$classif.auc[2])),
                          mean(c(df_1$classif.auc[3],df_2$classif.auc[3],df_3$classif.auc[3],df_4$classif.auc[3],df_5$classif.auc[3])),
                          mean(c(df_1$classif.auc[4],df_2$classif.auc[4],df_3$classif.auc[4],df_4$classif.auc[4],df_5$classif.auc[4])),
                          mean(c(df_1$classif.auc[5],df_2$classif.auc[5],df_3$classif.auc[5],df_4$classif.auc[5],df_5$classif.auc[5])),
                          mean(c(df_1$classif.auc[6],df_2$classif.auc[6],df_3$classif.auc[6],df_4$classif.auc[6],df_5$classif.auc[6])),
                          mean(c(df_1$classif.auc[7],df_2$classif.auc[7],df_3$classif.auc[7],df_4$classif.auc[7],df_5$classif.auc[7])),
                          mean(c(df_1$classif.auc[8],df_2$classif.auc[8],df_3$classif.auc[8],df_4$classif.auc[8],df_5$classif.auc[8])),
                          mean(c(df_1$classif.auc[9],df_2$classif.auc[9],df_3$classif.auc[9],df_4$classif.auc[9],df_5$classif.auc[9])),
                          mean(c(df_1$classif.auc[10],df_2$classif.auc[10],df_3$classif.auc[10],df_4$classif.auc[10],df_5$classif.auc[10])),
                          mean(c(df_1$classif.auc[11],df_2$classif.auc[11],df_3$classif.auc[11],df_4$classif.auc[11],df_5$classif.auc[11])),
                          mean(c(df_1$classif.auc[12],df_2$classif.auc[12],df_3$classif.auc[12],df_4$classif.auc[12],df_5$classif.auc[12])),
                          mean(c(df_1$classif.auc[13],df_2$classif.auc[13],df_3$classif.auc[13],df_4$classif.auc[13],df_5$classif.auc[13])),
                          mean(c(df_1$classif.auc[14],df_2$classif.auc[14],df_3$classif.auc[14],df_4$classif.auc[14],df_5$classif.auc[14])),
                          mean(c(df_1$classif.auc[15],df_2$classif.auc[15],df_3$classif.auc[15],df_4$classif.auc[15],df_5$classif.auc[15])),
                          mean(c(df_1$classif.auc[16],df_2$classif.auc[16],df_3$classif.auc[16],df_4$classif.auc[16],df_5$classif.auc[16])),
                          mean(c(df_1$classif.auc[17],df_2$classif.auc[17],df_3$classif.auc[17],df_4$classif.auc[17],df_5$classif.auc[17])),
                          mean(c(df_1$classif.auc[18],df_2$classif.auc[18],df_3$classif.auc[18],df_4$classif.auc[18],df_5$classif.auc[18])),
                          mean(c(df_1$classif.auc[19],df_2$classif.auc[19],df_3$classif.auc[19],df_4$classif.auc[19],df_5$classif.auc[19])),
                          mean(c(df_1$classif.auc[20],df_2$classif.auc[20],df_3$classif.auc[20],df_4$classif.auc[20],df_5$classif.auc[20])),
                          mean(c(df_1$classif.auc[21],df_2$classif.auc[21],df_3$classif.auc[21],df_4$classif.auc[21],df_5$classif.auc[21])),
                          mean(c(df_1$classif.auc[22],df_2$classif.auc[22],df_3$classif.auc[22],df_4$classif.auc[22],df_5$classif.auc[22])),
                          mean(c(df_1$classif.auc[23],df_2$classif.auc[23],df_3$classif.auc[23],df_4$classif.auc[23],df_5$classif.auc[23])),
                          mean(c(df_1$classif.auc[24],df_2$classif.auc[24],df_3$classif.auc[24],df_4$classif.auc[24],df_5$classif.auc[24])),
                          mean(c(df_1$classif.auc[25],df_2$classif.auc[25],df_3$classif.auc[25],df_4$classif.auc[25],df_5$classif.auc[25]))
                          
  )
)

# Return plot with size 5*10
smote_gg1 <- ggplot(knn_df, aes(
  x = smote.dup_size,
  y = average_classif.auc, col = factor(smote.K))) + geom_point(size = 3) +
  geom_line()+
  geom_text(aes(label = round(average_classif.auc, 4)),
            vjust = "inward", hjust = "inward",
            show.legend = FALSE)+
  theme(legend.position = "bottom")
  
smote_gg1





#------------------------------------------------------------------------------------------
#### Smote 2


# list all kernel types
kernel_type <- c( "triangular", "epanechnikov", "biweight", "inv", "gaussian")

# set parameter tuning for each parameter
para_k <- ParamSet$new(params = list(ParamInt$new("classif.kknn.k",
                                                  lower = 10, upper = 100), 
                                     ParamFct$new("classif.kknn.kernel",
                                                  levels = kernel_type)))

# para_dist <- ParamSet$new(params = list(ParamInt$new("classif.kknn.distance", lower = 1, upper = 5)))

# para_kernel <- ParamSet$new(params = list(ParamFct$new("classif.kknn.kernel", levels = kernel_type)))

# define smote, and combine it with knn learner
po_smote_tuned <- po("smote", dup_size = 60, K = 25) # based on previous results

knn_smote_lrn <- GraphLearner$new(po_smote_tuned %>>% lrn("classif.kknn",predict_type = "prob"), 
                                  predict_type = "prob")


# create autotuner, using the inner sampling and tuning parameter with random search
inner_rsmp <- rsmp("cv",folds = 5L)

knn_smote_auto <- AutoTuner$new(learner = knn_smote_lrn, resampling = inner_rsmp,
                                measure = msr("classif.auc"),
                                search_space = para_k,
                                terminator = trm("none"),
                                tuner = tnr("grid_search", resolution = 5))


# set outer_resampling, and create a design with it
outer_rsmp <- rsmp("cv", folds = 5L) 

design_smote = benchmark_grid(
  tasks = task,
  learners = knn_smote_auto, 
  resamplings = outer_rsmp
)

set.seed(2020)


# run the benchmark, save the results afterwards
# knn_smote_bmr <- run_benchmark(design_smote)
knn_smote_bmr <- benchmark(design_smote, store_models = TRUE) # Runtime: 


# plot oversample ratio vs AUC
library(ggplot2)

knn_over_path1 <- knn_smote_bmr$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
knn_over_path2 <- knn_smote_bmr$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
knn_over_path3 <- knn_smote_bmr$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
knn_over_path4 <- knn_smote_bmr$data$learners()[[2]][[4]]$model$tuning_instance$archive$data()
knn_over_path5 <- knn_smote_bmr$data$learners()[[2]][[5]]$model$tuning_instance$archive$data()

# Order the AUC value by dup size
df_1 <- knn_over_path1[order(knn_over_path1$classif.kknn.k, knn_over_path1$classif.kknn.kernel)]
df_2 <- knn_over_path2[order(knn_over_path2$classif.kknn.k, knn_over_path1$classif.kknn.kernel)]
df_3 <- knn_over_path3[order(knn_over_path3$classif.kknn.k, knn_over_path1$classif.kknn.kernel)]
df_4 <- knn_over_path4[order(knn_over_path4$classif.kknn.k, knn_over_path1$classif.kknn.kernel)]
df_5 <- knn_over_path5[order(knn_over_path5$classif.kknn.k, knn_over_path1$classif.kknn.kernel)]

knn_df_2 <- data_frame(
  classif.kknn.k = df_1$classif.kknn.k,
  classif.kknn.kernel = df_1$classif.kknn.kernel,
  average_classif.auc = c(mean(c(df_1$classif.auc[1],df_2$classif.auc[1],df_3$classif.auc[1],df_4$classif.auc[1],df_5$classif.auc[1])),
                          mean(c(df_1$classif.auc[2],df_2$classif.auc[2],df_3$classif.auc[2],df_4$classif.auc[2],df_5$classif.auc[2])),
                          mean(c(df_1$classif.auc[3],df_2$classif.auc[3],df_3$classif.auc[3],df_4$classif.auc[3],df_5$classif.auc[3])),
                          mean(c(df_1$classif.auc[4],df_2$classif.auc[4],df_3$classif.auc[4],df_4$classif.auc[4],df_5$classif.auc[4])),
                          mean(c(df_1$classif.auc[5],df_2$classif.auc[5],df_3$classif.auc[5],df_4$classif.auc[5],df_5$classif.auc[5])),
                          mean(c(df_1$classif.auc[6],df_2$classif.auc[6],df_3$classif.auc[6],df_4$classif.auc[6],df_5$classif.auc[6])),
                          mean(c(df_1$classif.auc[7],df_2$classif.auc[7],df_3$classif.auc[7],df_4$classif.auc[7],df_5$classif.auc[7])),
                          mean(c(df_1$classif.auc[8],df_2$classif.auc[8],df_3$classif.auc[8],df_4$classif.auc[8],df_5$classif.auc[8])),
                          mean(c(df_1$classif.auc[9],df_2$classif.auc[9],df_3$classif.auc[9],df_4$classif.auc[9],df_5$classif.auc[9])),
                          mean(c(df_1$classif.auc[10],df_2$classif.auc[10],df_3$classif.auc[10],df_4$classif.auc[10],df_5$classif.auc[10])),
                          mean(c(df_1$classif.auc[11],df_2$classif.auc[11],df_3$classif.auc[11],df_4$classif.auc[11],df_5$classif.auc[11])),
                          mean(c(df_1$classif.auc[12],df_2$classif.auc[12],df_3$classif.auc[12],df_4$classif.auc[12],df_5$classif.auc[12])),
                          mean(c(df_1$classif.auc[13],df_2$classif.auc[13],df_3$classif.auc[13],df_4$classif.auc[13],df_5$classif.auc[13])),
                          mean(c(df_1$classif.auc[14],df_2$classif.auc[14],df_3$classif.auc[14],df_4$classif.auc[14],df_5$classif.auc[14])),
                          mean(c(df_1$classif.auc[15],df_2$classif.auc[15],df_3$classif.auc[15],df_4$classif.auc[15],df_5$classif.auc[15])),
                          mean(c(df_1$classif.auc[16],df_2$classif.auc[16],df_3$classif.auc[16],df_4$classif.auc[16],df_5$classif.auc[16])),
                          mean(c(df_1$classif.auc[17],df_2$classif.auc[17],df_3$classif.auc[17],df_4$classif.auc[17],df_5$classif.auc[17])),
                          mean(c(df_1$classif.auc[18],df_2$classif.auc[18],df_3$classif.auc[18],df_4$classif.auc[18],df_5$classif.auc[18])),
                          mean(c(df_1$classif.auc[19],df_2$classif.auc[19],df_3$classif.auc[19],df_4$classif.auc[19],df_5$classif.auc[19])),
                          mean(c(df_1$classif.auc[20],df_2$classif.auc[20],df_3$classif.auc[20],df_4$classif.auc[20],df_5$classif.auc[20])),
                          mean(c(df_1$classif.auc[21],df_2$classif.auc[21],df_3$classif.auc[21],df_4$classif.auc[21],df_5$classif.auc[21])),
                          mean(c(df_1$classif.auc[22],df_2$classif.auc[22],df_3$classif.auc[22],df_4$classif.auc[22],df_5$classif.auc[22])),
                          mean(c(df_1$classif.auc[23],df_2$classif.auc[23],df_3$classif.auc[23],df_4$classif.auc[23],df_5$classif.auc[23])),
                          mean(c(df_1$classif.auc[24],df_2$classif.auc[24],df_3$classif.auc[24],df_4$classif.auc[24],df_5$classif.auc[24])),
                          mean(c(df_1$classif.auc[25],df_2$classif.auc[25],df_3$classif.auc[25],df_4$classif.auc[25],df_5$classif.auc[25]))
                          
  )
)

# Return plot with size 5*10
smote_gg2 <- ggplot(knn_df_2, aes(
  x = classif.kknn.k,
  y = average_classif.auc, col = factor(classif.kknn.kernel))) + geom_point(size = 3) +
  geom_line()+
  geom_text(aes(label = round(average_classif.auc, 4)),
            vjust = "inward", hjust = "inward",
            show.legend = FALSE)+
  theme(legend.position = "bottom")

smote_gg2


####################################################################
##########                Step 5.3: SVM              ###############
####################################################################

#------------------------------------------------------------------------------------------

#### Oversample: Parameter: oversample.ratio

task <- TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y") 

## test for best matches with smote function's parameter with SVM learner
svm_learner <- lrn("classif.svm", predict_type = "prob")

po_over <- po("classbalancing",
              id = "oversample", adjust = "minor",
              reference = "minor", shuffle = FALSE, ratio = 5)

# create oversample
svm_over_learner <- GraphLearner$new(po_over %>>% svm_learner, predict_type = "prob")

# define parameter
svm_over_param_set <- ParamSet$new(list(ParamDbl$new("oversample.ratio", 
                                                     lower = 10, upper = 70)))
# set inner resampling
inner_rsmp <- rsmp("cv", folds = 3L)

# create autotuner
svm_over_auto <- AutoTuner$new(
  learner = svm_over_learner, resampling = inner_rsmp,
  measure = msr("classif.auc"), 
  search_space = svm_over_param_set,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 5)
)

# set outer_resampling, and create a design with it
outer_rsmp <- rsmp("cv", folds = 3L)

svm_over_design <- benchmark_grid(
  tasks = task,
  learners = svm_over_auto,
  resamplings = outer_rsmp
)

#-----

set.seed(2020)

# create benchmark
svm_over_bmr <- benchmark(svm_over_design, store_models = TRUE) # Runtime:1h

# aggregate to get results of model
svm_over_results <- svm_over_bmr$aggregate(measure = msr("classif.auc"))



# plot oversample ratio vs AUC
library(ggplot2)

svm_over_path1 <- svm_over_bmr$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
svm_over_path2 <- svm_over_bmr$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
svm_over_path3 <- svm_over_bmr$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()
svm_over_path4 <- svm_over_bmr$data$learners()[[2]][[4]]$model$tuning_instance$archive$data()
svm_over_path5 <- svm_over_bmr$data$learners()[[2]][[5]]$model$tuning_instance$archive$data()

# Order the AUC value by dup size
df_1 <- svm_over_path1[order(svm_over_path1$classif.auc)]
df_2 <- svm_over_path2[order(svm_over_path2$classif.auc)]
df_3 <- svm_over_path3[order(svm_over_path3$classif.auc)]


svm_df <- data_frame(
  oversample.ratio = df_1$oversample.ratio,
  average_classif.auc = c(mean(c(df_1$classif.auc[1],df_2$classif.auc[1],df_3$classif.auc[1])),
                          mean(c(df_1$classif.auc[2],df_2$classif.auc[2],df_3$classif.auc[2])),
                          mean(c(df_1$classif.auc[3],df_2$classif.auc[3],df_3$classif.auc[3])),
                          mean(c(df_1$classif.auc[4],df_2$classif.auc[4],df_3$classif.auc[4])),
                          mean(c(df_1$classif.auc[5],df_2$classif.auc[5],df_3$classif.auc[5]))

  )
)

# Return plot with size 5*10
oversample_knn <- ggplot(svm_df, aes(
  x = oversample.ratio,
  y = average_classif.auc
  # col = factor(smote.K)
)) +
  theme(legend.position = "none")+
  geom_point(size = 3) +
  geom_line(aes(color = "red"),linetype="dashed")+
  geom_text(aes(label = round(average_classif.auc, 4)),
            vjust = "inward", hjust = "inward",
            show.legend = FALSE) # best oversample.ratio =  85 and average_classif.auc = 0.7214

oversample_knn


#------------------------------------------------------------------------------------------
#### Oversampling: fine-tuning kernel

task <- TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y") 

kernel_type <- c("linear", "polynomial", "radial", "sigmoid")

# create oversampling with previously tuned value ratio = 25 with maximal level of AUC 
po_over_tuned <- po("classbalancing",
                    id = "oversample", adjust = "minor",
                    reference = "minor", shuffle = FALSE, ratio = 25) 

svm_over_lrn <- GraphLearner$new(po_over_tuned %>>% lrn("classif.svm", predict_type = "prob"),
                                 predict_type = "prob")

para_kernel <- ParamSet$new(params = list(ParamFct$new("classif.svm.kernel",
                                                       levels = kernel_type)))

# set inner resampling
inner_rsmp <- rsmp("cv", folds = 3L)


svm_over_kernel <- AutoTuner$new(
  learner = svm_over_lrn, resampling = inner_rsmp,
  measure = msr("classif.auc"), search_space = para_kernel,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 5))

# set outer_resampling, and create a design with it
outer_rsmp <- rsmp("cv", folds = 3L)

design_over_kernel <- benchmark_grid( tasks = task,
                                      learners = svm_over_kernel, resamplings = outer_rsmp)

bmr_over_kern <- benchmark(design_over_kernel, store_models = TRUE) # Runtime: 5 h

svm2_smote_path1 <- bmr_over_kern$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()[c(1,3,4),]
svm2_smote_path2 <- bmr_over_kern$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()[c(1,3,4),]
svm2_smote_path3 <- bmr_over_kern$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()[c(1,3,4),]


p<-ggplot(data=svm2_smote_path1, aes(x=classif.svm.kernel, y=classif.auc, fill=classif.svm.kernel)) +
  geom_bar(stat="identity")+
  theme_bw()+
  geom_text(aes(label=round(classif.auc,4)), vjust=-0.3, size=5)+
  xlab("classif.kknn.kernel") + ylab("average_classif.auc")+
  ggtitle("Approval Rate")+
  theme(legend.position="none")+
  #theme(axis.text.x=element_text(size=rel(3)))+
  theme(text = element_text(size=15))+
  ggtitle("Comparison of the average AUC level for different kernel by using SVM") 
p  




#------------------------------------------------------------------------------------------
#### SMOTE

#### Oversample: Parameter: oversample.ratio
task <- TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y") 

## test for best matches with smote function's parameter with SVM learner
svm_lrn <- lrn("classif.svm", predict_type = "prob")

po_smote <- po("smote", dup_size = 50)

# create smote learner fixed in svm learner
svm_smote_lrn <- GraphLearner$new(po_smote %>>% svm_lrn, predict_type = "prob")

# train with smote function's 2 parameters
svm_param_smote <- ParamSet$new(params = list( ParamInt$new("smote.dup_size", lower = 10, upper = 60), 
                                           ParamInt$new("smote.K", lower = 10, upper = 25)
))
# inner resampling set
inner_rsmp <- rsmp("cv", folds = 3L) 


# create Autolearner
svm_auto_smote <- AutoTuner$new(
  learner = svm_smote_lrn, resampling = inner_rsmp,
  measure = msr("classif.auc"), search_space = svm_param_smote,
  terminator = trm("none"), tuner = tnr("grid_search", resolution = 5)
)

# set outer_resampling, and create a design with it
outer_rsmp <- rsmp("cv", folds = 3L) 

design_smote = benchmark_grid(
  tasks = task,
  learners = svm_auto_smote, resamplings = outer_rsmp
)

set.seed(2020)

# run the benchmark, save the results afterwards
# knn_smote_bmr <- run_benchmark(design_smote)
svm_smote_bmr <- benchmark(design_smote, store_models = TRUE) # Runtime: 5 h

svm_smote_path1 <- svm_smote_bmr$data$learners()[[2]][[1]]$model$tuning_instance$archive$data()
svm_smote_path2 <- svm_smote_bmr$data$learners()[[2]][[2]]$model$tuning_instance$archive$data()
svm_smote_path3 <- svm_smote_bmr$data$learners()[[2]][[3]]$model$tuning_instance$archive$data()

df_1 <- svm_smote_path1[order(svm_smote_path1$smote.dup_size, svm_smote_path1$smote.K)]
df_2 <- svm_smote_path2[order(svm_smote_path2$smote.dup_size, svm_smote_path2$smote.K)]
df_3 <- svm_smote_path3[order(svm_smote_path3$smote.dup_size, svm_smote_path3$smote.K)]


# Calculate average level of AUC
svm_df_2 <- data_frame(
  smote.dup_size = df_1$smote.dup_size,
  smote.K = df_1$smote.K,
  average_classif.auc = c(mean(c(df_1$classif.auc[1],df_2$classif.auc[1],df_3$classif.auc[1])),
                          mean(c(df_1$classif.auc[2],df_2$classif.auc[2],df_3$classif.auc[2])),
                          mean(c(df_1$classif.auc[3],df_2$classif.auc[3],df_3$classif.auc[3])),
                          mean(c(df_1$classif.auc[4],df_2$classif.auc[4],df_3$classif.auc[4])),
                          mean(c(df_1$classif.auc[5],df_2$classif.auc[5],df_3$classif.auc[5])),
                          mean(c(df_1$classif.auc[6],df_2$classif.auc[6],df_3$classif.auc[6])),
                          mean(c(df_1$classif.auc[7],df_2$classif.auc[7],df_3$classif.auc[7])),
                          mean(c(df_1$classif.auc[8],df_2$classif.auc[8],df_3$classif.auc[8])),
                          mean(c(df_1$classif.auc[9],df_2$classif.auc[9],df_3$classif.auc[9])),
                          mean(c(df_1$classif.auc[10],df_2$classif.auc[10],df_3$classif.auc[10])),
                          mean(c(df_1$classif.auc[11],df_2$classif.auc[11],df_3$classif.auc[11])),
                          mean(c(df_1$classif.auc[12],df_2$classif.auc[12],df_3$classif.auc[12])),
                          mean(c(df_1$classif.auc[13],df_2$classif.auc[13],df_3$classif.auc[13])),
                          mean(c(df_1$classif.auc[14],df_2$classif.auc[14],df_3$classif.auc[14])),
                          mean(c(df_1$classif.auc[15],df_2$classif.auc[15],df_3$classif.auc[15])),
                          mean(c(df_1$classif.auc[16],df_2$classif.auc[16],df_3$classif.auc[16])),
                          mean(c(df_1$classif.auc[17],df_2$classif.auc[17],df_3$classif.auc[17])),
                          mean(c(df_1$classif.auc[18],df_2$classif.auc[18],df_3$classif.auc[18])),
                          mean(c(df_1$classif.auc[19],df_2$classif.auc[19],df_3$classif.auc[19])),
                          mean(c(df_1$classif.auc[20],df_2$classif.auc[20],df_3$classif.auc[20])),
                          mean(c(df_1$classif.auc[21],df_2$classif.auc[21],df_3$classif.auc[21])),
                          mean(c(df_1$classif.auc[22],df_2$classif.auc[22],df_3$classif.auc[22])),
                          mean(c(df_1$classif.auc[23],df_2$classif.auc[23],df_3$classif.auc[23])),
                          mean(c(df_1$classif.auc[24],df_2$classif.auc[24],df_3$classif.auc[24])),
                          mean(c(df_1$classif.auc[25],df_2$classif.auc[25],df_3$classif.auc[25]))
                          
  )
)



# Return plot with size 5*10
svm_smote_gg2 <- ggplot(svm_df_2, aes(
  x = smote.dup_size,
  y = average_classif.auc, col = factor(smote.K))) + geom_point(size = 3) +
  geom_line()+
  geom_text(aes(label = round(average_classif.auc, 4)),
            vjust = "inward", hjust = "inward",
            show.legend = FALSE)+
  theme(legend.position = "bottom")

svm_smote_gg2


####################################################################
##########            Step 5.4: xgboost              ##############
####################################################################

#------------------------------------------------------------------------------------------
#### Oversample








####################################################################
##########         Step 5.5: Random Forest            ##############
####################################################################

#------------------------------------------------------------------------------------------
#### Oversample









