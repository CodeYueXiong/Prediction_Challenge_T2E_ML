library(mlr3)
library(mlr3proba)
library(mlr3learners)
library(mlr3viz)
library(purrr)


tsks_train <- readRDS("Data-20210107/train_data.Rds")
tsks_test <- readRDS("Data-20210107/test_list_x.Rds")

eval_times = map(
  tsks_train_list,
  ~{
    time = .x$data()$time[.x$data()$status == 1]
    quantile(time, prob=.5)
  }
)


# evaluation
# msrs_q75 <- map(eval_times, ~msr("surv.graf", times = .x[3]))
msrs_q50 <- map(eval_times, ~msr("surv.graf", times = .x))
submission_files <- list.files("submissions", pattern = "*.Rds", full.names = TRUE)
submissions <- map(submission_files, readRDS)
names(submissions) <- tools::file_path_sans_ext(basename(submission_files))


## Error in UseMethod("as_task") : 
## no applicable method for 'as_task' applied to an object of class "data.frame"
# residual?
# ..1, ..2, ..3某个函数的第一个，第二个第三个参数
res <- imap(
  submissions,
  ~ pmap_dfr(
    list(.x, msrs_q50, tsks_test),
    ~{
      pred <- ..1$predict(task = ..3)
      pred$score(..2, task = ..3)
    },
    .id = "dataset"
  )
)

imap(
  res,
  ~write.csv(
    .x,
    file = paste0("submissions/", .y, ".csv"),
    row.names = FALSE,
    quote = FALSE
  )
)
