submission1 = read.csv("Submissions/xgboost_submission.csv")
submission2 = read.csv("Submissions/08/gbm_submission.csv")
submission3 = read.csv("")

ifelse((as.numeric(submission1$Class == 's') + 
          as.numeric(submission2$Class == 's') +
          as.numeric(submission3$Class == 's')/3 > .5,
  's',
  'b')

