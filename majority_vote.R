prob1 = 1 - read.csv("Submissions/xgboost_prob.csv")$xgboostTestPred
threshold1 = 1-.662
#AUC for xgboost = .9254
prob2 = read.csv("Submissions/gbm_prob.csv")$s
threshold2 = .002
#AUC for gbm = .855
submission3 = read.csv("")
threshold2 = 

ensembled.prob = (as.numeric(submission1$Class == 's') + 
                   as.numeric(submission2$Class == 's') +
                   as.numeric(submission3$Class == 's'))/3
threshold = (threshold1 + threshold2 + threshold3)/3

final.prediction = ifelse(ensembled.prob < threshold,
                         's',
                         'b')

EventId = read.csv("Submissions/EventID.csv")$higgs.testId
weightRank = rank(ensembled.prob, ties.method= "random")
  
submission = data.frame(EventId = EventId, RankOrder = weightRank, Class = final.prediction)
write.csv(submission, "Submissions/ensembled_submission.csv", row.names=FALSE)