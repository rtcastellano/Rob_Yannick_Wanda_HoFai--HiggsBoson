library(xgboost)
library(methods)
library(pROC)
library(caret)
library(xgboost)
library(readr)
library(dplyr)
library(tidyr)
library(dummy)
library(doMC)
registerDoMC(cores = 4)

higgs.train.full = read.csv('./data/training.csv', header=T)
higgs.test.full = read.csv('./data/test.csv', header=T)
higgs.testId = higgs.test.full$EventId

higgs.testId = higgs.test$EventId

higgs.train = higgs.train.full
higgs.test = higgs.test.full
higgs.train$PRI_jet_num <- as.factor(higgs.train$PRI_jet_num)
higgs.test$PRI_jet_num <- as.factor(higgs.test$PRI_jet_num)
str(higgs.test)

higgs.weight <- higgs.train$Weight
higgs.labels <- make.names(as.factor(as.numeric(higgs.train$Label == 's')))

scaled.weight = higgs.weight * nrow(higgs.test)/length(higgs.labels)

higgs.train = higgs.train[, -c(1,32,33)]
higgs.test <- higgs.test[,-1]

#higgs.train[higgs.train==-999.0] <- NA
#higgs.test[higgs.test==-999.0] <- NA

data.mat <- as.matrix(higgs.train)

sumwpos <- sum(scaled.weight * (higgs.labels=='X0'))
sumwneg <- sum(scaled.weight * (higgs.labels=='X1'))

higgs.train.dummy = dummy.data.frame(higgs.train, names = "PRI_jet_num")

# set up the cross-validated hyper-parameter search
xgb_grid_1 = expand.grid(
  #eta = c(1, .1, 0.01),
  #max_depth = c(2, 4, 6, 8, 10),
  eta = 1,
  max_depth = 4,
  nrounds = 5,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1
)

# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                         # save losses across all models
  classProbs = TRUE,                            # set to TRUE for AUC to be computed
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# train the model for each parameter combination in the grid, 
#   using CV to evaluate
xgb_train_1 = train(
  x = higgs.train.dummy,
  y = higgs.labels,
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree",
  na.action = na.pass,
  missing = NA,
  metric = "ROC",
  weights = scaled.weight
)


# scatter plot of the AUC against max_depth and eta
ggplot(xgb_train_1$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) + 
  geom_point() + 
  theme_bw() + 
  scale_size_continuous(guide = "none")

xgb_train_1
xgb_train_1$bestTune
xgb_train_1$finalModel







#######################################
#Fitting best model
##################################
xgb_train_1$bestTune
xgmat.train <- xgb.DMatrix(as.matrix(higgs.train.dummy), 
                           label = as.numeric(higgs.labels == "X0"),
                           weight = scaled.weight)
nround = 5
param <- list("objective" = "binary:logistic",
              "bst:max_depth" = 4,
              "bst:eta" = 1,
              "eval_metric" = "auc",
              "eval_metric" = "ams@0.15",
              "silent" = 1,
              "nthread" = 8)
watchlist <- list("train" = xgmat.train)
bst = xgboost(xgmat.train, nrounds = 5, params = param)


predict.train = predict(bst, xgmat.train)
auc = roc(higgs.labels, predict.train)
plot(auc, print.thres=TRUE)
auc$auc
threshold = .996

err <- mean(as.numeric(predict.train >= threshold) != (higgs.labels == "X0"))

###################################################################

xgmat.test <- xgb.DMatrix(as.matrix(dummy.data.frame(higgs.test, names = "PRI_jet_num")))


predict <- predict(bst, xgmat.test)

importance_matrix = xgb.importance(feature_names = colnames(data), model = bst)
xgb.plot.importance(importance_matrix)



