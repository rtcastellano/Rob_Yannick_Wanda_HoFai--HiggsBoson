library(xgboost)
library(methods)
library(pROC)
library(caret)
library(xgboost)
library(readr)
library(plyr)
library(dplyr)
library(tidyr)
library(dummies)
library(doMC)
registerDoMC(cores = 4)

#Read in the data
#higgs.___.full is raw data
higgs.train.full = read.csv('./data/training.csv', header=T)
higgs.test.full = read.csv('./data/test.csv', header=T)
higgs.testId = higgs.test.full$EventId

#############################################
########### DATA MUNGING ###################
##########################################
#higgs.__ will be what is analyzed
higgs.train = higgs.train.full
higgs.test = higgs.test.full

#Tranform PRI_jet_num into a factor, as instructed
higgs.train$PRI_jet_num <- as.factor(higgs.train$PRI_jet_num)
higgs.test$PRI_jet_num <- as.factor(higgs.test$PRI_jet_num)

#higgs.weight is the weight of the training data
higgs.weight <- higgs.train$Weight

#We make labels of the outcomes.
#The make.names is because the "train" function requires the factors to have names that are valid
# variable names (unlike 0,1 or True, False)
higgs.labels <- make.names(as.factor(as.numeric(higgs.train$Label == 's')))

#Scale the weight according to the length of the data.
scaled.weight = higgs.weight * nrow(higgs.test)/length(higgs.labels)

#Remove the ID, Weight, and Outcome columns
higgs.train = higgs.train[, -c(1,32,33)]
higgs.test <- higgs.test[,-1]

#Weighted sum of the signal/background data
sumwpos <- sum(scaled.weight * (higgs.labels=='X0'))
sumwneg <- sum(scaled.weight * (higgs.labels=='X1'))

#Create a dummy variable for the "PRI_jet_num" variable
higgs.train.dummy = dummy.data.frame(higgs.train, names = "PRI_jet_num")

#############################################
##############################################
# Grid for the parameter search
#The guidlines for how to tune parameters are commented below and are taken from
# Owen Zheng http://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions
xgb_grid_1 = expand.grid(
  eta = c(.5, 1, 1.5),                #[2-10]/num trees
  max_depth = c(4, 6, 8),             #Start with 6
  nrounds = 100,                      #Fix at 100
  gamma = 0,                          #Usually ok to leave at 0
  colsample_bytree = c(.3, .5, .7),   #.3 - .5
  min_child_weight = 1                #start with 1/sqrt(eventrate)
)

# Tuning control parameters
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

# Train the model on each set of parameters in the grid and evaluate using cross-validation
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

xgb_grid_2 = expand.grid(
  eta = c(.4, .5, .6),                #[2-10]/num trees
  max_depth = c(3, 4, 5),             #Start with 6
  nrounds = 100,                      #Fix at 100
  gamma = 0,                          #Usually ok to leave at 0
  colsample_bytree = c(.6, .7, .8, .9),   #.3 - .5
  min_child_weight = 1                #start with 1/sqrt(eventrate)
)

xgb_train_2 = train(
  x = higgs.train.dummy,
  y = higgs.labels,
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_2,
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
#######################################
xgb_train_1
xgb_train_1$bestTune
xgb_train_1$finalModel





###################################################
###################################################


xgmat.train <- xgb.DMatrix(as.matrix(higgs.train.dummy), 
                           label = as.numeric(higgs.labels == "X0"),
                           weight = scaled.weight)
bst = xgb_train_1$finalModel

predict.train = predict(bst, xgmat.train)
auc = roc(higgs.labels, predict.train)
plot(auc, print.thres=TRUE)
auc$auc
threshold = .996

err <- mean(as.numeric(predict.train >= threshold) != (higgs.labels == "X0"))

importance_matrix = xgb.importance(feature_names = colnames(higgs.train.dummy), model = bst)
xgb.plot.importance(importance_matrix)

###################################################################

xgmat.test <- xgb.DMatrix(as.matrix(dummy.data.frame(higgs.test, names = "PRI_jet_num")))
predict <- predict(bst, xgmat.test)



