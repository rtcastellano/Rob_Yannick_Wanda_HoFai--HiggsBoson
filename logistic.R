#http://www.r-bloggers.com/evaluating-logistic-regression-models/

library(caret)
dfTrain <- read.csv('./data/training.csv', header=T)
dfTrain[dfTrain==-999.0] <- 0  
weight <- dfTrain$Weight
labels <- dfTrain$Label #Factor w/ 2 levels "b","s"
dfTrain$PRI_jet_num <- as.factor(dfTrain$PRI_jet_num)
train2 <- dfTrain[, -c(1,32)] #keeping Label column as our Target variable 
#80% training/20% for validation split
Train <- createDataPartition(train2$Label, p=0.8, list=FALSE)
training <- train2[ Train, ] #200001 obs.
ttesting <- train2[ -Train, ] #49999 obs.
#overall fit
mod_fit <- train(Label ~ .,  data=training, method="glm", family="binomial")
exp(coef(mod_fit$finalModel))
varImp(mod_fit)  #vs. varImp(mod_fit_one)
#predict the value of the target variable on validation set 
predict(mod_fit, newdata=ttesting, type="prob")
#using glm screws the predict function up down below
mod_fit_one <- glm(Label ~ ., data=training, family="binomial") #Accuracy of 0.75
mod_fit_two <- glm(Label ~ DER_deltar_tau_lep + DER_mass_transverse_met_lep + DER_mass_vis + DER_pt_ratio_lep_tau +
DER_lep_eta_centrality + DER_met_phi_centrality + DER_mass_jet_jet + PRI_jet_num +  
  DER_mass_MMC, data=training, family="binomial")

library(pscl)
pR2(mod_fit_one) # McFadden 4.209024e-01

pred = predict(mod_fit, newdata=ttesting)
accuracy <- table(pred, ttesting[,"Label"])
sum(diag(accuracy))/sum(accuracy) #[1] mod_fit: 0.7528075 mod_fit_two: 2.00004e-05

confusionMatrix(data=pred, ttesting$Label) #Accuracy : 0.7507, with all variables.



