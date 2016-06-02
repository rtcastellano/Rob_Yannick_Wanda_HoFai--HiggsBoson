#EDA 
#Variable Importance
#Amy's sheet 
#Logistic Regression, AIC 
#Missingness 

dfTrain <- read.csv('./data/training.csv', header=T)
dfTrain[dfTrain==-999.0] <- 0  
weight <- dfTrain$Weight
labels <- dfTrain$Label
dfTrain$PRI_jet_num <- as.factor(dfTrain$PRI_jet_num)
train2 <- dfTrain[, -c(1,32)]
Train <- createDataPartition(train2$Label, p=0.8, list=FALSE) #80/20 split
training <- train2[ Train, ] #200001 obs.
ttesting <- train2[ -Train, ] #49999 obs.

#overall model
model.full = glm(formula = Label ~ ., family = "binomial", data = train2)
#AIC: 247718.4 

#model with no variables
model.empty = glm(formula = Label ~ 1, family = "binomial", data = train2)
summary(model.empty) #AIC: 321397
scope = list(lower = formula(model.empty), upper = formula(model.full))
#AIC and BIC as of June 2nd

forwardAIC = step(model.empty, scope, direction = "forward", k = 2)
backwardAIC = step(model.full, scope, direction = "backward", k = 2)
bothAIC.empty = step(model.empty, scope, direction = "both", k = 2)
bothAIC.full = step(model.full, scope, direction = "both", k = 2)

forwardBIC = step(model.empty, scope, direction = "forward", k = log(nrow(training)))
backwardBIC = step(model.full, scope, direction = "backward", k = log(nrow(training)))
bothBIC.empty = step(model.empty, scope, direction = "both", k = log(nrow(training)))
bothBIC.full = step(model.full, scope, direction = "both", k = log(nrow(training)))

AIC(forwardAIC, backwardAIC, bothAIC.empty, bothAIC.full)
BIC(forwardBIC, backwardBIC, bothBIC.empty, bothBIC.full)

summary(forwardAIC)
plot(forwardAIC)
avPlots(forwardAIC)
vif(forwardAIC)

AIC(forwardAIC, model.full)
BIC(forwardAIC, model.full)
#

#new model
model.empty.m = glm(formula = Label ~ DER_mass_MMC + PRI_jet_num + DER_mass_transverse_met_lep + DER_mass_vis + DER_pt_h, family = "binomial", data = train2)
summary(model.empty.m) #AIC: 276182

#curvy plot 
scatter.smooth(model.full$fit,
               residuals(model.full, type = "deviance"),
               lpars = list(col = "red"),
               xlab = "Fitted Probabilities",
               ylab = "Deviance Residual Values",
               main = "Residual Plot for\nLogistic Regression of Data")
abline(h = 0, lty = 2) #looks alot better

library(car)
influencePlot(model.full) #? bubble split
summary(model.full)

exp(model.full$coefficients)

cbind("Log Odds" = model.full$coefficients,
      "Odds" = exp(model.full$coefficients))

confint(model.full)

# Call: summary(model.full)
#   glm(formula = Label ~ ., family = "binomial", data = train2)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -4.5433  -0.7850  -0.4093   0.9044   5.0793  
#
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
#   (Intercept)                   -3.127e+00  4.135e-02 -75.619  < 2e-16 ***
#   DER_mass_MMC                 3.387e-03  1.563e-04  21.669  < 2e-16 ***
#   DER_mass_transverse_met_lep -1.883e-02  2.210e-04 -85.217  < 2e-16 ***
#   DER_mass_vis                -3.027e-02  3.774e-04 -80.186  < 2e-16 ***
#   DER_pt_h                     4.184e-03  2.933e-04  14.265  < 2e-16 ***
#   DER_deltaeta_jet_jet        -1.010e-01  1.401e-02  -7.209 5.64e-13 ***
#   DER_mass_jet_jet             2.187e-03  6.957e-05  31.433  < 2e-16 ***
#   DER_prodeta_jet_jet          1.872e-02  4.860e-03   3.851 0.000118 ***
#   DER_deltar_tau_lep           1.337e+00  1.514e-02  88.301  < 2e-16 ***
#   DER_pt_ratio_lep_tau          -9.197e-01  1.686e-02 -54.561  < 2e-16 ***
#   DER_met_phi_centrality       1.764e-01  5.541e-03  31.842  < 2e-16 ***
#   DER_lep_eta_centrality       1.003e+00  2.972e-02  33.748  < 2e-16 ***
#   PRI_met                        5.128e-03  3.042e-04  16.853  < 2e-16 ***
#   PRI_met_sumet                 -1.493e-03  1.011e-04 -14.773  < 2e-16 ***
#   PRI_jet_num1                 3.960e-01  1.621e-02  24.424  < 2e-16 ***
#   PRI_jet_num2                 2.499e-01  3.625e-02   6.895 5.40e-12 ***
#   PRI_jet_num3                -2.239e-01  4.627e-02  -4.840 1.30e-06 ***

#17 variables above  have good p values#
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# Null deviance: 321395  on 249999  degrees of freedom
# Residual deviance: 247652  on 249967  degrees of freedom
# AIC: 247718
# Number of Fisher Scoring iterations: 5

pchisq(model.full$deviance, model.full$df.residual, lower.tail = FALSE)
#[1] 0.999486  this is very high?

exp(model.full$coefficients)
varImp(model.full)

# > varImp(model.full) 12 variables
# Overall
# DER_mass_MMC                21.66877807
# DER_mass_transverse_met_lep 85.21689265
# DER_mass_vis                80.18645816
# DER_mass_jet_jet            31.43334481
# DER_deltar_tau_lep          88.30106449
# DER_pt_ratio_lep_tau        54.56148667
# DER_met_phi_centrality      31.84153497
# DER_lep_eta_centrality      33.74806260
# PRI_jet_num1                24.42359195
# PRI_met                     16.85300515
# DER_pt_h                    14.26482920
# PRI_met_sumet               14.77320070

pred = predict(model.full, newdata=ttesting)
accuracy <- table(pred, ttesting[,"Label"])

AIC(model.full) #[1] 247718.4 vs 321397(model.empty)
BIC(model.full) #[1] 248062.6
#select the model that has the smallest AIC.Only k = 2 gives the genuine AIC: k = log(n) is sometimes referred to as BIC or SBC.
#not very dramatic differences

backwards = step(model.full) # is default both?very long output, paste to document
summary(backwards)
formula(backwards)


AIC(backwards, forwards, model.full)#?

#forwardAIC
forwards = step(model.empty, scope = list(lower=formula(model.empty), 
       upper=formula(model.full)), direction="forward")
formula(forwards)

#Accuracy Testing
Rsquared	= 1 - model.full$deviance/model.full$null.deviance #[1] 0.2294445
#http://stats.stackexchange.com/questions/82105/mcfaddens-pseudo-r2-interpretation

#prediction?
label.predicted = round(model.full$fitted.values) #?

?step

#sum(diag(accuracy))/sum(accuracy) #[1] 0??
#diag(accuracy) #0?
#confusionMatrix(data=pred, ttesting$Label) #the data cannot have more levels than the reference
#Named num [1:49999]


#Logistic Ridge Regression glmnet


# Label ~ DER_mass_MMC + DER_mass_transverse_met_lep + DER_mass_vis + 
#   DER_pt_h + DER_deltaeta_jet_jet + DER_mass_jet_jet + DER_prodeta_jet_jet + 
#   DER_deltar_tau_lep + DER_pt_tot + DER_sum_pt + DER_pt_ratio_lep_tau + 
#   DER_met_phi_centrality + DER_lep_eta_centrality + PRI_tau_pt + 
#   PRI_tau_eta + PRI_tau_phi + PRI_lep_pt + PRI_lep_eta + PRI_lep_phi + 
#   PRI_met + PRI_met_phi + PRI_met_sumet + PRI_jet_num + PRI_jet_leading_pt + 
#   PRI_jet_leading_eta + PRI_jet_leading_phi + PRI_jet_subleading_pt + 
#   PRI_jet_subleading_eta + PRI_jet_subleading_phi + PRI_jet_all_pt
