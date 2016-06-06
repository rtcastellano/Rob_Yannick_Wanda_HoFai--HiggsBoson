library(caret)
library(moments)
library(dplyr)
library(car)
library(pscl)
library(ROCR)

setwd("~/Desktop/kaggle_jumpstart")

dfTrain <- read.csv('./data/training.csv', header=T)
dfTrain[dfTrain==-999.0] <- NA  
dfTrain$PRI_jet_num <- as.factor(dfTrain$PRI_jet_num)
train2 <- dfTrain[, -c(1,32)] # keeping Label

# #log graphs
#   g <- ggplot(data = dfTrain, aes(x = log(PRI_tau_pt)))
#   g+geom_density(aes(fill=factor(Label)), alpha = 0.7) #+xlim(0.0001, 400)
#  # 
#   g <- ggplot(data = dfTrain, aes(x = PRI_tau_pt))
#   g+geom_density(aes(fill=factor(Label)), alpha = 0.7)#+xlim(0.0001, 400)
 # + 1e-8 is for when there's zeros. 

set.seed(0)
Train <- createDataPartition(train2$Label, p=0.8, list=FALSE) #80/20 split
training <- train2[Train, ] #200001 obs.
ttesting <- train2[-Train, ] #49999 obs. #validation set 
ttesting[is.na(ttesting)] <- 0
#rename ttesting column names to match log ones. 
ttesting <- ttesting %>% rename(., PRI_jet_subleading_pt_log =PRI_jet_subleading_pt, DER_deltaeta_jet_jet_log = DER_deltaeta_jet_jet,
   DER_mass_jet_jet_log = DER_mass_jet_jet, DER_sum_pt_log = DER_sum_pt, PRI_met_sumet_log = PRI_met_sumet, 
   DER_pt_ratio_lep_tau_log = DER_pt_ratio_lep_tau, DER_mass_vis_log = DER_mass_vis, DER_mass_transverse_met_lep_log = DER_mass_transverse_met_lep,
   PRI_lep_pt_log=PRI_lep_pt, DER_pt_tot_log=DER_pt_tot, PRI_met_log=PRI_met, PRI_jet_leading_pt_log=PRI_jet_leading_pt,
   DER_pt_h_log = DER_pt_h, PRI_tau_pt_log = PRI_tau_pt)

#Adding transformed variables. keep NA's at first because log(0) = -Inf
trans.training <- training %>% mutate(., PRI_jet_subleading_pt_log = log(training$PRI_jet_subleading_pt + 1e-8),
                                      DER_deltaeta_jet_jet_log = log(training$DER_deltaeta_jet_jet+ 1e-8), DER_mass_jet_jet_log = log(training$DER_mass_jet_jet+ 1e-8),
                                      DER_sum_pt_log = log(training$DER_sum_pt+ 1e-8), PRI_met_sumet_log=log(training$PRI_met_sumet+ 1e-8), 
                                      DER_pt_ratio_lep_tau_log = log(training$DER_pt_ratio_lep_tau+ 1e-8),
                                      DER_mass_vis_log = log(training$DER_mass_vis+ 1e-8), DER_mass_transverse_met_lep_log = log(training$DER_mass_transverse_met_lep+ 1e-8),
                                      PRI_lep_pt_log = log(training$PRI_lep_pt+ 1e-8), DER_pt_tot_log = log(training$DER_pt_tot+ 1e-8),
                                      PRI_met_log = log(training$PRI_met+ 1e-8), PRI_jet_leading_pt_log = log(training$PRI_jet_leading_pt+ 1e-8),
                                      DER_pt_h_log = log(training$DER_pt_h+ 1e-8), PRI_tau_pt_log = log(training$PRI_tau_pt+ 1e-8))

#training set with transformed variables #200001 obs.
transf.training <- dplyr::select(trans.training, DER_mass_MMC, DER_prodeta_jet_jet, DER_deltar_tau_lep, DER_met_phi_centrality, 
                          DER_lep_eta_centrality,PRI_tau_eta,PRI_tau_phi,PRI_lep_eta, PRI_lep_phi,PRI_met_phi,
                          PRI_jet_num,PRI_jet_leading_eta, PRI_jet_leading_phi,
                          PRI_jet_subleading_eta, PRI_jet_subleading_phi,PRI_jet_all_pt,Label,PRI_jet_subleading_pt_log,DER_deltaeta_jet_jet_log,
                          DER_mass_jet_jet_log,DER_sum_pt_log,PRI_met_sumet_log,DER_pt_ratio_lep_tau_log,DER_mass_vis_log,DER_mass_transverse_met_lep_log,
                          PRI_lep_pt_log,DER_pt_tot_log,PRI_met_log,PRI_jet_leading_pt_log,DER_pt_h_log,PRI_tau_pt_log)
#length(is.na(transf.training))
#[1] 6200031

transf.training[is.na(transf.training)] <- 0 
length(is.na(transf.training)) #is it counting the zeros

#Saturated model
saturated.model = glm(formula = Label ~., family = "binomial", data = transf.training) 
summary(saturated.model) 

#Not statistically significant: 12 PRI, 1 DER
# PRI_tau_eta , PRI_tau_phi, PRI_lep_eta, PRI_lep_phi, PRI_met_phi, PRI_jet_leading_eta ,PRI_jet_leading_phi, 
# PRI_jet_subleading_eta, PRI_jet_subleading_phi, DER_sum_pt_log,PRI_met_sumet_log,DER_pt_ratio_lep_tau_log,
# PRI_lep_pt_log, PRI_tau_pt_log 
# significant variables include mostly DER variables

# Call: 06/04... AIC: 205175
#   glm(formula = Label ~ ., family = "binomial", data = transf.training)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -3.9454  -0.7830  -0.4712   0.9141   3.5716  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)                     -5.368e+00  1.615e-01 -33.234  < 2e-16 ***
#   DER_mass_MMC                    -1.574e-03  1.226e-04 -12.840  < 2e-16 ***
#   DER_prodeta_jet_jet             -1.007e-01  5.072e-03 -19.861  < 2e-16 ***
#   DER_deltar_tau_lep               1.238e+00  1.932e-02  64.091  < 2e-16 ***
#   DER_met_phi_centrality           2.784e-01  6.235e-03  44.652  < 2e-16 ***
#   DER_lep_eta_centrality           1.182e+00  3.236e-02  36.525  < 2e-16 ***
# PRI_jet_num1                    -4.130e+00  1.243e-01 -33.219  < 2e-16 ***
#   PRI_jet_num2                    -6.340e+00  2.174e-01 -29.165  < 2e-16 ***
#   PRI_jet_num3                    -6.696e+00  2.082e-01 -32.158  < 2e-16 ***
# PRI_jet_all_pt                  -7.408e-03  2.598e-04 -28.511  < 2e-16 ***
#   PRI_jet_subleading_pt_log        4.762e-01  3.958e-02  12.032  < 2e-16 ***
#   DER_deltaeta_jet_jet_log        -1.571e-01  1.459e-02 -10.765  < 2e-16 ***
#   DER_mass_jet_jet_log             9.279e-02  2.476e-02   3.747 0.000179 ***
# DER_mass_vis_log                -1.604e+00  3.746e-02 -42.810  < 2e-16 ***
#   DER_mass_transverse_met_lep_log -3.768e-01  5.647e-03 -66.723  < 2e-16 ***
# DER_pt_tot_log                  -5.418e-02  5.200e-03 -10.419  < 2e-16 ***
#   PRI_met_log                     -1.588e-01  8.662e-03 -18.330  < 2e-16 ***
#   PRI_jet_leading_pt_log           1.183e+00  3.917e-02  30.209  < 2e-16 ***
#   DER_pt_h_log                     6.647e-02  7.959e-03   8.352  < 2e-16 ***
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 257117  on 200000  degrees of freedom
# Residual deviance: 205109  on 199968  degrees of freedom
# AIC: 205175

#8 primary are insignificant; PRI_tau_eta,PRI_tau_phi,PRI_lep_eta,PRI_lep_phi,PRI_jet_leading_eta,
#PRI_jet_subleading_eta,PRI_jet_subleading_phi, PRI_tau_pt_log
anova(saturated.model, test="Chisq")

#varImp graphs
impVars <- varImp(saturated.model)
varnames <- rownames(impVars)
varOrders<-data.frame(varnames=varnames,impVars)
varOrders<-arrange(varOrders,Overall)[1:20,]
c<- ggplot(varOrders, aes(x=reorder(varnames,Overall),y=Overall,fill=varnames)) 
c+geom_bar(stat = 'identity') + coord_flip() + xlab('Feature Name')+ theme(legend.position="none")  

impVarz <- varImp(BIC.model)
varnamez <- rownames(impVarz)
varOrderz<-data.frame(varnames=varnamez,impVarz)
varOrderz<-arrange(varOrderz,Overall)[1:20,]
c<- ggplot(varOrderz, aes(x=reorder(varnamez,Overall),y=Overall,fill=varnamez)) 
c+geom_bar(stat = 'identity') + coord_flip() + xlab('Feature Name')+ theme(legend.position="none") 

#bothBIC.full

# absolute value of the t-statistic for each model parameter..
#determines if it’s significantly different from zero.

#summary(BIC.model)
# Null deviance: 257117  on 200000  degrees of freedom
# Residual deviance: 205118  on 199980  degrees of freedom
# AIC: 205160

#1-pchisq(205087,199965) #p-value  p-value of approximately zero showing that there is a significant lack of evidence to support the null hypothesis.
#[1] 4.440892e-16 

# > saturated.model$df.residual
# [1] 199968
# > saturated.model$df.null
# [1] 200000
# > saturated.model$null.deviance
# [1] 257117.1
# > saturated.model$aic
# [1] 205174.6
# > BIC.model$aic
# [1] 205160.1

saturated.model$df.residual #[1] 199968
# Null deviance: 257117  on 200000  degrees of freedom
# Residual deviance: 205109  on 199968  degrees of freedom
# AIC: 205175

#model with no variables
model.empty = glm(formula = Label ~ 1, family = "binomial", data = transf.training) #AIC: 257119
scope = list(lower = formula(model.empty), upper = formula(saturated.model))
n=nrow(transf.training)
#Stepwise Regression k = log(n) is BIC. refer to formula.
bothBIC.full = step(saturated.model, scope, direction = "both", k = log(n)) 
#takes 10 minutes to run. run again later. save workspace

BIC(bothBIC.full) #[1] 247909.6 why is this number different from the smallest one from bothBIC.full? 205486.7
#BIC penalizes a model depending on how complex it is
# Step:  BIC=205486.7

BIC.model = glm(formula = Label ~ DER_mass_MMC + DER_prodeta_jet_jet + DER_deltar_tau_lep + 
                    DER_met_phi_centrality + DER_lep_eta_centrality + PRI_jet_num +
                    PRI_jet_all_pt + PRI_jet_subleading_pt_log + DER_deltaeta_jet_jet_log +
                    DER_mass_jet_jet_log + DER_pt_ratio_lep_tau_log + DER_mass_vis_log +
                    DER_mass_transverse_met_lep_log + DER_pt_tot_log + PRI_met_log +
                    PRI_jet_leading_pt_log + DER_pt_h_log + PRI_tau_pt_log, family = "binomial", data = transf.training)
summary(BIC.model)
# all have signifcant p values. BIC: 205160

# Evaluate Collinearity
vif(BIC.model) # GVIF very high for PRI_jet_num and all jet variables - which indicates it is influencing other variables as expected
collinear<- as.data.frame(vif(BIC.model)[,1])

vif(saturated.model) 

#new model
#curvy plot 

scatter.smooth(saturated.model$fit,
               residuals(saturated.model, type = "deviance"),
               lpars = list(col = "red"),
               xlab = "Fitted Probabilities",
               ylab = "Deviance Residual Values",
               main = "Residual Plot for\nLogistic Regression of Data: Saturated")
abline(h = 0, lty = 2, col="grey") 

scatter.smooth(BIC.model$fit,
               residuals(BIC.model, type = "deviance"),
               lpars = list(col = "red"),
               xlab = "Fitted Probabilities",
               ylab = "Deviance Residual Values",
               main = "Residual Plot for\nLogistic Regression: BIC")
abline(h = 0, lty = 2) 
#saturated.model.clean
#influencePlot(saturated.model.clean)

influencePlot(BIC.model) #different numbers
# StudRes          Hat        CookD
# 32646 -1.872680 1.051175e-02 0.0023692260
# 47385 -1.623547 1.259978e-02 0.0016447192
# 84758 -3.943281 6.811860e-06 0.0007654224

influencePlot(saturated.model) # There are some high leverage points but luckily they have low residuals 
# StudRes          Hat       CookD
# 32646 -1.873092 1.055037e-02 0.001514562
# 47385 -1.643628 1.243064e-02 0.001078767
# 84758 -3.947462 6.831570e-06 0.000496547

View(transf.training[84750:84758,])
transf.training.clean = transf.training[-c(32646, 47385, 84758),] 
#deltaeta jet jet..32646,-18.42/// 47385 -18.4
#der mass transverse met lep log ...84758,-18.42

exp(saturated.model$coefficients)

#cbind("Log Odds" = model.full$coefficients,
#      "Odds" = exp(model.full$coefficients))
#confint(model.full)

pchisq(saturated.model$deviance, saturated.model$df.residual, lower.tail = FALSE)
#[1] very close to 0 ??
#3.779272e-16 june 4
pchisq(BIC.model$deviance, BIC.model$df.residual, lower.tail = FALSE)
#[1]  very close to 0 
#3.908636e-16 june 4
#1-pchisq(BIC.model$deviance, BIC.model$df.residual, lower.tail = FALSE) = 1

pred = predict(saturated.model, newdata=ttesting) # have to update ttesting column names above. 
accuracy <- table(pred, ttesting[,"Label"])
sum(diag(accuracy))/sum(accuracy) #[1] 4.00008e-05 is so low
#confusionMatrix(data=pred, ttesting$Label) doesn't work

#http://stats.stackexchange.com/questions/82105/mcfaddens-pseudo-r2-interpretation
#null deviance is the same between the 2
saturated.model$deviance-BIC.model$deviance #[1] -6.073273
# june 4 -9.506332
pR2(saturated.model)
# june 4 ... 0.2022753
#[1] 0.2018732...scale of 0 to 1, 0 indicating no predictive power

#pR2(saturated.model.clean) 
#june 4... 0.20235... very slightly better fit than the dirty saturated model. 

pR2(BIC.model)
#june 4 .... 0.2022383
#[1] 0.2018496.... worser fit than saturated

# Alternative way to find R^2
# Rsqaured.saturated.clean = 1 - saturated.model.clean$deviance/saturated.model.clean$null.deviance
# june 4 .... [1] 0.2023526
# Rsquared.saturated= 1 - saturated.model$deviance/saturated.model$null.deviance #[1]  0.2022753
# Rsquared.BIC= 1 - BIC.model$deviance/BIC.model$null.deviance #[1] 0.2018496..worse fit than the saturated

#Accuracy Testing
# Compute AUC for predicting Class with the model
#ttesting to be run from earlier. 
prob <- predict(saturated.model, newdata=ttesting, type="response")
pred <- prediction(prob, ttesting$Label)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc
#Area under the curve: 0.66
#http://www.r-bloggers.com/how-to-perform-a-logistic-regression-in-r/


# Compute AUC for predicting Class with the model
prob <- predict(BIC.model, newdata=ttesting, type="response")
pred <- prediction(prob, ttesting$Label)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc
#Saturated model: 
# 80%: 0.799
# 20%: Area under the curve: [1] 0.6634277//////[1] 0.6557786
# BIC model: 
#80%: 
#20% [1] 0.6638395 #not much different ///////[1] 0.663007

#prediction?
label.predicted = round(saturated.model$fitted.values) #?
label.predicted.BIC = round(BIC.model$fitted.values)
#sum(diag(accuracy))/sum(accuracy) #[1] 0??
#diag(accuracy) #0?
#confusionMatrix(data=pred, ttesting$Label)

#KEY TAKEAWAYS#
#Data Pre-processing
#Missingness Patterns
#Positively skewed variables were log transformed
# Logistic Regression 
# Collinearity with Jet variables (VIF)
# Top Variables - Mass
# BIC stepwise regression reduced the number of PRI variables, 
#but is not much a better fit than Saturated model
# AUC not much different, pschi not much different
# try Ridge Regression instead

#17 variables are positively skewed...mean>median..If skewness = 0, the data are perfectly symmetrical.
#If skewness is less than −1 or greater than +1, the distribution is highly skewed.
# qplot(training$PRI_jet_subleading_pt, geom = 'histogram', binwidth = .2) # positively skewed
# qplot(training$DER_deltaeta_jet_jet, geom = 'histogram', binwidth = .2) skewed
# qplot(training$DER_mass_jet_jet, geom = 'histogram', binwidth = .2) skewed
# qplot(training$DER_sum_pt, geom = 'histogram', binwidth = .2) skewed #2.3
# qplot(training$PRI_met_sumet, geom = 'histogram', binwidth = .2) skewed #1.85
# qplot(training$DER_pt_ratio_lep_tau, geom = 'histogram', binwidth = .2) skewed #2.7
# qplot(training$DER_mass_vis, geom = 'histogram', binwidth = .2) skewed #3.75
# qplot(training$DER_mass_transverse_met_lep, geom = 'histogram', binwidth = .2) skewed #1.2
# qplot(training$PRI_lep_pt, geom = 'histogram', binwidth = .2) skewed
# qplot(training$DER_pt_tot, geom = 'histogram', binwidth = .2) skewed
# qplot(training$PRI_met, geom = 'histogram', binwidth = .2) skewed
# qplot(training$PRI_jet_leading_pt, geom= 'histogram', binwidth = .2)  skewed
# qplot(training$DER_pt_h, geom= 'histogram', binwidth = .2) skewed
# qplot(training$PRI_tau_pt, geom= 'histogram', binwidth = .2) skewed
# qplot(training$DER_mass_MMC, geom= 'histogram', binwidth = .2) skewed ### i don't want to touch this?
# qplot(training$DER_met_phi_centrality, geom= 'histogram', binwidth = .2) #wierd u shape
# library(moments)
# 

reduced.deviance = BIC.model$deviance #Comparing the deviance of the reduced
reduced.df = BIC.model$df.residual    #model (the one without rank) to...
###
#reduced.deviance.clean = saturated.model.clean$deviance #[1] 205086.8
#reduced.df.clean = saturated.model.clean$df.residual #[1] 199965
###
full.deviance = saturated.model$deviance #...the deviance of the full model (the
full.df = saturated.model$df.residual    #one with the rank terms).

pchisq(reduced.deviance - full.deviance,
       reduced.df - full.df,
       lower.tail = FALSE)
###[1] 0.6591819

#pchisq(reduced.deviance.clean - full.deviance, #-21.87
#       reduced.df.clean - full.df, #-3
 #      lower.tail = FALSE)
### giving me NaNs produced
anova(saturated.model, BIC.model, test = "Chisq")
# either model is fine
# june 4 0.659
anova(saturated.model, saturated.model.clean, test = "Chisq") #not the same size 

pchisq(saturated.model$deviance, saturated.model$df.residual, lower.tail = FALSE)
#The p-value for the overall test of deviance is <.05, indicating that this model
#is not a good overall fit!

