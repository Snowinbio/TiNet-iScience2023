# calculate the comprehensive prediction socre of nodule
data_explainable<-read.csv() # your data path
model_E=glm(formula=label~texture + edge + p_norm_echo + ratio + location, family=binomial(link="logit"),data=data)
summary(model_E)
anova(model_E)