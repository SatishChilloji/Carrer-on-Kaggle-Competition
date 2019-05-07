rm(list = ls())
options("scipen"=100, "digits"=4)
library(data.table)
library(tidyverse)
library(DataExplorer)
library(corrplot)
library(ggplot2)
library(caret)
library(xgboost)
library(h2o)

fread("X_train.csv",na.strings = c(NA,''),stringsAsFactors = T)%>%data.frame()->c_train
fread("X_test.csv",na.strings = c(NA,''),stringsAsFactors = T)%>%data.frame()->c_test
fread("sample_submission.csv")%>%data.frame()->subm
fread("y_train.csv",na.strings = c(NA,''),stringsAsFactors = T)%>%data.frame()->c_train_tar

#quaternions to euler distance convesion.
quaternion_to_euler<-function(x,y,z,w){
  t0 <- +2.0 * (w * x + y * z)
  t1 <- +1.0 - 2.0 * (x * x + y * y)
  X <- atan2(t0, t1)
  
  t2 <- +2.0 * (w * y - z * x)
  t2 <-if(t2 > +1.0) {+1.0} else{t2}
  t2 <-if(t2 < -1.0) {-1.0} else{t2}
  Y = asin(t2)
  
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  Z = atan2(t3, t4)
  #dis<-c()
  return(c(X,Y,Z))
}

x<-c_train$orientation_X
y<-c_train$orientation_Y
z<-c_train$orientation_Z
w<-c_train$orientation_W

for (i in 1:length(x)) {
  quaternion_to_euler(x[i],y[i],z[i],w[i])[1]->c_train$euler_x[i]
  quaternion_to_euler(x[i],y[i],z[i],w[i])[2]->c_train$euler_y[i]
  quaternion_to_euler(x[i],y[i],z[i],w[i])[3]->c_train$euler_z[i]
  #print(paste("Running on",i,sep = " "))
  }
#write.csv(c_train,"c_train_euler.csv",row.names = F)
fread("c_train_euler.csv")%>%data.frame()->c_train

x<-c_test$orientation_X
y<-c_test$orientation_Y
z<-c_test$orientation_Z
w<-c_test$orientation_W


for (i in 1:length(x)) {
  quaternion_to_euler(x[i],y[i],z[i],w[i])[1]->c_test$euler_x[i]
  quaternion_to_euler(x[i],y[i],z[i],w[i])[2]->c_test$euler_y[i]
  quaternion_to_euler(x[i],y[i],z[i],w[i])[3]->c_test$euler_z[i]
  #print(paste("Running on",i,sep = " "))
  }
#write.csv(c_test,"c_test_euler.csv",row.names = F)
fread("c_test_euler.csv")%>%data.frame()->c_test

#Normalizing the Orientation.

normalize<-function(df_norm){
  df_norm$norm_quat<-(df_norm$orientation_X^2+df_norm$orientation_Y^2+df_norm$orientation_Z^2+df_norm$orientation_W^2)
  df_norm$mod_quat<-sqrt(df_norm$norm_quat)
  df_norm$norm_x<-df_norm$orientation_X/df_norm$mod_quat
  df_norm$norm_y<-df_norm$orientation_Y/df_norm$mod_quat
  df_norm$norm_z<-df_norm$orientation_Z/df_norm$mod_quat
  df_norm$norm_w<-df_norm$orientation_W/df_norm$mod_quat
  df_norm$tot_anglr_vel<-sqrt(df_norm$angular_velocity_X^2+df_norm$angular_velocity_Y^2+df_norm$angular_velocity_Z^2)
  df_norm$tot_lnr_accr<-sqrt(df_norm$linear_acceleration_X^2+df_norm$linear_acceleration_Y^2+df_norm$linear_acceleration_Z^2)
  df_norm$acc_vs_vel<-df_norm$tot_lnr_accr/df_norm$tot_anglr_vel
  return(df_norm)
}

normalize(c_train)->c_train
normalize(c_test)->c_test

range_mm<-function(x){max(x)-min(x)}
maxtomin<-function(x){max(x)/min(x)}
mean_abs_fn<-function(x){mean(abs(diff(x)))}
mean_change_abs_fn<-function(x){mean(diff(abs(diff(x))))}
abs_max<-function(x){max(abs(x))}
abs_min<-function(x){min(abs(x))}
abs_avg<-function(x){(abs_max(x)+abs_min(x))/2}

fn <- funs(mean,sum,median,max,min,sd,n_distinct,range_mm,maxtomin,mean_abs_fn,mean_change_abs_fn,abs_max,abs_min,abs_avg)

sum_c_train<-c_train%>%
  select(-row_id,-measurement_number)%>%
  group_by(series_id)%>%
  summarise_all(fn)%>%
  mutate_all(funs(ifelse(is.nan(.), 0, .)))%>% 
  mutate_all(funs(ifelse(is.infinite(.), 0, .)))%>%
  data.frame()

sum_c_test<-c_test%>%
  select(-row_id,-measurement_number)%>%
  group_by(series_id)%>%
  summarise_all(fn)%>%
  mutate_all(funs(ifelse(is.nan(.), 0, .)))%>% 
  mutate_all(funs(ifelse(is.infinite(.), 0, .)))%>%
  data.frame()


#corelation<-cor(sum_c_train[,-1],use = "everything")
#orrplot(corelation,method = "circle",type = "lower",title="Correlation Plot",
#         sig.level = 0.001,insig = "blank",bg="lightblue")
#rm(corelation);gc()
#apply(sum_c_test, 2, function(x){sum(is.infinite(x)/length(x)*100)})

#----------------------------------------------------------------------------------------------------------------------------
#Label encoding using the h2o package for getting the group id.
#h2o.init()
#h2o.no_progress()

#h2o.importFile("y_train.csv")->c_train_tar
#as.h2o(sum_c_train)->sum_c_train.h20
#h2o.merge(sum_c_train.h20,c_train_tar,by = "series_id",all.x = TRUE)->sum_c_train.h20

#Lets predict the group_id using the train data and we will use it for predicting surface.

#as.numeric(c_train_tar$surface)->c_train_tar$surface_num
#c_train_tar%>%select(surface,surface_num)%>%distinct()->mapping
#c_train_tar<-c_train_tar%>%mutate(surface_num=surface_num-1,surface=NULL,group_id=NULL)

#summary(c_train_tar)

sum_c_train%>%left_join(c_train_tar[,-3],by='series_id')->sum_c_train

rm(c_train,c_train_tar,c_test);gc()

#-------------------------------------------------------------------------------------------------


#Sampling.

train_index<-sample(1:nrow(sum_c_train),nrow(sum_c_train)*0.75)

#Full Dataset
data_variables<-as.matrix(sum_c_train[,-c(1,266)])
data_label<-sum_c_train[,"group_id"]

#Train and xgb data matrix.
train_data<-data_variables[train_index,]
train_label<-data_label[train_index]
train_matrix<-xgb.DMatrix(data = train_data,label=train_label)

#Test and xgb data matrix.

test_data<-data_variables[-train_index,]
test_label<-data_label[-train_index]
test_matrix<-xgb.DMatrix(data = test_data,label=test_label)

#Validation Daset

te_final<-data.matrix(sum_c_test[,-1])

#-------------------------------------------------------------------------------------------------------------------
#Parameter tuning using grid search.
searchGridSubCol <- expand.grid(subsample = seq(0.5,0.9,by=0.1), 
                                colsample_bytree = seq(0.5,0.9,by=0.1),
                                max_depth = seq(3,10,by=3),
                                min_child_weight = seq(3,10,by=3), 
                                eta = c(0.1)
)

ntrees<-20
cv.nfold  <- 5
numberOfClasses <- length(unique(sum_c_train$group_id))
system.time(
  mloglossErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
    
    #Extract Parameters to test
    currentSubsampleRate <- parameterList[["subsample"]]
    currentColsampleRate <- parameterList[["colsample_bytree"]]
    currentDepth <- parameterList[["max_depth"]]
    currentEta <- parameterList[["eta"]]
    currentMinChild <- parameterList[["min_child_weight"]]
    xgboostModelCV <-xgb.cv(data = train_matrix,
                            nrounds = ntrees,
                            nfold = cv.nfold,
                            showsd = TRUE,
                            verbose = TRUE,
                            "eval_metric" = "mlogloss",
                            "objective" = "multi:softprob",
                            "num_class" = numberOfClasses,
                            "max.depth" = currentDepth,
                            "eta" = currentEta,                               
                            "subsample" = currentSubsampleRate,
                            "colsample_bytree" = currentColsampleRate,
                            print_every_n = 5,
                            "min_child_weight" = currentMinChild,
                            booster = "gbtree",
                            early_stopping_rounds = 5)
    xvalidationScores <- as.data.frame(xgboostModelCV$evaluation_log)
    mlogloss <- tail(xvalidationScores$test_mlogloss_mean, 1)
    tmlogloss <- tail(xvalidationScores$train_mlogloss_mean,1)
    output <- return(c(mlogloss, tmlogloss, currentSubsampleRate, currentColsampleRate, currentDepth, currentEta, currentMinChild))}))  

output <- as.data.frame(t(mloglossErrorsHyperparameters))
varnames <- c("Testmlogloss", "Trainmlogloss", "SubSampRate", "ColSampRate", "Depth", "eta", "currentMinChild")
names(output) <- varnames
head(output)

output%>%arrange(Testmlogloss)%>%head()
#After tuning the parameters we select the one which are performing well. And we tune for number itertions

numberOfClasses <- length(unique(sum_c_train$group_id))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses,
                   eta=0.1,
                   gamma=0,
                   max_depth=9,
                   min_child_weight=3,
                   subsample=0.9,
                   booster = "gbtree",
                   colsample_bytree = 0.5)
nround    <- 2000 # number of XGBoost rounds
cv.nfold  <- 5

cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   early_stopping_rounds = 50,
                   print_every_n = 50,
                   showsd = T, 
                   stratified = T,
                   maximize = F)


model<-xgb.train(params = xgb_params,
                 data = train_matrix,
                 nrounds = 220,
                 watchlist = list(train=train_matrix,val=test_matrix),
                 print_every_n = 50,
                 early_stopping_rounds = 50,
                 maximize = F)


test_pred<-predict(model,newdata = test_matrix)
test_prediction<-matrix(test_pred,nrow = numberOfClasses,
                        ncol = length(test_pred)/numberOfClasses)%>%t()%>%
  data.frame()%>%mutate(label=test_label+1,
                        max_prop=max.col(.,"last"))

confusionMatrix(factor(test_prediction$max_prop),
                factor(test_prediction$label),
                mode = "everything")


#-----------------------------------------------------------------------------------------------
#Prediction on the Final Dataset for group ID.

val_pred<-predict(model,newdata = te_final)

vals_prediction<-matrix(val_pred,nrow = numberOfClasses,
                        ncol = length(val_pred)/numberOfClasses)%>%t()%>%
  data.frame()%>%mutate(group_id=max.col(.,"last"))

cbind(sum_c_test,data.frame("group_id"=vals_prediction[,74]))->sum_c_test

rm(test_pred,test_prediction,val_pred,vals_prediction);gc()
#-------------------------------------------------------------------------------------------------
#Now prediction on surface.


read.csv("y_train.csv",na.strings = c(NA,''),stringsAsFactors = T)->c_train_tar
as.numeric(c_train_tar$surface)->c_train_tar$surface_num
c_train_tar%>%select(surface,surface_num)%>%distinct()->mapping
c_train_tar<-c_train_tar%>%mutate(surface_num=surface_num-1)

sum_c_train%>%left_join(c_train_tar[,c(1,4)],by='series_id')->sum_c_train

rm(c_train,c_train_tar,c_test);gc()

#-------------------------------------------------------------------------------------------------

#Sampling.

train_index<-sample(1:nrow(sum_c_train),nrow(sum_c_train)*0.75)

#Full Dataset
data_variables<-as.matrix(sum_c_train[,-c(1,310)])
data_label<-sum_c_train[,"surface_num"]

#Train and xgb data matrix.
train_data<-data_variables[train_index,]
train_label<-data_label[train_index]
train_matrix<-xgb.DMatrix(data = train_data,label=train_label)

#Test and xgb data matrix.

test_data<-data_variables[-train_index,]
test_label<-data_label[-train_index]
test_matrix<-xgb.DMatrix(data = test_data,label=test_label)

#Validation Daset

te_final<-data.matrix(sum_c_test[,-1])


#--------------------------------------------------------------------------------------------------------

#Parameter tuning using grid search.
searchGridSubCol <- expand.grid(subsample = seq(0.5,0.9,by=0.1), 
                                colsample_bytree = seq(0.5,0.9,by=0.1),
                                max_depth = seq(3,10,by=3),
                                min_child_weight = seq(3,10,by=3), 
                                eta = c(0.1)
)

ntrees<-10
cv.nfold  <- 5
numberOfClasses <- length(unique(sum_c_train$surface_num))
system.time(
  mloglossErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
    
    #Extract Parameters to test
    currentSubsampleRate <- parameterList[["subsample"]]
    currentColsampleRate <- parameterList[["colsample_bytree"]]
    currentDepth <- parameterList[["max_depth"]]
    currentEta <- parameterList[["eta"]]
    currentMinChild <- parameterList[["min_child_weight"]]
    xgboostModelCV <-xgb.cv(data = train_matrix,
                            nrounds = ntrees,
                            nfold = cv.nfold,
                            showsd = TRUE,
                            verbose = TRUE,
                            "eval_metric" = "mlogloss",
                            "objective" = "multi:softprob",
                            "num_class" = numberOfClasses,
                            "max.depth" = currentDepth,
                            "eta" = currentEta,                               
                            "subsample" = currentSubsampleRate,
                            "colsample_bytree" = currentColsampleRate,
                            print_every_n = 5,
                            "min_child_weight" = currentMinChild,
                            booster = "gbtree",
                            early_stopping_rounds = 5)
    xvalidationScores <- as.data.frame(xgboostModelCV$evaluation_log)
    mlogloss <- tail(xvalidationScores$test_mlogloss_mean, 1)
    tmlogloss <- tail(xvalidationScores$train_mlogloss_mean,1)
    output <- return(c(mlogloss, tmlogloss, currentSubsampleRate, currentColsampleRate, currentDepth, currentEta, currentMinChild))}))  

output <- as.data.frame(t(mloglossErrorsHyperparameters))
varnames <- c("Testmlogloss", "Trainmlogloss", "SubSampRate", "ColSampRate", "Depth", "eta", "currentMinChild")
names(output) <- varnames
head(output)

output%>%arrange(Testmlogloss)%>%head()

#After tuning.

numberOfClasses <- length(unique(sum_c_train$surface_num))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses,
                   eta=0.1,
                   gamma=0,
                   max_depth=9,
                   min_child_weight=3,
                   subsample=0.9,
                   colsample_bytree=0.9)
nround    <- 2000 # number of XGBoost rounds
cv.nfold  <- 5

cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   early_stopping_rounds = 50,
                   print_every_n = 100,
                   showsd = T, 
                   stratified = T,
                   maximize = F)

model<-xgb.train(params = xgb_params,
                 data = train_matrix,
                 nrounds = 209,
                 watchlist = list(val=test_matrix,train=train_matrix),
                 print_every_n = 50,
                 early_stopping_rounds = 50,
                 maximize = F)

test_pred<-predict(model,newdata = test_matrix)

test_prediction<-matrix(test_pred,nrow = numberOfClasses,
                        ncol = length(test_pred)/numberOfClasses)%>%t()%>%
  data.frame()%>%mutate(label=test_label+1,
                        max_prop=max.col(.,"last"))

confusionMatrix(factor(test_prediction$max_prop),
                factor(test_prediction$label),
                mode = "everything")

#Feture Importance.
names<-names(sum_c_train[,-c(1,310)])
importance_matrix = xgb.importance(feature_names = names, model = model)


xgb.plot.importance(head(importance_matrix,30))

#Prediction on the Final Dataset.

val_pred<-predict(model,newdata = te_final)

vals_prediction<-matrix(val_pred,nrow = numberOfClasses,
                        ncol = length(val_pred)/numberOfClasses)%>%t()%>%
  data.frame()%>%mutate(surface_num=max.col(.,"last"))%>%left_join(mapping,by=c("surface_num"))

cbind(series_id=sum_c_test[,1],data.frame("surface"=vals_prediction[,11]))%>%
  write.csv(.,"Xgboost_with_agg_pt.csv",row.names = F)
