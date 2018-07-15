###########################################################################
# Trip Cancel Classification
# 9/16/14
###########################################################################

#set working directory
setwd("C:\\Users\\Blake.Abbenante\\Google Drive\\Work\\r\\PaxForecast\\")

#load the required libraries
require(SDMTools) || install.packages("SDMTools", repos="http://cran.rstudio.org") 
library(SDMTools)
require(ROCR) || install.packages("ROCR", repos="http://cran.rstudio.org")
library(ROCR)
require(randomForest) || install.packages("randomForest", repos="http://cran.rstudio.org")
library(randomForest)
require(rpart) || install.packages("rpart", repos="http://cran.rstudio.org") 
library(rpart)
require(gbm) || install.packages("gbm", repos="http://cran.rstudio.org") 
library(gbm)
require(e1071) || install.packages("e1071", repos="http://cran.rstudio.org") 
library(e1071)
require(ada) || install.packages("ada", repos="http://cran.rstudio.org") 
library(ada)
require(gbm) || install.packages("gbm", repos="http://cran.rstudio.org") 
library(gbm)
#require(h2o) || install.packages("h2o", repos="http://cran.rstudio.org") 
require(h2o) || install.packages(install.packages("C:\\Users\\Blake.Abbenante\\Google Drive\\Work\\r\\packages\\h2o_2.8.6.2.tar.gz",repos=NULL,type='source'))
library(h2o)


## clear the console of all variables
rm(list = ls())
gc()

## read the Pax data file
pax_data_full=read.csv("C:\\Users\\Blake.Abbenante\\Google Drive\\Work\\r\\PaxForecast\\data\\traveler-data.csv")
pax_data=pax_data_full
#factor our ordinal and binary data
pax_data$TravelFlag = factor(pax_data$TravelFlag)
#pax_data$HasTraveled = factor(pax_data$HasTraveled)
pax_data$SchoolScore = factor(pax_data$SchoolScore)
pax_data$TravelMonth = factor(pax_data$TravelMonth)
pax_data$TravelYear = factor(pax_data$TravelYear)
pax_data$IsPrivateBus = factor(pax_data$IsPrivateBus)
pax_data$IsCustom = factor(pax_data$IsCustom)
pax_data$TourCode = factor(pax_data$TourCode)
##drop some unneeded columns
pax_data=subset(pax_data, select=-c(SalesOp_id,SalesYearNumber,ProductMarket,BusinessUnitCode,StatusCode,Individual_Id,IsCustom,GrossPaxAct10,GrossPaxAct20,GrossPaxAct30,GrossPaxAct40,GrossPaxAct50,GrossPaxAct60,GrossPaxAct70,GrossPaxAct80,GrossPaxAct90,GrossPaxAct100,GrossPaxAct110,GrossPaxAct120,GrossPaxAct130,GrossPaxAct140,GrossPaxAct150))
#remove time series for timeframes we haven't "seen" yet
#70
pax_data=subset(pax_data, select=-c(NetPaxAct60,NetPaxAct70,NetPaxAct80,NetPaxAct90,NetPaxAct100,NetPaxAct110,NetPaxAct120,NetPaxAct130,NetPaxAct140,NetPaxAct150,
                                    EstPaxAct60,EstPaxAct70,EstPaxAct80,EstPaxAct90,EstPaxAct100,EstPaxAct110,EstPaxAct120,EstPaxAct130,EstPaxAct140,EstPaxAct150,
                                    Cancel60,Cancel70,Cancel80,Cancel90,Cancel100,Cancel110,Cancel120,Cancel130,Cancel140,Cancel150,
                                    ActResDiff60,ActResDiff70,ActResDiff80,ActResDiff90,ActResDiff100,ActResDiff110,ActResDiff120,ActResDiff130,ActResDiff140,ActResDiff150))

gotime<-pax_data[which(pax_data$TravelYear==2016),]
pax_data<-pax_data[which(pax_data$TravelYear!=2016),]
set.seed(94)
#split into test and train
indexes=sample(1:nrow(pax_data),size=0.2*nrow(pax_data))
test=pax_data[indexes,]
train=pax_data[-indexes,]



gc()


#build a Linear Regression model
#300
paxsize.lm <- lm(NetPax ~ NetPaxAct10+NetPaxAct20+NetPaxAct30+NetPaxAct40+NetPaxAct50+
                   EstPaxAct10+EstPaxAct20+EstPaxAct30+EstPaxAct40+EstPaxAct50+
                   Cancel10+Cancel20+Cancel30+Cancel40+Cancel50+
                   ActResDiff10+ActResDiff20+ActResDiff30+ActResDiff40+ActResDiff50+
                   InserttoAC+SchoolScore+TravelMonth+RegionCode+PreviousACTours+PreviousACPpax+IsPrivateBus,
                 data=train)
#predict the test set with the trained model
paxsize.lm.response = predict(paxsize.lm,type="response",newdata=test)
paxsize.lm.response.2016 = predict(paxsize.lm,type="response",newdata=gotime)



#build a Random Forest model
#300
paxsize.rf <- randomForest(NetPax ~ NetPaxAct10+NetPaxAct20+NetPaxAct30+NetPaxAct40+
                             EstPaxAct10+EstPaxAct20+
                             Cancel10+Cancel20+
                             ActResDiff10+ActResDiff20+
                             InserttoAC+SchoolScore+TravelMonth+RegionCode+PreviousACTours+PreviousACPpax+IsPrivateBus,
                           data=train, ntree=200, nodesize=10)


#predict the test set with the trained model
paxsize.rf.response = predict(paxsize.rf,type="response",newdata=test)
paxsize.rf.response.2016 = predict(paxsize.rf,type="response",newdata=gotime)

#build a DLNN
# Number of Pax
## Start a local cluster with 7 cores and 12GB RAM
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, max_mem_size = '12g', nthreads = 7)


h2o_train <- as.h2o(localH2O, train)
h2o_test <- as.h2o(localH2O, test)
h2o_gotime <- as.h2o(localH2O, gotime)


paxsize.dnn <- h2o.deeplearning(x = c(2,4,7,11,13:16,19:26),
                                #columns for 100days post AC -  c(2,4,11,13:16,18:27,35:44,50:59,65:74)
                                #columns for 70days post AC -  c(2,4,7,11,13:16,18:24,26:46),
                           y = 17, # column number for label
                           data = h2o_train, # data in H2O format
                           classification = FALSE,
                           autoencoder = FALSE,
                           #                           nfolds=10,
                           activation = "RectifierWithDropout", #There are several options here
                           input_dropout_ratio = 0.1, # % of inputs dropout
                           hidden_dropout_ratios = c(0.5,0.5,0.5,0.5,0.5), # % for nodes dropout
                           l2=.0005, #l2 penalty for regularization
                           seed=5,
                           hidden = c(200,200,200,200,200), # four layers of 100 nodes
                           variable_importances=TRUE,
                           epochs = 10) # max. no. of epochs

## Using the DNN model for predictions
paxsize.dnn.response <- h2o.predict(paxsize.dnn, h2o_test)
## Converting H2O format into data frame
paxsize.dnn.response.df <- as.data.frame(paxsize.dnn.response)


## Using the DNN model for predictions
paxsize.dnn.response.2016 <- h2o.predict(paxsize.dnn, h2o_gotime)
## Converting H2O format into data frame
paxsize.dnn.response.2016.df <- as.data.frame(paxsize.dnn.response.2016)

groupsize.prediction<-data.frame(test$GroupTrip_Id)
names(groupsize.prediction)[1]<-paste("Group Trip")
groupsize.prediction<-cbind(groupsize.prediction,test$ActualizedDate)
names(groupsize.prediction)[2]<-paste("AC Date")
groupsize.prediction<-cbind(groupsize.prediction,test$TourDate)
names(groupsize.prediction)[3]<-paste("Tour Date")
groupsize.prediction<-cbind(groupsize.prediction,test$NetPax)
names(groupsize.prediction)[4]<-paste("Net PPax")
groupsize.prediction<-cbind(groupsize.prediction,test$EstPaxAct20)
names(groupsize.prediction)[5]<-paste("Rez Num")
groupsize.prediction<- cbind(groupsize.prediction,paxsize.rf.response)
names(groupsize.prediction)[6]<-paste("RF")
groupsize.prediction<-cbind(groupsize.prediction,paxsize.dnn.response.df$predict)
names(groupsize.prediction)[7]<-paste("DLNN")
groupsize.prediction<-cbind(groupsize.prediction,paxsize.lm.response)
names(groupsize.prediction)[8]<-paste("LM")

write.csv(groupsize.prediction, file = "GroupSizePrediction_20AC.20150818.csv")

#write the 2016 predictions
groupsize.prediction.2016<-data.frame(gotime$GroupTrip_Id)
names(groupsize.prediction.2016)[1]<-paste("GroupTrip")
groupsize.prediction.2016<-cbind(groupsize.prediction.2016,gotime$EstTotalPax300DPD)
names(groupsize.prediction.2016)[2]<-paste("Rez Num")
groupsize.prediction.2016<- cbind(groupsize.prediction.2016,paxsize.rf.response.2016)
names(groupsize.prediction.2016)[3]<-paste("RF")
groupsize.prediction.2016<-cbind(groupsize.prediction.2016,paxsize.dnn.response.2016.df$predict)
names(groupsize.prediction.2016)[4]<-paste("DLNN")
groupsize.prediction.2016<-cbind(groupsize.prediction.2016,paxsize.lm.response.2016)
names(groupsize.prediction.2016)[5]<-paste("LM")

write.csv(groupsize.prediction.2016, file = "GroupSizePrediction_300_2016.90.20150806.csv")


#get intervals for the group size

paxsize.pred.rf <- predict(paxsize.rf, test, predict.all=TRUE)
paxsize.pred.rf.int <- apply( paxsize.pred.rf$individual, 1, function(x) 
{ c( mean(x) + c(-1,1)*sd(x), quantile(x, c(0.05,0.95)) )
})
write.csv(t(paxsize.pred.rf.int), file = "PaxSizeIntervals.csv")