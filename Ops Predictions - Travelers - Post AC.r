
#set working directory
setwd("C:\\Users\\Blake.Abbenante\\Google Drive\\Work\\r\\PaxForecast\\")

#load the required libraries
require(randomForest) || install.packages("randomForest", repos="http://cran.rstudio.org")
library(randomForest)
require(caret) || install.packages("caret", repos="http://cran.rstudio.org")
library(caret)
require(ggplot2) || install.packages("ggplot2", repos="http://cran.rstudio.org")
library(ggplot2)


## clear the console of all variables
rm(list = ls())
gc()


## read the Pax data file
pax_data_full=read.csv("C:\\Users\\Blake.Abbenante\\Google Drive\\Work\\r\\PaxForecast\\data\\traveler-data-RAW.csv")
pax_data=pax_data_full
pax_data$ACYear<-as.numeric(as.character(pax_data$ACYear))
pax_data <- pax_data[which(pax_data$TravelYear>=2010|pax_data$ACYear>=2010),]
pax_data <- pax_data[which(pax_data$NetPax360DPD>0),]
#factor our ordinal and binary data
pax_data$TravelFlag = factor(pax_data$TravelFlag)
#pax_data$HasTraveled = factor(pax_data$HasTraveled)
pax_data$SchoolScore = factor(pax_data$SchoolScore)
pax_data$TravelMonth = factor(pax_data$TravelMonth)
pax_data$TravelYear = factor(pax_data$TravelYear)
pax_data$ACYear = factor(pax_data$ACYear)
pax_data$IsPrivateBus = factor(pax_data$IsPrivateBus)
pax_data$IsCustom = factor(pax_data$IsCustom)
pax_data$TourCode = factor(pax_data$TourCode)
##drop some unneeded columns
pax_data=subset(pax_data, select=-c(SalesOp_id,SalesYearNumber,ProductMarket,BusinessUnitCode,StatusCode,Individual_Id,NetPax330DPD,NetPax300DPD,NetPax270DPD,EstTotalPax,EstTotalPax330DPD,EstTotalPax300DPD,EstTotalPax270DPD,EstTotalPax090DPD))
#remove time series for timeframes we haven't "seen" yet
#70
pax_data=subset(pax_data, select=-c(NetPaxAct60,NetPaxAct70,NetPaxAct80,NetPaxAct90,NetPaxAct100,NetPaxAct110,NetPaxAct120,NetPaxAct130,NetPaxAct140,NetPaxAct150,
                                    EstPaxAct60,EstPaxAct70,EstPaxAct80,EstPaxAct90,EstPaxAct100,EstPaxAct110,EstPaxAct120,EstPaxAct130,EstPaxAct140,EstPaxAct150,
                                    Cancel70,Cancel80,Cancel90,Cancel100,Cancel110,Cancel120,Cancel130,Cancel140,Cancel150))

go_time<-pax_data[which(pax_data$TravelYear==2016),]
pax_data<-pax_data[which(pax_data$TravelYear!=2016),]
set.seed(197)
#split into test and train
indexes=sample(1:nrow(pax_data),size=0.4*nrow(pax_data))
train_data=pax_data[indexes,]
test_data=pax_data[-indexes,]



gc()


rf_model<-train(NetPax090DPD ~ NetPaxAct10+NetPaxAct20+NetPaxAct30+NetPaxAct40+NetPaxAct50+
                  EstPaxAct10+EstPaxAct20+EstPaxAct30+EstPaxAct40+EstPaxAct50+
#                  Cancel10+Cancel20+Cancel30+Cancel40+Cancel50+Cancel60+
                  NetPax540DPD+NetPax510DPD+NetPax480DPD+NetPax450DPD+NetPax420DPD+NetPax390DPD+NetPax360DPD+
                  EstTotalPax540DPD+EstTotalPax510DPD+EstTotalPax480DPD+EstTotalPax450DPD+EstTotalPax420DPD+EstTotalPax390DPD+EstTotalPax360DPD+
                  InserttoAC+SchoolScore+TravelMonth+RegionCode+PreviousACTours+PreviousACPpax+IsPrivateBus+IsCustom,
                data=train_data,method="rf",
                trControl=trainControl(method="cv",number=5),
                prox=TRUE,allowParallel=TRUE)
print(rf_model)

my.results<-predict(rf_model,train_data)
groupsize.prediction.2016<-data.frame(my.results)
names(groupsize.prediction.2016)[1]<-paste("Pred")
groupsize.prediction.2016<-cbind(groupsize.prediction.2016,train_data$EstTotalPax360DPD)
names(groupsize.prediction.2016)[2]<-paste("Rez Num")
groupsize.prediction.2016<-cbind(groupsize.prediction.2016,train_data$GroupTrip_Id)
names(groupsize.prediction.2016)[3]<-paste("Group Trip")
groupsize.prediction.2016<-cbind(groupsize.prediction.2016,train_data$NetPax090DPD)
names(groupsize.prediction.2016)[4]<-paste("Pax90DPD")
groupsize.prediction.2016<-cbind(groupsize.prediction.2016,train_data$ActualizedDate)
names(groupsize.prediction.2016)[5]<-paste("AC Date")
groupsize.prediction.2016<-cbind(groupsize.prediction.2016,train_data$TourDate)
names(groupsize.prediction.2016)[6]<-paste("Trip Date")


write.csv(groupsize.prediction.2016, file = "blake8.csv")


my.results<-predict(rf_model,go_time)
groupsize.prediction.2016<-data.frame(my.results)
names(groupsize.prediction.2016)[1]<-paste("Pred")
groupsize.prediction.2016<-cbind(groupsize.prediction.2016,go_time$EstTotalPax360DPD)
names(groupsize.prediction.2016)[2]<-paste("Rez Num")
groupsize.prediction.2016<-cbind(groupsize.prediction.2016,go_time$GroupTrip_Id)
names(groupsize.prediction.2016)[3]<-paste("Group Trip")
groupsize.prediction.2016<-cbind(groupsize.prediction.2016,go_time$NetPax090DPD)
names(groupsize.prediction.2016)[4]<-paste("Pax")
groupsize.prediction.2016<-cbind(groupsize.prediction.2016,go_time$ActualizedDate)
names(groupsize.prediction.2016)[5]<-paste("AC Date")
groupsize.prediction.2016<-cbind(groupsize.prediction.2016,go_time$TourDate)
names(groupsize.prediction.2016)[6]<-paste("Trip Date")


write.csv(groupsize.prediction.2016, file = "blake2016.csv")

#save our model
save(rf_model, file="RF_prod_model_ETUS.RData")
