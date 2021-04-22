#! /usr/bin/Rscript

################
### set path ###
################
path = "~/Documents/Fall_2020/STAT_5825/Project"
setwd(path)

##################
### mps_traffic_data ###
##################
traffic <- read.csv("darima-master/data/MPS_Traffic_Volume_train_xreg.csv", header = T)
frcst.xreg <- read.csv("darima-master/data/MPS_Traffic_Volume_test_xreg.csv", header = T)[, c("temp_l1", "holiday_l1")]
# print(str(traffic))
traffic$x <- ts(traffic$traffic_volume, frequency = 24)
xreg <- cbind(traffic$temp_l1, traffic$holiday)
n.test <- (nrow(traffic) - 1440):nrow(traffic)
xreg.fore <- cbind(frcst.xreg[,1], frcst.xreg[,2])

traf <- list(x = traffic[n.test, "x"], xreg = xreg[n.test,], h = 480, xreg.fore = xreg.fore[1:480,])

################
### packages ###
################
library(forecast)
library(magrittr)
library(polynom)
library(quantmod)
# parallel computation
library(parallel)
library(doParallel)
library(foreach)

###################################
### auto.arima (method = "CSS") ###
###################################
for (ncores in c(3)){
  print(paste0("Begin: ncores = ", ncores))
  t0_arima <- Sys.time()
  f_arima <- lapply(list(traf), function(lentry){
    t0 <- Sys.time()
    fit <- forecast::auto.arima(lentry$x, 
                                xreg = lentry$xreg,
                                method = "CSS", 
                                max.p = 5, max.q = 5, 
                                max.P = 2, max.Q = 2,
                                max.order = 5,
                                stepwise = FALSE, parallel = TRUE, 
                                approximation = FALSE, 
                                num.cores = ncores)
    forec <- forecast(fit,
           xreg = lentry$xreg.fore,
           h = lentry$h)
    t1 <- Sys.time()
    tt <- t1 - t0
    return(append(lentry, list(fit = fit, forec = forec, time = tt)))
  })
  assign(paste0("f_arima_NC", ncores), f_arima)
  t1_arima <- Sys.time()
  time_arima <- t1_arima - t0_arima
  print(paste0("End: ncores = ", ncores))
  print(time_arima)
  rm(t0_arima, t1_arima, time_arima, f_arima)
}

str(f_arima_NC3)




# save.image("darima-master/data/auto_arima_result_xreg.RData")
