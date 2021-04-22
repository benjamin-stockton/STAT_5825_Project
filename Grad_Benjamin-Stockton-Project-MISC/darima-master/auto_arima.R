#! /usr/bin/Rscript

################
### set path ###
################
path = "~/Documents/Fall_2020/STAT_5825/Project"
setwd(path)

##################
### mps_traffic_data ###
##################
traffic <- read.csv("darima-master/data/MPS_Traffic_Volume_train.csv", header = T)
traffic.fore <- read.csv("darima-master/data/MPS_Traffic_Volume_test.csv", header = T)

# print(str(traffic))
traffic$x <- ts(traffic$traffic_volume, frequency = 24)
traf <- list(x = traffic$x, h = 1440)

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
for (ncores in c(1,2,3)){
# for (ncores in c(3)){
  print(paste0("Begin: ncores = ", ncores))
  t0_arima <- Sys.time()
  f_arima <- lapply(list(traf), function(lentry){
    t0 <- Sys.time()
    fit <- forecast::auto.arima(lentry$x, 
                                method = "CSS", 
                                max.p = 5, max.q = 5,
                                max.P = 2, max.Q = 2,
                                max.order = 5,
                                stepwise = FALSE, parallel = TRUE, 
                                approximation = FALSE, 
                                num.cores = ncores)
    forec <- forecast(fit,
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

save.image("darima-master/data/auto_arima_result.RData")
# save.image("darima-master/data/auto_arima_result_xreg.RData")

