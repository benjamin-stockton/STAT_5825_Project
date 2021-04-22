#! /usr/bin/Rscript

################
### set path ###
################
path = "~/Documents/Fall_2020/STAT_5825/Project"
setwd(path)


eval_func <- function(x, xx, period, pred, lower, upper, level){
    freq <- period
    scaling <- mean(abs(diff(as.vector(x), freq)))
    mase <- abs(as.vector(xx) - as.vector(pred)) / scaling
    
    outsample <- as.numeric(xx)
    forecasts <- as.numeric(pred)
    smape <- (abs(outsample-forecasts)*200)/(abs(outsample)+abs(forecasts))
    
    # eg: level = 95
    alpha <- (100 - level)/100
    msis <- (upper - lower + 
              (2 / alpha) * (lower - xx) * (lower > xx) + 
              (2 / alpha) * (xx - upper) * (upper < xx)) / scaling
    
    return(list(mase = mase, smape = smape, msis = msis))
}

par.arima <- function(dat, model.order, ncores = 3) {
  fit <- forecast::auto.arima(traf$x, 
                              method = "CSS", 
                              max.p = model.order[1], max.q = model.order[2],
                              max.P = model.order[3], max.Q = model.order[4],
                              max.order = model.order[5],
                              stepwise = FALSE, parallel = TRUE, 
                              approximation = FALSE, 
                              num.cores = ncores)
  return(fit)
}

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

model.orders <- matrix(c(5,6,6,8, 5,6,6,8, 2,2,3,4, 2,2,3,4, 5,7,10,10), nrow = 4)
for (i in 1:4) {
  print("#########################################################################")
  print("Max Model Orders:")
  print(model.orders[i,])
  
  ncores = 3
  print(paste0("Begin: ncores = ", ncores))

  l.fit <- list()
  fits <- list()
  t0_arima <- Sys.time()
  t0 <- Sys.time()
  cl <- makeCluster(3)
  registerDoParallel(cl)
  l.fit <- foreach(j = 1:1) %dopar% {
    fits[[j]] <- par.arima(traf$x[1:1440], model.orders[i,])
  }
  stopCluster(cl)
  fit <- l.fit[[1]]
  forec <- forecast(fit,
          h = traf$h)
  t1 <- Sys.time()
  tt <- t1 - t0
  f_arima <- append(traf, list(fit = fit, forec = forec, time = tt))
  assign(paste0("f_arima_NC", ncores), f_arima)
  t1_arima <- Sys.time()
  time_arima <- t1_arima - t0_arima
  print(paste0("End: ncores = ", ncores))
  print(time_arima)
  rm(t0_arima, t1_arima, time_arima, f_arima)


  print(summary(f_arima_NC3$fit))

  y <- traffic.fore[,"traffic_volume"]
  yhat.arima <- f_arima_NC3$forec$mean
  L.arima.95 <- f_arima_NC3$forec$lower[,2]
  U.arima.95 <- f_arima_NC3$forec$upper[,2]

  y.calib <- traf$x[1:1440]

  print("ARIMA Forecast Metrics: ")
  print(t(lapply(eval_func(y.calib, y, 24, yhat.arima, L.arima.95, U.arima.95, level = 95), mean)))
  rm(f_arima_NC3)
}


