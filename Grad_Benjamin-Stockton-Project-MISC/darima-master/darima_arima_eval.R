# darima_arima_eval.R

library(dplyr)
library(forecast)
library(astsa)
library(greybox)
library(ggplot2)
library(ggthemes)
library(lubridate)
library(gridExtra)

theme_set(theme_base())

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

load("~/Documents/Fall_2020/STAT_5825/Project/darima-master/data/auto_arima_result.RData")
darima.frct <- read.csv("~/Documents/Fall_2020/STAT_5825/Project/darima-master/result/darima_forec_MPS_Traffic_Volume_2020-12-11-13:31:36.csv", header = T)
traffic.fore <- read.csv("~/Documents/Fall_2020/STAT_5825/Project/darima-master/data/MPS_Traffic_Volume_test.csv", header = T)
traffic.calib <- read.csv("~/Documents/Fall_2020/STAT_5825/Project/darima-master/data/MPS_Traffic_Volume_train.csv", header = T)
traffic <- rbind(traffic.calib, traffic.fore)


summary(f_arima_NC1[[1]]$fit)
# Fit a ARIMA(3,0,1,1,0,0)_24 model with intercept

y <- traffic.fore[,"traffic_volume"]
yhat.arima <- f_arima_NC1[[1]]$forec$mean
yhat.darima <- darima.frct$pred
L.arima.95 <- f_arima_NC1[[1]]$forec$lower[,2]
U.arima.95 <- f_arima_NC1[[1]]$forec$upper[,2]
L.darima <- darima.frct$lower
U.darima <- darima.frct$upper

y.calib <- traffic.calib$traffic_volume

print("ARIMA Forecast Metrics: ")
print(t(lapply(eval_func(y.calib, y, 24, yhat.arima, L.arima.95, U.arima.95, level = 95), mean)))
#      mase     smape    msis    
# [1,] 1.106261 62.06266 4.793611

print("DARIMA Forecast Metrics: ")
print(t(lapply(eval_func(y.calib, y, 24, yhat.darima, L.darima, U.darima, level = 95), mean)))
#      mase     smape    msis    
# [1,] 1.111263 62.42791 4.177812

forec.times <- ymd_hms(traffic$date_time)[(nrow(traffic) - 1439):nrow(traffic)]

df.forecast <- data.frame(t = forec.times,
                          y = y,
                          yhat.arima = yhat.arima,
                          yhat.darima = yhat.darima,
                          L.arima = L.arima.95,
                          U.arima = U.arima.95,
                          L.darima = L.darima,
                          U.darima = U.darima)

colors <- c("ARIMA" = "tomato", "DARIMA" = "dodgerblue")
ggplot(df.forecast, aes(x = t, y = y)) +
    geom_ribbon(aes(ymin = L.arima, ymax = U.arima, color = "ARIMA")
                , alpha = .25, fill = "lightgray", linetype = "dashed") +
    geom_ribbon(aes(ymin = L.darima, ymax = U.darima, color = "DARIMA")
                , alpha = .25, fill = "lightgray", linetype = "dashed") +
    geom_line(aes(y = yhat.arima, color = "ARIMA")) +
    geom_line(aes(y = yhat.darima, color = "DARIMA")) +
    geom_line(color = "black") +
    scale_x_datetime(date_labels = "%Y %b %d") +
    scale_color_manual(values = colors) +
    labs(x = "Date", y = "Traffic Volume", color = "Legend")
#########################################################################


H <- 1:1440
h.mase <- matrix(numeric(1440 * 2), ncol = 2)
h.msis <- matrix(numeric(1440 * 2), ncol = 2)

mad <- mean(abs(diff(as.vector(y.calib), 1)))
for (h in H) {
    y_h <- y[1:h]
    forec_a <- yhat.arima[1:h]
    forec_d <- yhat.darima[1:h]
    La_h <- L.arima.95[1:h]
    Ua_h <- U.arima.95[1:h]
    Ld_h <- L.darima[1:h]
    Ud_h <- U.darima[1:h]
    
    f_mets_arima <- lapply(eval_func(y.calib, y_h, 24, forec_a, La_h, Ua_h, level = 95), mean)
    f_mets_darima <- lapply(eval_func(y.calib, y_h, 24, forec_d, Ld_h, Ud_h, level = 95), mean)
    
    h.mase[h,1] <- f_mets_arima[["mase"]]
    h.mase[h,2] <- f_mets_darima[["mase"]]
    h.msis[h,1] <- f_mets_arima[["msis"]]
    h.msis[h,2] <- f_mets_darima[["msis"]]
}

horizon <- as.data.frame(cbind(H, h.mase, h.msis))
colnames(horizon) <- c("H", "MASE_ARIMA", "MASE_DARIMA", "MSIS_ARIMA", "MSIS_DARIMA")
head(horizon)

p1 <- ggplot(horizon, aes(x = H, y = MASE_ARIMA)) +
    geom_line(aes(color = "ARIMA")) +
    geom_line(aes(y = MASE_DARIMA, color = "DARIMA")) +
    labs(x = "Forecast Horizon", y = "MASE", color = "Legend")

p2 <- ggplot(horizon, aes(x = H, y = MSIS_ARIMA)) +
    geom_line(aes(color = "ARIMA")) +
    geom_line(aes(y = MSIS_DARIMA, color = "DARIMA")) +
    labs(x = "Forecast Horizon", y = "MSIS", color = "Legend")

grid.arrange(arrangeGrob(grobs = list(p1, p2), nrow = 1))

##########################################################

parts <- read.csv("/home/bsuconn/Documents/Fall_2020/STAT_5825/Project/darima/result_no_loop/darima_results_n_par.csv",
                  header = T)
parts$tot_time <- parts$time_mapred + parts$time_forec + parts$time_modeval

p1 <- ggplot(parts, aes(n_par, mase)) +
    geom_point(size = 2) +
    labs(y = "MASE", x = "Number of Subseries")

p2 <- ggplot(parts, aes(n_par, msis)) +
    geom_point(size = 2) +
    labs(y = "MSIS", x = "Number of Subseries")

p3 <- ggplot(parts, aes(n_par, tot_time)) +
    geom_point(size = 2) +
    labs(y = "Total Time", x = "Number of Subseries")

grid.arrange(arrangeGrob(grobs = list(p1, p2, p3), nrow = 1))


# Model forecasts for K = 150 vs K = 600

darima.frct.150 <- read.csv("~/Documents/Fall_2020/STAT_5825/Project/darima-master/result/darima_forec_MPS_Traffic_Volume_2020-12-11-13:31:36.csv", header = T)

darima.frct.600 <- read.csv("~/Documents/Fall_2020/STAT_5825/Project/darima-master/result/darima_forec_MPS_Traffic_Volume_2020-12-05-16:07:00.csv", header = T)


y <- traffic.fore[,"traffic_volume"]
yhat.150 <- darima.frct.150$pred
yhat.600 <- darima.frct.600$pred
L.150 <- darima.frct.150$lower
U.150 <- darima.frct.150$upper
L.600 <- darima.frct.600$lower
U.600 <- darima.frct.600$upper

y.calib <- traffic.calib$traffic_volume

print("DARIMA Forecast Metrics (K = 150): ")
print(t(lapply(eval_func(y.calib, y, 24, yhat.150, L.150, U.150, level = 95), mean)))
#      mase     smape    msis    
# [1,] 1.111263 62.42791 4.177812

print("DARIMA Forecast Metrics (K = 600): ")
print(t(lapply(eval_func(y.calib, y, 24, yhat.600, L.600, U.600, level = 95), mean)))
#      mase     smape    msis    
# [1,] 1.090227 60.25367 14.52058

forec.times <- ymd_hms(traffic$date_time)[(nrow(traffic) - 1439):nrow(traffic)]

df.forecast <- data.frame(t = forec.times,
                          y = y,
                          yhat.150 = yhat.150,
                          yhat.600 = yhat.600,
                          L.150 = L.150,
                          U.150 = U.150,
                          L.600 = L.600,
                          U.600 = U.600)

colors <- c("K = 150" = "tomato", "K = 600" = "dodgerblue")
ggplot(df.forecast, aes(x = t, y = y)) +
  geom_ribbon(aes(ymin = L.150, ymax = U.150, fill = "K = 150"),
              alpha = .25, linetype = "dashed") +
  geom_ribbon(aes(ymin = L.600, ymax = U.600, fill = "K = 600"),
              alpha = .25, linetype = "dashed") +
  geom_line(aes(y = yhat.150, color = "K = 150")) +
  geom_line(aes(y = yhat.600, color = "K = 600")) +
  geom_line(color = "black") +
  scale_x_datetime(date_labels = "%Y %b %d") +
  scale_color_manual(values = colors) +
  labs(x = "Date", y = "Traffic Volume", fill = "Legend", color = "Legend")
