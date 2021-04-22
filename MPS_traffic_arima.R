library(dplyr)
library(forecast)
library(astsa)
library(greybox)
library(ggplot2)
library(ggthemes)
library(lubridate)
library(gridExtra)
theme_set(theme_base())

forecast_metrics <- function(forecast, y, lower, upper, scale, alpha = 0.05) {
    me=ME(y, forecast)
    mpe=MPE(y, forecast)
    mse=MSE(y, forecast)
    mae=MAE(y, forecast)
    mape=MAPE(y, forecast)
    
    msis <- sMIS(y, lower, upper, scale, level = 1 - alpha)
    
    mets <- list(me = me, mpe = mpe, mse = mse, mae = mae, mape = mape, msis = msis)
    return(mets)
}

Metro_Interstate_Traffic_Volume <- read.csv("~/Documents/Fall_2020/STAT_5825/Project/Data/Metro_Interstate_Traffic_Volume.csv", header = T)
traffic <- Metro_Interstate_Traffic_Volume
traffic$t <- 1:nrow(traffic)
head(traffic)
table(traffic$holiday)
ts.plot(traffic$traffic_volume)

traffic$temp_F <- (traffic$temp - 273.15) * 9/5 + 32
h_ind <- numeric(nrow(traffic))
for (i in 1:length(h_ind)) {
    if (traffic[i, "holiday"] != "None") {
        h_ind[i:(i + 23)] <- 1
    }
}
traffic$holiday_ind <- h_ind

par(mfrow = c(4,1))
ts.plot(traffic$temp_F, main = "Temperature (F)")
ts.plot(traffic$rain_1h, main = "Rainfall by Hr (mm)")
ts.plot(traffic$snow_1h, main = "Snowfall by Hr (mm)")
ts.plot(traffic$traffic_volume, main = "Traffic Volume")
dev.off()

imp_temps <- which(traffic$temp_F < -50)
traffic[imp_temps, "temp_F"]
# 10 impossible temps, all -460 degrees. These should be replaced/imputed somehow.
tsoutliers(traffic$temp_F)
traffic$temp_F_clean <- tsclean(traffic$temp_F)

imp_rain <- which(traffic$rain_1h > 100)
imp_rain
traffic$rain_1h_clean <- traffic$rain_1h
traffic[imp_rain, "rain_1h_clean"]
# One impossible reading of 9000+ mm. Should be imputed.
traffic[imp_rain, "rain_1h"] <- mean(traffic[c(imp_rain - 1, imp_rain + 1), "rain_1h"])

length(which(traffic$snow_1h > 0))
# Only 63 hours in which there was snowfall in Minnesota over the course of 6 years? This whole series seems suspect.

# Adding lagged variables for holdiay_ind and temp_F
tr.vol <- ts(traffic$traffic_volume, frequency = 24)
temp <- ts(traffic$temp_F_clean, frequency = 24)
holi <- ts(traffic$holiday_ind, frequency = 24)
traf <- ts.intersect(traffic_volume = tr.vol, temp_l1 = stats::lag(temp, -1), holiday = holi, dframe = T)
traf$t <- 1:nrow(traf)
head(traffic)
str(traf)

par(mfrow = c(3,1))
ts.plot(traffic$temp_F_clean, main = "Temperature (F)")
ts.plot(traffic$rain_1h, main = "Rainfall by Hr (mm)")
ts.plot(traffic$traffic_volume, main = "Traffic Volume")
dev.off()

colors <- c("TrafficVolume" = "black", "Temperature" = "tomato")

traffic$t <- as_datetime(1:nrow(traffic)*3600 + 1349168400)
p1 <- ggplot(traffic, aes(t, y = traffic_volume)) +
    geom_line(color = "black") +
    scale_x_datetime(date_labels = "%Y %b %d", date_breaks = "3 month") +
    scale_color_manual(values = colors) +
    labs(subtitle = "Full Time Series", x= "", y = "Traffic Volume", color = "Legend") +
    theme(axis.text.x=element_text(angle=45, hjust=1)) 
p2 <- ggplot(traffic[1:168,], aes(t, y = traffic_volume)) +
    geom_line(color = "black") +
    scale_x_datetime(date_labels = "%Y %b %d", date_breaks = "1 day") +
    scale_color_manual(values = colors) +
    labs(subtitle = "First Week in the Time Series", x = "Date", y = "Traffic Volume", color = "Legend")
p3 <- ggplot(traffic[1:2880,], aes(t, y = traffic_volume)) +
    geom_line(color = "black") +
    # geom_point(aes(t, y = traffic_volume, color = as.factor(holiday_ind), alpha = holiday_ind)) +
    scale_x_datetime(date_labels = "%Y %b %d", date_breaks = "2 week") +
    # scale_color_manual(values = colors) +
    labs(subtitle = "First 4 Months in the Time Series", x = "Date", y = "Traffic Volume")
p4 <- ggplot(traffic[1:2880,], aes(t, y = temp_F_clean)) +
    geom_line(color = "black") +
    # geom_point(aes(t, y = temp_F_clean, color = as.factor(holiday_ind), alpha = holiday_ind)) +
    scale_x_datetime(date_labels = "%Y %b %d", date_breaks = "2 week") +
    # scale_color_manual(values = colors) +
    labs(subtitle = "First 4 Months in the Time Series", x = "Date", y = "Temperature (F)")


grid.arrange(grobs = list(p1, p2), nrow = 2)
grid.arrange(grobs = list(p3, p4), nrow = 2)    


# Now we can see some weirdness in the rainfall TS too.
# No measured rainfall for nearly 20000 hours (~ 2.5 yrs) from 25000 to 48000 hrs?

ts.plot(traffic$traffic_volume, main = "Traffic Volume")

# Splitting into Calibration and Forecasting sets
# Saving two months for forecasting
traffic.calib <- traffic[1:46764,]
traffic.fore <- traffic[46765:48204,]
traf.calib <- traf[1:46763,]
traf.fore <- traf[46764:48203,]

write.csv(traffic.calib[,c("traffic_volume", "t", "temp_F_clean", "holiday_ind")], "~/Documents/Fall_2020/STAT_5825/Project/Data/MPS_Traffic_Volume_train.csv", row.names = F)
write.csv(traffic.fore[,c("traffic_volume", "t", "temp_F_clean", "holiday_ind")], "~/Documents/Fall_2020/STAT_5825/Project/Data/MPS_Traffic_Volume_test.csv", row.names = F)
write.csv(traf.calib, "~/Documents/Fall_2020/STAT_5825/Project/Data/MPS_Traffic_Volume_train_xreg.csv", row.names = F)
write.csv(traf.fore, "~/Documents/Fall_2020/STAT_5825/Project/Data/MPS_Traffic_Volume_test_xreg.csv", row.names = F)


start <- Sys.time()
fit <- auto.arima(ts(traf.calib$traffic_volume, frequency = 24),
                     # xreg = cbind(traf.calib$temp_l1, traf.calib$holiday),
                     max.p = 6, max.q = 6,
                      max.P = 3, max.Q = 3, max.order = 10,
                       seasonal = T, stepwise = T,parallel = F, num.cores = 2)
end <- Sys.time()
runtime <- end - start
runtime

summary(fit)

fit2 <- sarima(ts(traffic.calib$traffic_volume, frequency = 24), 3,0,5,1,0,2,S = 24)
fit2

frcst <- predict(fit, n.ahead = 1440
                 # , newxreg = cbind(traf.fore$temp_l1, traf.fore$holiday)
                 )
U <- frcst$pred + 1.96 * frcst$se
L <- frcst$pred - 1.96 * frcst$se

y <- traffic.fore[,"traffic_volume"]
# y <- traf.fore[,"traffic_volume"]
yhat <- frcst$pred
err <- y - yhat
y.calib <- traffic$traffic_volume
mad <-  mean(abs(diff(as.vector(y.calib), 1)))
mad
print(forecast_metrics(err, y, L, U, scale = mad))

df.forecast <- data.frame(t = traffic[(nrow(traffic)-1439):nrow(traffic), "t"],
                          y = y,
                          yhat = yhat,
                          upper = U,
                          lower = L)

ggplot(df.forecast, aes(t, y)) +
    geom_line() +
    geom_line(aes(y = yhat), color = "tomato") +
    geom_ribbon(aes(ymin = lower, ymax = upper, fill = "tomato"), alpha = .25)

write.csv(frcst, "~/Documents/Fall_2020/STAT_5825/Project/Data/forecast_arima_203201_24_nahead_1440.csv", row.names = F)


# Ran the auto_arima.R script from the darima GitHub to get the following data and times
# [1] "Begin: ncores = 1"
# [1] "End: ncores = 1"
# Time difference of 4.031571 mins
# [1] "Begin: ncores = 2"
# [1] "End: ncores = 2"
# Time difference of 3.607387 mins
# [1] "Begin: ncores = 3"
# [1] "End: ncores = 3"
# Time difference of 2.841271 mins

load("~/Documents/Fall_2020/STAT_5825/Project/Data/auto_arima_result.RData")

summary(f_arima_NC1[[1]]$fit)
# Fit a ARIMA(3,0,1,1,0,0)_24 model with intercept

fit <- sarima(ts(traffic.calib$traffic_volume, frequency = 24), 3,0,1,1,0,0,S = 24)
fit

y <- traffic.fore[,"traffic_volume"]
yhat <- f_arima_NC1[[1]]$forec$mean
err <- y - yhat
y.calib <- traffic.calib$traffic_volume
fit.calib <- f_arima_NC1[[1]]$fit$fitted
lag1 <- lag(fit.calib, 1)
mad <- MAE(y.calib, lag1)
mad
print(forecast_metrics(err, y, L, U, scale = mad))
# $me
# [1] 3253.452
# 
# $mpe
# [1] 2.311479
# 
# $mse
# [1] 10594273
# 
# $mae
# [1] 3253.452
# 
# $mape
# [1] 2.311479
# 
# $msis
# [1] 15.50575


summary(f_arima_NC2[[1]]$fit)
# Again, fit a ARIMA(3,0,1,1,0,0)_24 model with intercept
# This is expected. It just did the same fit a bit faster.

summary(f_arima_NC3[[1]]$fit)
# Again, fit a ARIMA(3,0,1,1,0,0)_24 model with intercept
# Again, same model but fit much faster.
