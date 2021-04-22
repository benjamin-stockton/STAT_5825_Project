# README.txt

###################################
Most code for this project requires Spark to run because DARIMA is not implemented as a package.
Setting up Spark was very time consuming and difficult on my
personal machine, so it is not worth a graders time to figure out.
###################################

###################################
Files that can be run without Spark:

- MPS_traffic_arima.R
    - generates initial exploratory results
- darima-master/auto_arima.sh
    - performs ARIMA modeling 
    - MUST SET THE LOCATION OF THE DATA IN auto_arima.R
- darima-master/auto_arima.sh
    - performs ARIMA modeling and forecasting on various max model orders.
    - Takes >3 hours to run.
    - MUST SET THE LOCATION OF THE DATA IN auto_arima_diff_orders.R
- darima-master/darima_arima_eval.R
    - Produces graphical and table outputs for the report
    - MUST SET THE LOCATION OF THE DATA