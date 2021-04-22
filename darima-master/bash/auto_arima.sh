# Get current dir path for this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#echo Run the DARIMA model from $DIR

cd "$DIR/.."

Rscript  auto_arima.R  > auto_arima.out 2> auto_arima.log
