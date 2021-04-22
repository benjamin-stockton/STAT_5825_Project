# # Save data as csv
# load("/home/bsuconn/Documents/Fall_2020/STAT_5825/Project/darima-master/MPS_traffic.RData")
# if (!require("magrittr")) {
#   install.packages("magrittr")
# }
# library(magrittr)

# zones_name <- names(MPS_traffic)
# for(i in seq(length(MPS_traffic))){
#   # get series data
#   demand <- MPS_traffic[[i]]$x %>% as.numeric()
#   time <- attr(MPS_traffic[[i]]$x, "index") %>% as.character()
#   series_data <- data.frame(demand = demand, time = time)
#   assign(zones_name[i], series_data)
#   rm(demand, time, series_data)
  
#   # save series as csv
#   file_name <- paste("/home/bsuconn/Documents/Fall_2020/STAT_5825/Project/darima-master/data/",
#                      zones_name[i], "_train.csv", sep = "")
#   write.csv(get(zones_name[i]), file = file_name, row.names = FALSE)
# }

# for(i in seq(length())){
#   # get series data
#   demand <- MPS_traffic[[i]]$xx %>% as.numeric()
#   time <- attr(MPS_traffic[[i]]$xx, "index") %>% as.character()
#   series_data <- data.frame(demand = demand, time = time)
#   assign(zones_name[i], series_data)
#   rm(demand, time, series_data)
  
#   # save series as csv
#   file_name <- paste("/home/bsuconn/Documents/Fall_2020/STAT_5825/Project/darima-master/data/",
#                      zones_name[i], "_test.csv", sep = "")
#   write.csv(get(zones_name[i]), file = file_name, row.names = FALSE)
# }
