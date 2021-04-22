exoTest <- read.csv("~/Downloads/exoTest.csv")

star1 <- exoTest[1,]


plot(1:(length(star1)-1), star1[2:length(star1)], type = "l")


exoPlanetStars <- exoTest[exoTest$LABEL == 2,]
exoStar <- exoPlanetStars[1,]

plot(1:(length(exoStar)-1), exoStar[2:length(exoStar)], type = "l")

noPlanetStars <- exoTest[exoTest$LABEL == 1,]
noPlanet <- noPlanetStars[1,]

plot(1:(length(noPlanet)-1), noPlanet[2:length(noPlanet)], type = "l")

CTCovid <- read.csv("~/Downloads/connecticut-history.csv", header = T)
hospcurr <- rev(ts(log(CTCovid$hospitalizedCurrently)))
hospdiff7 <- diff(hospcurr, differences = 7)

par(mfrow = c(2,1))
ts.plot(hospcurr)
ts.plot(hospdiff7)

par(mfrow = c(4,2))
for (i in 1:7){
    ts.plot(diff(hospcurr, differences = i), main = paste0("Diff = ", i))
}


