
data <- read.csv("B:/Users/pathr/Documents/Education/JHU/DATA/605.449.81.FA18/forestfires.csv", as.is = TRUE)


data$month <- sapply(data$month, FUN = function(mon){
  
  if(mon == "jan"){ out <- 1 }
  else if(mon == "feb"){ out <- 2 }
  else if(mon == "mar"){ out <- 3 }
  else if(mon == "apr"){ out <- 4 }
  else if(mon == "may"){ out <- 5 }
  else if(mon == "jun"){ out <- 6 }
  else if(mon == "jul"){ out <- 7 }
  else if(mon == "aug"){ out <- 8 }
  else if(mon == "sep"){ out <- 9 }
  else if(mon == "oct"){ out <- 10 }
  else if(mon == "nov"){ out <- 11 }
  else if(mon == "dec"){ out <- 12 }
  else{ out <- 0 }
  
  return(out)
})

data$day <- sapply(data$day, FUN = function(day){
  
  if(day == "mon"){ out <- 1 }
  else if(day == "tue"){ out <- 2 }
  else if(day == "wed"){ out <- 3 }
  else if(day == "thu"){ out <- 4 }
  else if(day == "fri"){ out <- 5 }
  else if(day == "sat"){ out <- 6 }
  else if(day == "sun"){ out <- 7 }
  else{ out <- 0 }
  
  return(out)
})




write.csv(data, "B:/Users/pathr/Documents/Education/JHU/DATA/605.449.81.FA18/in_forestfires.csv", row.names = F)










