
setwd("B:/Users/pathr/Documents/Education/JHU/DATA/605.449.81.FA18/P5")

#### Breast Cancer ####
data <- read.csv("breast_cancer_modified.txt", as.is = TRUE)

#remove id
new_data <- data[, -1]

new_data$diagnosis <- ifelse( new_data$diagnosis == 'M', 1, 0 )

#normalize data to (0 - 1)
#careful, the class field is being ignored here
normalized <- sapply(
  new_data[,-1]
  , FUN = function(num_array){
  
    min_val <- min( num_array )
    max_val <- max( num_array )
    diff <- max_val - min_val
    
    return( (num_array - min_val) / diff )
  }
)

#
norm_data <- data.frame(diagnosis = new_data$diagnosis, normalized )

write.csv(
  norm_data
  , "in_norm_breast_cancer.csv"
  , row.names = F
)

write.csv(
  new_data
  , "in_breast_cancer.csv"
  , row.names = F
)

rm(list = ls())

#### IMAGE SEGMENTATION ####










