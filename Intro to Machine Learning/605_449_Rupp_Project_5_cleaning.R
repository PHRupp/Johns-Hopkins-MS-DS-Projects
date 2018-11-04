#Set the working directory
setwd("B:/Users/pathr/Documents/Education/JHU/DATA/605.449.81.FA18/P5")

#### Breast Cancer ####
data <- read.csv("breast_cancer_modified.txt", as.is = TRUE)

#remove id
new_data <- data[, -1]

new_data$diagnosis <- ifelse( new_data$diagnosis == 'M', 1, 0 )


#Write the normalized values to CSV
write.csv(
  new_data
  , "in_standard_breast_cancer.csv"
  , row.names = F
)

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

#Create data frame from the normalized matrix
norm_data <- data.frame(diagnosis = new_data$diagnosis, normalized )

#Write the normalized values to CSV
write.csv(
  norm_data
  , "in_normalized_breast_cancer.csv"
  , row.names = F
)


#Turn the numerical data into categorical based on
#buckets made by 1st quartile, median, and 3rd quartile
#  A < 1st quartile <= B < median <= C < 3rd quartile <= D
categories_set <- sapply(
  new_data[,-1]
  , FUN = function(num_array){
    
    summary_of_data <- summary( num_array )
    
    quartile_1 <- summary_of_data[2]
    median_val <- summary_of_data[3]
    quartile_3 <- summary_of_data[5]
    
    #  A < 1st quartile <= B < median <= C < 3rd quartile <= D
    categories <- ifelse( 
      num_array < quartile_1
      , 'A'
      , ifelse(
        num_array < median_val
        , 'B'
        , ifelse(
          num_array < quartile_3
          , 'C'
          , 'D'
        )
      )
    )
    
    return( categories )
  }
)

#Create data frame from the normalized matrix
category_data <- data.frame(diagnosis = data$diagnosis, categories_set )

#Write the normalized values to CSV
write.csv(
  category_data
  , "in_categories_breast_cancer.csv"
  , row.names = F
)

rm(list = ls())

#### IMAGE SEGMENTATION ####










