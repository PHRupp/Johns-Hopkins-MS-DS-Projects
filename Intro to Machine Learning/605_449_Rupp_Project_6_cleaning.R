#Set the working directory
setwd("B:/Users/pathr/Documents/Education/JHU/DATA/605.449.81.FA18/P6")

predicted_breast_cancer_val = 'M'
predicted_glass_val = 6
predicted_iris_val = "Iris-setosa"
predicted_soybean_val = "D2"
predicted_house_votes_val = "republican"

#### Breast Cancer ####
data <- read.csv("breast_cancer_modified.txt", as.is = TRUE)

#remove id
new_data <- data[, -1]

new_data$diagnosis <- ifelse( new_data$diagnosis == predicted_breast_cancer_val, 1, 0 )


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
  , paste0("in_normalized_predict_diagnosis_", predicted_breast_cancer_val, "_breast_cancer.csv")
  , row.names = F
)







#### Glass ####
data <- read.csv("glass_modified.txt", as.is = TRUE)

#remove id
new_data <- data[, -1]

new_data$Type <- ifelse( new_data$Type == predicted_glass_val, 1, 0 )

#shift the class column to the front
cols <- colnames(new_data)
new_data <- new_data[, c( cols[length(cols)], cols[-length(cols)] )]

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
norm_data <- data.frame(Type = new_data$Type, normalized )

#Write the normalized values to CSV
write.csv(
  norm_data
  , paste0("in_normalized_predict_Type_", predicted_glass_val, "_glass.csv")
  , row.names = F
)








#### Iris ####
data <- read.csv("iris_modified.txt", as.is = TRUE)

new_data <- data

new_data$class <- ifelse( new_data$class == predicted_iris_val, 1, 0 )

#shift the class column to the front
cols <- colnames(new_data)
new_data <- new_data[, c( cols[length(cols)], cols[-length(cols)] )]

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
norm_data <- data.frame(class = new_data$class, normalized )

#Write the normalized values to CSV
write.csv(
  norm_data
  , paste0("in_normalized_predict_Type_", predicted_iris_val, "_iris.csv")
  , row.names = F
)









#### Soybean ####
data <- read.csv("soybean_small_modified.txt", as.is = TRUE)

new_data <- data

new_data$class <- ifelse( new_data$class == predicted_soybean_val, 1, 0 )

#shift the class column to the front
cols <- colnames(new_data)
new_data <- new_data[, c( cols[length(cols)], cols[-length(cols)] )]

#normalize data to (0 - 1)
#careful, the class field is being ignored here
normalized <- sapply(
  new_data[,-1]
  , FUN = function(num_array){
    
    min_val <- min( num_array )
    max_val <- max( num_array )
    diff <- max_val - min_val
    
    if( diff == 0){
      return( min_val )
    } else {
      return( (num_array - min_val) / diff )
    }
  }
)



#Create data frame from the normalized matrix
norm_data <- data.frame(class = new_data$class, normalized )

#Write the normalized values to CSV
write.csv(
  norm_data
  , paste0("in_normalized_predict_Type_", predicted_soybean_val, "_soybean_small.csv")
  , row.names = F
)







#### House Votes 84 ####
data <- read.csv("house-votes-84_modified.txt", as.is = TRUE)

new_data <- data


new_data$class <- ifelse( new_data$class == predicted_house_votes_val, 1, 0 )

#normalize data to (0 - 1)
#careful, the class field is being ignored here
fixed_missing <- sapply(
  new_data[,-1]
  , FUN = function(col){
    
    return( ifelse(col == '?', 'u', col) )
  }
)

new_data <- data.frame(class = new_data$class, fixed_missing )


#normalize data to (0 - 1)
#careful, the class field is being ignored here
numbers <- sapply(
  new_data[,-1]
  , FUN = function(col){
    
    return( 
      ifelse(
        col == '?'
        , '0.5'
        , ifelse(
          col == 'y'
          , 1
          , 0
        )
      ) 
    )
  }
)

#Create data frame from the normalized matrix
norm_data <- data.frame(class = new_data$class, numbers )

#Write the normalized values to CSV
write.csv(
  norm_data
  , paste0("in_normalized_predict_Type_", predicted_house_votes_val, "_house-votes-84.csv")
  , row.names = F
)












#### clear variables ####
rm(list = ls())
