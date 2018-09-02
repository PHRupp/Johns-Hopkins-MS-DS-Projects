#######################################################################
# Project #1
# Author: Patrick H. Rupp
# Course: 605.449 - Introduction to Machine Learning
# Due: September 9, 2018

#### DEFINE GLOBAL REQUIREMENTS ####

# Set global environment options
options(stringAsFactors = FALSE)

#Seed makes the results reproducible
set.seed(150)

#

#### RETRIEVE ALL DATA FROM WEB ####

# Add the source archive url to the sub sets to make full paths
source_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/"
breast_cancer_url <- paste0(source_url, "breast-cancer-wisconsin/wdbc.data")
soybean_url <- paste0(source_url, "soybean/soybean-small.data")
glass_url <- paste0(source_url, "glass/glass.data")
iris_url <- paste0(source_url, "iris/iris.data")
vote_url <- paste0(source_url, "voting-records/house-votes-84.data")

#Define the columns for the iris data
glass_cols <- c(
  "Id"
  , "Refractice_Index"
  , "Sodium"
  , "Magnesium"
  , "Aluminum"
  , "Silicon"
  , "Potassium"
  , "Calcium"
  , "Barium"
  , "Iron"
  , "Type"
)

#retrieve the data from the source
glass_set <- read.csv(
  glass_url
  , as.is = TRUE
  , col.names = glass_cols
)

#Define the columns for the iris data
iris_cols <- c(
  "Sepal_Length"
  , "Sepal_Width"
  , "Petal_Length"
  , "Petal_Width"
  , "Class"
)

#retrieve the data from the source
iris_set <- read.csv(
  iris_url
  , as.is = TRUE
  , col.names = iris_cols
)

#Define the columns for the voter data
vote_cols <- c(
  "Class"
  , "Handicapped_Infants"
  , "Water_Project_Cost_Sharing"
  , "Adoption_Of_Budget_Resolution"
  , "Physician_Fee_Freeze"
  , "El_Salvador_Aid"
  , "Religious_Groups_In_Schools"
  , "Anti_Satellite_Test_Ban"
  , "Aid_To_Nicaraguan_Contras"
  , "Mx_Missile"
  , "Immigration"
  , "Synfuels_Corporation_Cutback"
  , "Education_Spending"
  , "Superfund_Right_To_Sue"
  , "Crime"
  , "Duty_Free_Exports"
  , "Export_Administration_Act_South_Africa"
)

#retrieve the data from the source
vote_set <- read.csv(
  vote_url
  , as.is = TRUE
  , na.strings = "?"
  , col.names = vote_cols
)

#### DEFINE WINNOW FUNCTION ####

#https://en.wikipedia.org/wiki/Winnow_(algorithm)
#http://www.cs.yale.edu/homes/aspnes/pinewiki/WinnowAlgorithm.html

#Building the weights using the Winnow2 algorithm
get_winnow2_weights <- function(
  training_data_set
  , cols_for_prediction #columns must have values 0 or 1
  , col_to_pred #column must have values 0 or 1
  , bounds = length(cols_for_prediction)
  , weight_step = 2
){
  
  #Initialize the weights
  col_weights <- replicate( length(cols_for_prediction), 1)
  
  #Go through all records and change the weights for the model
  for(row_index in 1:nrow(training_data_set)){
    
    #Calculate the sum of all weighted values and compare to the bound to make a prediction
    if( sum( col_weights * training_data_set[row_index, cols_for_prediction] ) > bounds ){ 
      prediction <- 1 
    } else { prediction <- 0 }
    
    #Incorrect predictions will have their weights changed
    if( prediction != training_data_set[row_index, col_to_pred] ){
      
      #Determine which values need to have their weights changed
      cols_to_change <- which( training_data_set[row_index, cols_for_prediction] == 1 )
      
      #Promote weights up if 1 otherwise demote the weights
      if( prediction == 1 ){ 
        col_weights[cols_to_change] <- col_weights[cols_to_change] / weight_step 
      } else { col_weights[cols_to_change] <- col_weights[cols_to_change] * weight_step }
    }
  }
  
  #Send the column weights back
  return(col_weights)
}

#
predict_winnow2 <- function(
  testing_data_set
  , cols_for_prediction #columns must have values 0 or 1
  , col_to_pred #column must have values 0 or 1
  , weights
  , bounds = length(cols_for_prediction)
){
  #Initialize the predictions
  predictions <- replicate( nrow(testing_data_set), NA)
  
  #Go through all records and change the weights for the model
  for(row_index in 1:nrow(testing_data_set)){
    
    #Calculate the sum of all weighted values and compare to the bound to make a prediction
    if( sum( col_weights * testing_data_set[row_index, cols_for_prediction] ) > bounds ){ 
      predictions[row_index] <- 1 
    } else { predictions[row_index] <- 0 }
  }
  
  return(predictions)
}

#

#### WINNOW: ANALYZE IRIS DATA ####

#Analyze the data
iris_summary <- summary(iris_set)

#Set the value that we want to predict
iris_val_to_predict <- "Iris-setosa"

#Define the column to predict
iris_set$Predict <- ifelse(iris_set$Class == iris_val_to_predict, 1, 0)

#Define values for binary indicators for the predictions( using medians of numeric values )
sepal_length_median <- as.numeric( trimws( substring(iris_summary[3,1], 9) ))
sepal_width_median <- as.numeric( trimws( substring(iris_summary[3,2], 9) ))
petal_length_median <- as.numeric( trimws( substring(iris_summary[3,3], 9) ))
petal_width_median <- as.numeric( trimws( substring(iris_summary[3,4], 9) ))

#Recreate the columns used for the prediction
iris_set$Sepal_Length_calc <- ifelse(iris_set$Sepal_Length > sepal_length_median, 1, 0)
iris_set$Sepal_Width_calc <- ifelse(iris_set$Sepal_Width > sepal_width_median, 1, 0)
iris_set$Petal_Length_calc <- ifelse(iris_set$Petal_Length > petal_length_median, 1, 0)
iris_set$Petal_Width_calc <- ifelse(iris_set$Petal_Width > petal_width_median, 1, 0)

#Set the index for the columns to use for the prediction
cols_for_prediction <- 7:10

#Make the portion of the training set P%
training_size <- floor(0.75 * nrow(iris_set))

#define which indices will be used for the training set
train_indices <- sample(seq_len(nrow(iris_set)), size = training_size)

#Split the data into the training and testing set
train_set <- iris_set[train_indices, ]
test_set <- iris_set[-train_indices, ]

#Calculate the models attribute weights
iris_weights <- get_winnow2_weights(
  training_data_set = train_set
  , cols_for_prediction = cols_for_prediction
  , col_to_pred = 6
)

#Use the model to predict what the values are
test_set$Predictions <- predict_winnow2(
  testing_data_set = test_set
  , cols_for_prediction = cols_for_prediction
  , col_to_pred = 6
  , iris_weights
)

#See % of successful predictions
length( which(test_set$Predict == test_set$Predictions) ) / nrow(test_set)






















