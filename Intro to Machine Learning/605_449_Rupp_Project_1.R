#######################################################################
# Project #1
# Author: Patrick H. Rupp
# Course: 605.449 - Introduction to Machine Learning
# Due: September 9, 2018

#### DEFINE GLOBAL REQUIREMENTS ####

# Set global environment options
options(stringAsFactors = FALSE)

#Seed makes the results reproducible
set.seed(455)

#Define the % of training set to use
training_percent <- 0.7

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
#http://www.cs.tau.ac.il/~mansour/ml-course-10/scribe4.pdf

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

#Using the weights output above, we predict the values
predict_winnow2 <- function(
  testing_data_set
  , cols_for_prediction #columns must have values 0 or 1
  , col_to_pred #column must have values 0 or 1
  , col_weights
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
iris_summary

#Set the value that we want to predict
iris_val_to_predict <- "Iris-versicolor"

#Define the column to predict
iris_set$Predict <- ifelse(iris_set$Class == iris_val_to_predict, 1, 0)

#Define values for binary indicators for the predictions( using medians of numeric values )
iris_sepal_length_binary_val <- median(iris_set$Sepal_Length)
iris_sepal_width_binary_val <- median(iris_set$Sepal_Width)
iris_petal_length_binary_val <- median(iris_set$Petal_Length)
iris_petal_width_binary_val <- median(iris_set$Petal_Width)

#Recreate the columns used for the prediction as binary indicators
iris_set$Sepal_Length_calc <- ifelse(iris_set$Sepal_Length > iris_sepal_length_binary_val, 1, 0)
iris_set$Sepal_Width_calc <- ifelse(iris_set$Sepal_Width > iris_sepal_width_binary_val, 1, 0)
iris_set$Petal_Length_calc <- ifelse(iris_set$Petal_Length > iris_petal_length_binary_val, 1, 0)
iris_set$Petal_Width_calc <- ifelse(iris_set$Petal_Width > iris_petal_width_binary_val, 1, 0)

#Set the index for the columns to use for the prediction
iris_cols_for_prediction <- 7:10

#Make the portion of the training set P%
iris_training_size <- floor(training_percent * nrow(iris_set))

#define which indices will be used for the training set
iris_train_indices <- sample(seq_len(nrow(iris_set)), size = iris_training_size)

#Split the data into the training and testing set
iris_train_set <- iris_set[iris_train_indices, ]
iris_test_set <- iris_set[-iris_train_indices, ]

#Calculate the models attribute weights
iris_weights <- get_winnow2_weights(
  training_data_set = iris_train_set
  , cols_for_prediction = iris_cols_for_prediction
  , col_to_pred = 6
)

#Use the model to predict what the values are
iris_test_set$Predictions <- predict_winnow2(
  testing_data_set = iris_test_set
  , cols_for_prediction = iris_cols_for_prediction
  , col_to_pred = 6
  , col_weights = iris_weights
)

#See % of successful predictions
length( which(iris_test_set$Predict == iris_test_set$Predictions) ) / nrow(iris_test_set)

#Clean up all variables used in the iris analysis
iris_list <- NULL
iris_list <- which( grepl("iris", ls(), fixed = TRUE) )
rm(list = ls()[iris_list])

#

#### WINNOW: ANALYZE GLASS DATA ####

#Analyze the data
glass_summary <- summary(glass_set)
glass_summary

#Set the value that we want to predict
glass_val_to_predict <- 5

#Define the column to predict
glass_set$Predict <- ifelse(glass_set$Type == glass_val_to_predict, 1, 0)

#Define values for binary indicators for the predictions( using medians of numeric values )
glass_ref_index_binary_val <- median(glass_set$Refractice_Index)
glass_sodium_binary_val <- median(glass_set$Sodium)
glass_magnesium_binary_val <- median(glass_set$Magnesium)
glass_aluminum_binary_val <- median(glass_set$Aluminum)
glass_silicon_binary_val <- median(glass_set$Silicon)
glass_potassium_binary_val <- median(glass_set$Potassium)
glass_calcium_binary_val <- median(glass_set$Calcium)
glass_barium_binary_val <- mean(glass_set$Barium) #median was zero
glass_iron_binary_val <- mean(glass_set$Iron) #median was zero

#Recreate the columns used for the prediction
glass_set$Refractice_Index_calc <- ifelse(glass_set$Refractice_Index > glass_ref_index_binary_val, 1, 0)
glass_set$Sodium_calc <- ifelse(glass_set$Sodium > glass_sodium_binary_val, 1, 0)
glass_set$Magnesium_calc <- ifelse(glass_set$Magnesium > glass_magnesium_binary_val, 1, 0)
glass_set$Aluminum_calc <- ifelse(glass_set$Aluminum > glass_aluminum_binary_val, 1, 0)
glass_set$Silicon_calc <- ifelse(glass_set$Silicon > glass_silicon_binary_val, 1, 0)
glass_set$Potassium_calc <- ifelse(glass_set$Potassium > glass_potassium_binary_val, 1, 0)
glass_set$Calcium_calc <- ifelse(glass_set$Calcium > glass_calcium_binary_val, 1, 0)
glass_set$Barium_calc <- ifelse(glass_set$Barium > glass_barium_binary_val, 1, 0)
glass_set$Iron_calc <- ifelse(glass_set$Iron > glass_iron_binary_val, 1, 0)

#Set the index for the columns to use for the prediction
glass_cols_for_prediction <- 13:21

#Make the portion of the training set P%
glass_training_size <- floor(training_percent * nrow(glass_set))

#define which indices will be used for the training set
glass_train_indices <- sample(seq_len(nrow(glass_set)), size = glass_training_size)

#Split the data into the training and testing set
glass_train_set <- glass_set[glass_train_indices, ]
glass_test_set <- glass_set[-glass_train_indices, ]

#Calculate the models attribute weights
glass_weights <- get_winnow2_weights(
  training_data_set = glass_train_set
  , cols_for_prediction = glass_cols_for_prediction
  , col_to_pred = 12
)

#Use the model to predict what the values are
glass_test_set$Predictions <- predict_winnow2(
  testing_data_set = glass_test_set
  , cols_for_prediction = glass_cols_for_prediction
  , col_to_pred = 12
  , col_weights = glass_weights
)

#See % of successful predictions
length( which(glass_test_set$Predict == glass_test_set$Predictions) ) / nrow(glass_test_set)

#Clean up all variables used in the glass analysis
glass_list <- NULL
glass_list <- which( grepl("glass", ls(), fixed = TRUE) )
rm(list = ls()[glass_list])

#


#### WINNOW: ANALYZE vote DATA ####

#Analyze the data
summary(vote_set)

#Set the value that we want to predict
glass_val_to_predict <- 5

#Define the column to predict
vote_set$Predict <- ifelse(vote_set$Type == glass_val_to_predict, 1, 0)

#Define values for binary indicators for the predictions( using medians of numeric values )
glass_ref_index_binary_val <- median(vote_set$Refractice_Index)
glass_sodium_binary_val <- median(vote_set$Sodium)
glass_magnesium_binary_val <- median(vote_set$Magnesium)
glass_aluminum_binary_val <- median(vote_set$Aluminum)
glass_silicon_binary_val <- median(vote_set$Silicon)
glass_potassium_binary_val <- median(vote_set$Potassium)
glass_calcium_binary_val <- median(vote_set$Calcium)
glass_barium_binary_val <- mean(vote_set$Barium) #median was zero
glass_iron_binary_val <- mean(vote_set$Iron) #median was zero

#Recreate the columns used for the prediction
vote_set$Refractice_Index_calc <- ifelse(vote_set$Refractice_Index > glass_ref_index_binary_val, 1, 0)
vote_set$Sodium_calc <- ifelse(vote_set$Sodium > glass_sodium_binary_val, 1, 0)
vote_set$Magnesium_calc <- ifelse(vote_set$Magnesium > glass_magnesium_binary_val, 1, 0)
vote_set$Aluminum_calc <- ifelse(vote_set$Aluminum > glass_aluminum_binary_val, 1, 0)
vote_set$Silicon_calc <- ifelse(vote_set$Silicon > glass_silicon_binary_val, 1, 0)
vote_set$Potassium_calc <- ifelse(vote_set$Potassium > glass_potassium_binary_val, 1, 0)
vote_set$Calcium_calc <- ifelse(vote_set$Calcium > glass_calcium_binary_val, 1, 0)
vote_set$Barium_calc <- ifelse(vote_set$Barium > glass_barium_binary_val, 1, 0)
vote_set$Iron_calc <- ifelse(vote_set$Iron > glass_iron_binary_val, 1, 0)

#Set the index for the columns to use for the prediction
glass_cols_for_prediction <- 13:21

#Make the portion of the training set P%
glass_training_size <- floor(training_percent * nrow(vote_set))

#define which indices will be used for the training set
glass_train_indices <- sample(seq_len(nrow(vote_set)), size = glass_training_size)

#Split the data into the training and testing set
glass_train_set <- vote_set[glass_train_indices, ]
glass_test_set <- vote_set[-glass_train_indices, ]

#Calculate the models attribute weights
glass_weights <- get_winnow2_weights(
  training_data_set = glass_train_set
  , cols_for_prediction = glass_cols_for_prediction
  , col_to_pred = 12
)

#Use the model to predict what the values are
glass_test_set$Predictions <- predict_winnow2(
  testing_data_set = glass_test_set
  , cols_for_prediction = glass_cols_for_prediction
  , col_to_pred = 12
  , col_weights = glass_weights
)

#See % of successful predictions
length( which(glass_test_set$Predict == glass_test_set$Predictions) ) / nrow(glass_test_set)

#Clean up all variables used in the glass analysis
glass_list <- NULL
glass_list <- which( grepl("glass", ls(), fixed = TRUE) )
rm(list = ls()[glass_list])

#














