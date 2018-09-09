#######################################################################
# Project #1
# Author: Patrick H. Rupp
# Course: 605.449 - Introduction to Machine Learning
# Due: September 9, 2018

#### DEFINE GLOBAL REQUIREMENTS ####

import pandas
import numpy
import math
import random

#Define the % of training set to use
training_percent = 0.7

#Seed makes the results reproducible
random.seed(100)

#### RETRIEVE ALL DATA FROM WEB ####

# Add the source archive url to the sub sets to make full paths
source_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
breast_cancer_url = source_url + "breast-cancer-wisconsin/wdbc.data"
soybean_url = source_url + "soybean/soybean-small.data"
glass_url = source_url + "glass/glass.data"
iris_url = source_url + "iris/iris.data"
vote_url = source_url + "voting-records/house-votes-84.data"

#Define the columns for the data
glass_cols = [
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
]

#retrieve the data from the source
glass_set = pandas.read_csv(
  glass_url
  , names = glass_cols
)

#Define the columns for the data
iris_cols = [
  "Sepal_Length"
  , "Sepal_Width"
  , "Petal_Length"
  , "Petal_Width"
  , "Class"
]

#retrieve the data from the source
iris_set = pandas.read_csv(
  iris_url
  , names = iris_cols
)

#Define the columns for the data
vote_cols = [
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
]

#retrieve the data from the source
vote_set = pandas.read_csv(
  vote_url
  , names = vote_cols
  , na_values = '?'
)

#### DEFINE WINNOW FUNCTION ####

#https://en.wikipedia.org/wiki/Winnow_(algorithm)
#http://www.cs.yale.edu/homes/aspnes/pinewiki/WinnowAlgorithm.html
#http://www.cs.tau.ac.il/~mansour/ml-course-10/scribe4.pdf

#Building the weights using the Winnow2 algorithm
def get_winnow2_weights (
  training_data_set			#pandas dataframe
  , cols_for_prediction 	#columns must have values 0 or 1
  , col_to_pred 			#column must have values 0 or 1
  , bounds = None
  , weight_step = 2
):

	#Set the bounds as number of columns if not specified by user
	if( bounds == None ):
		bounds = len(cols_for_prediction)

	#Initialize the weights
	col_weights = numpy.asarray([1.0] * len(cols_for_prediction))

	#Go through all records and change the weights for the model
	for row_index in range( len(training_data_set) ):
	
		#Calculate the weighted values
		weighted_val = col_weights * numpy.asarray(training_data_set.iloc[row_index, cols_for_prediction])
    
		#Calculate the sum of all weighted values and compare to the bound to make a prediction
		if( sum( weighted_val ) > bounds ): 
			prediction = 1 
		else:
			prediction = 0
    
		#Incorrect predictions will have their weights changed
		if( prediction != training_data_set.iloc[row_index, col_to_pred] ):
		  
			#Determine which values need to have their weights changed
			cols_to_change = training_data_set.iloc[row_index, cols_for_prediction] == 1
		  
			#Promote weights up if 1 otherwise demote the weights
			if( prediction == 1 ):
				col_weights[cols_to_change] = col_weights[cols_to_change] / weight_step 
			else:
				col_weights[cols_to_change] = col_weights[cols_to_change] * weight_step
			
			print(col_weights)
  
	#Send the column weights back
	return(col_weights)


#Using the weights output above, we predict the values
def predict_winnow2(
  testing_data_set
  , cols_for_prediction #columns must have values 0 or 1
  , col_to_pred 		#column must have values 0 or 1
  , col_weights
  , bounds = None
):

	#Set the bounds as number of columns if not specified by user
	if( bounds == None ):
		bounds = len(cols_for_prediction)
		
	#Initialize the predictions
	predictions = numpy.asarray([-1] * len(testing_data_set))
  
	#Go through all records and change the weights for the model
	for row_index in range( len(testing_data_set) ):
	
		#Calculate the weighted values
		weighted_val = col_weights * numpy.asarray(testing_data_set.iloc[row_index, cols_for_prediction])
    
		#Calculate the sum of all weighted values and compare to the bound to make a prediction
		if( sum( weighted_val ) > bounds ): 
			predictions[row_index] = 1 
		else:
			predictions[row_index] = 0
  
	return(predictions)

#

#### WINNOW: ANALYZE IRIS DATA ####

#Set the value that we want to predict
iris_val_to_predict = "Iris-versicolor"

#Define the column to predict
iris_set['Predict'] = numpy.where(iris_set.Class == iris_val_to_predict, 1, 0)

#Define values for binary indicators for the predictions( using numpy.medians of numeric values )
iris_sepal_length_binary_val = numpy.median(iris_set.Sepal_Length)
iris_sepal_width_binary_val = numpy.median(iris_set.Sepal_Width)
iris_petal_length_binary_val = numpy.median(iris_set.Petal_Length)
iris_petal_width_binary_val = numpy.median(iris_set.Petal_Width)

#Recreate the columns used for the prediction as binary indicators
iris_set['Sepal_Length_calc'] = numpy.where(iris_set.Sepal_Length > iris_sepal_length_binary_val, 1, 0)
iris_set['Sepal_Width_calc'] = numpy.where(iris_set.Sepal_Width > iris_sepal_width_binary_val, 1, 0)
iris_set['Petal_Length_calc'] = numpy.where(iris_set.Petal_Length > iris_petal_length_binary_val, 1, 0)
iris_set['Petal_Width_calc'] = numpy.where(iris_set.Petal_Width > iris_petal_width_binary_val, 1, 0)

#Set the index for the columns to use for the prediction
iris_cols_for_prediction = numpy.arange(6,10)

#Set the index for the column that was predicted
iris_col_to_predict = 5

#Make the portion of the training set P%
iris_training_size = math.floor(training_percent * len(iris_set))

#define which indices will be used for the training set
iris_indices = list( range( len( iris_set )))
iris_train_indices = random.sample(iris_indices, iris_training_size)
iris_test_indices = ~iris_set.index.isin( iris_train_indices )

#Split the data into the training and testing set
iris_train_set = iris_set.iloc[iris_train_indices, ]
iris_test_set = iris_set.iloc[iris_test_indices, ]

#Calculate the models attribute weights
iris_weights = get_winnow2_weights(
  training_data_set = iris_train_set
  , cols_for_prediction = iris_cols_for_prediction
  , col_to_pred = iris_col_to_predict
)

print(iris_weights)

#Use the model to predict what the values are
iris_test_set['Predictions'] = predict_winnow2(
  testing_data_set = iris_test_set
  , cols_for_prediction = iris_cols_for_prediction
  , col_to_pred = iris_col_to_predict
  , col_weights = iris_weights
)

#See % of successful predictions
print( numpy.sum( iris_test_set.Predictions == iris_test_set.Predict ) / len(iris_test_set) )











































































