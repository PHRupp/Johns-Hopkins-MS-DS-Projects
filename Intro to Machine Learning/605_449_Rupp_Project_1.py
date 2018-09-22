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
training_percent = 0.78

#Seed makes the results reproducible
random.seed(23)

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
			
			#print(col_weights)
  
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

#Build the probabilities for Naive Bayes
#would be better to build with graphical structure
def get_naive_bayes_probailities(
  training_data_set			#pandas dataframe
  , cols_for_prediction 	#columns must have values 0 or 1
  , col_to_pred 			#column must have values 0 or 1
):

	#Define the columns that will be used in the model
	columns = [training_data_set.columns[col_to_pred]] + list(training_data_set.columns[cols_for_prediction])

	#Create the probility table
	prob_table = pandas.DataFrame(columns = columns)
	
	#Initialize a table with 4 records of probabilities for all relevant columns
	prob_table.loc[0] = [0] * len(columns)
	prob_table.loc[1] = [0] * len(columns)
	prob_table.loc[2] = [0] * len(columns)
	prob_table.loc[3] = [0] * len(columns)
	
	#Get list of rows that match
	set_1 = training_data_set.iloc[:,col_to_pred] == 1
	set_0 = training_data_set.iloc[:,col_to_pred] == 0
	predict_set_1 = numpy.concatenate( numpy.where( set_1 )).ravel().tolist()
	predict_set_0 = numpy.concatenate( numpy.where( set_0 )).ravel().tolist()
	
	#Probability of predicted column
	prob_table.iloc[0,0] = len(predict_set_0) / len(training_data_set)
	prob_table.iloc[1,0] = len(predict_set_0) / len(training_data_set)
	prob_table.iloc[2,0] = len(predict_set_1) / len(training_data_set)
	prob_table.iloc[3,0] = len(predict_set_1) / len(training_data_set)
	
	#Go through all predictor columns and get probabilities and store into probability table
	for col_index in range(0, len(cols_for_prediction)):
		
		#Get the column that we are doing the probability
		pred_col = cols_for_prediction[col_index]
		
		#get the set of those that are '0' for both prediction and predictor
		use = [a and b for a, b in zip(list(set_0), list(training_data_set.iloc[:,pred_col] == 0))]
		prob_table.iloc[0, col_index+1] = sum(use) / len(predict_set_0)
		
		#get the set of those that are '0' for prediction and '1' for predictor
		use = [a and b for a, b in zip(list(set_0), list(training_data_set.iloc[:,pred_col] == 1))]
		prob_table.iloc[1, col_index+1] = sum(use) / len(predict_set_0)
		
		#get the set of those that are '1' for prediction and '0' for predictor
		use = [a and b for a, b in zip(list(set_1), list(training_data_set.iloc[:,pred_col] == 0))]
		prob_table.iloc[2, col_index+1] = sum(use) / len(predict_set_1)
		
		#get the set of those that are '1' for both prediction and predictor
		use = [a and b for a, b in zip(list(set_1), list(training_data_set.iloc[:,pred_col] == 1))]
		prob_table.iloc[3, col_index+1] = sum(use) / len(predict_set_1)

	return( prob_table )

	
#Use a simple binary Naive Bayes to predict a particular classification
def predict_naive_bayes(
  testing_data_set		#pandas dataframe
  , cols_for_prediction #columns must have values 0 or 1
  , col_to_pred 		#column must have values 0 or 1
  , prob_table			#probability table
):

	#Predictions for all records
	predictions = []
	
	#Predict the value for all records
	for row_index in range(0, len(testing_data_set) ):
	
		#Product of probabilities starting with the probability of the chosen outputs
		prob_0 = prob_table.iloc[0,0]
		prob_1 = prob_table.iloc[2,0]
		
		#For all predictors, multiply their probability within each set
		for col_index in range(0, len(cols_for_prediction)):
	
			#Find the correct probability for the given predictor value and multiply it
			if( testing_data_set.iloc[row_index, cols_for_prediction[col_index]] == 0 ):
				prob_0 *= prob_table.iloc[0,col_index]
				prob_1 *= prob_table.iloc[2,col_index]
			else:
				prob_0 *= prob_table.iloc[1,col_index]
				prob_1 *= prob_table.iloc[3,col_index]
		
		#Predict the value with the highest probabilistic value
		if( prob_0 > prob_1 ):
			predictions.append(0)
		else:
			predictions.append(1)

	return(predictions)
#

#### PREPARE IRIS DATA ####

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


#### PREDICT IRIS DATA ####


#Calculate the models attribute weights
iris_weights = get_winnow2_weights(
  training_data_set = iris_train_set
  , cols_for_prediction = iris_cols_for_prediction
  , col_to_pred = iris_col_to_predict
)

#print(iris_weights)

#Use the model to predict what the values are
iris_test_set['Predictions_W2'] = predict_winnow2(
  testing_data_set = iris_test_set
  , cols_for_prediction = iris_cols_for_prediction
  , col_to_pred = iris_col_to_predict
  , col_weights = iris_weights
)

#Calculate the models probabilities
iris_prop_table = get_naive_bayes_probailities(
  training_data_set = iris_train_set
  , cols_for_prediction = iris_cols_for_prediction
  , col_to_pred = iris_col_to_predict
)

#print(iris_prop_table)

#Use the model to predict what the values are
iris_test_set['Predictions_NB'] = predict_naive_bayes(
  testing_data_set = iris_test_set
  , cols_for_prediction = iris_cols_for_prediction
  , col_to_pred = iris_col_to_predict
  , prob_table = iris_prop_table
)


#See % of successful predictions
print( "\r\nSuccess Rates of Predictions" )
print( "iris Winnow2: " + str( numpy.sum( iris_test_set.Predictions_W2 == iris_test_set.Predict ) / len(iris_test_set) ))
print( "iris Naive Bayes: " + str( numpy.sum( iris_test_set.Predictions_NB == iris_test_set.Predict ) / len(iris_test_set) ))


#### PREPARE GLASS DATA ####


#Set the value that we want to predict
glass_val_to_predict = 5

#Define the column to predict
glass_set['Predict'] = numpy.where(glass_set.Type == glass_val_to_predict, 1, 0)

#Define values for binary indicators for the predictions( using medians of numeric values )
glass_ref_index_binary_val = numpy.median(glass_set.Refractice_Index)
glass_sodium_binary_val = numpy.median(glass_set.Sodium)
glass_magnesium_binary_val = numpy.median(glass_set.Magnesium)
glass_aluminum_binary_val = numpy.median(glass_set.Aluminum)
glass_silicon_binary_val = numpy.median(glass_set.Silicon)
glass_potassium_binary_val = numpy.median(glass_set.Potassium)
glass_calcium_binary_val = numpy.median(glass_set.Calcium)
glass_barium_binary_val = numpy.mean(glass_set.Barium) #median was zero
glass_iron_binary_val = numpy.mean(glass_set.Iron) #median was zero

#Recreate the columns used for the prediction
glass_set['Refractice_Index_calc'] = numpy.where(glass_set.Refractice_Index > glass_ref_index_binary_val, 1, 0)
glass_set['Sodium_calc'] = numpy.where(glass_set.Sodium > glass_sodium_binary_val, 1, 0)
glass_set['Magnesium_calc'] = numpy.where(glass_set.Magnesium > glass_magnesium_binary_val, 1, 0)
glass_set['Aluminum_calc'] = numpy.where(glass_set.Aluminum > glass_aluminum_binary_val, 1, 0)
glass_set['Silicon_calc'] = numpy.where(glass_set.Silicon > glass_silicon_binary_val, 1, 0)
glass_set['Potassium_calc'] = numpy.where(glass_set.Potassium > glass_potassium_binary_val, 1, 0)
glass_set['Calcium_calc'] = numpy.where(glass_set.Calcium > glass_calcium_binary_val, 1, 0)
glass_set['Barium_calc'] = numpy.where(glass_set.Barium > glass_barium_binary_val, 1, 0)
glass_set['Iron_calc'] = numpy.where(glass_set.Iron > glass_iron_binary_val, 1, 0)

#Set the index for the columns to use for the prediction
glass_cols_for_prediction = numpy.arange(12,21)

#Set the index for the column that was predicted
glass_col_to_predict = 11

#Make the portion of the training set P%
glass_training_size = math.floor(training_percent * len(glass_set))

#Define which indices will be used for the training set
glass_indices = list( range( len( glass_set )))
glass_train_indices = random.sample(glass_indices, glass_training_size)
glass_test_indices = ~glass_set.index.isin( glass_train_indices )

#Split the data into the training and testing set
glass_train_set = glass_set.iloc[glass_train_indices, ]
glass_test_set = glass_set.iloc[glass_test_indices, ]


#### PREDICT GLASS DATA ####


#Calculate the models attribute weights
glass_weights = get_winnow2_weights(
  training_data_set = glass_train_set
  , cols_for_prediction = glass_cols_for_prediction
  , col_to_pred = glass_col_to_predict
)

#print(glass_weights)

#Use the model to predict what the values are
glass_test_set['Predictions_W2'] = predict_winnow2(
  testing_data_set = glass_test_set
  , cols_for_prediction = glass_cols_for_prediction
  , col_to_pred = glass_col_to_predict
  , col_weights = glass_weights
)

#Calculate the models probabilities
glass_prop_table = get_naive_bayes_probailities(
  training_data_set = glass_train_set
  , cols_for_prediction = glass_cols_for_prediction
  , col_to_pred = glass_col_to_predict
)

#print(glass_prop_table)

#Use the model to predict what the values are
glass_test_set['Predictions_NB'] = predict_naive_bayes(
  testing_data_set = glass_test_set
  , cols_for_prediction = glass_cols_for_prediction
  , col_to_pred = glass_col_to_predict
  , prob_table = glass_prop_table
)


#See % of successful predictions
print( "\r\nSuccess Rates of Predictions" )
print( "glass Winnow2: " + str( numpy.sum( glass_test_set.Predictions_W2 == glass_test_set.Predict ) / len(glass_test_set) ))
print( "glass Naive Bayes: " + str( numpy.sum( glass_test_set.Predictions_NB == glass_test_set.Predict ) / len(glass_test_set) ))



#### PREPARE VOTE DATA ####


#Set the value that we want to predict
vote_val_to_predict = 'democrat'

#Define the column to predict
vote_set['Predict'] = numpy.where(vote_set.Class == vote_val_to_predict, 1, 0)

#Recreate the columns used for the prediction
vote_maj_val = 'y'
vote_set['Handicapped_Infants_calc'] = numpy.where(vote_set.Handicapped_Infants == vote_maj_val, 1, 0)
vote_set['Water_Project_Cost_Sharing_calc'] = numpy.where(vote_set.Water_Project_Cost_Sharing == vote_maj_val, 1, 0)
vote_set['Adoption_Of_Budget_Resolution_calc'] = numpy.where(vote_set.Adoption_Of_Budget_Resolution == vote_maj_val, 1, 0)
vote_set['Physician_Fee_Freeze_calc'] = numpy.where(vote_set.Physician_Fee_Freeze == vote_maj_val, 1, 0)
vote_set['El_Salvador_Aid_calc'] = numpy.where(vote_set.El_Salvador_Aid == vote_maj_val, 1, 0)
vote_set['Religious_Groups_In_Schools_calc'] = numpy.where(vote_set.Religious_Groups_In_Schools == vote_maj_val, 1, 0)
vote_set['Anti_Satellite_Test_Ban_calc'] = numpy.where(vote_set.Anti_Satellite_Test_Ban == vote_maj_val, 1, 0)
vote_set['Aid_To_Nicaraguan_Contras_calc'] = numpy.where(vote_set.Aid_To_Nicaraguan_Contras == vote_maj_val, 1, 0)
vote_set['Immigration_calc'] = numpy.where(vote_set.Immigration == vote_maj_val, 1, 0)
vote_set['Synfuels_Corporation_Cutback_calc'] = numpy.where(vote_set.Synfuels_Corporation_Cutback == vote_maj_val, 1, 0)
vote_set['Education_Spending_calc'] = numpy.where(vote_set.Education_Spending == vote_maj_val, 1, 0)
vote_set['Superfund_Right_To_Sue_calc'] = numpy.where(vote_set.Superfund_Right_To_Sue == vote_maj_val, 1, 0)
vote_set['Crime_calc'] = numpy.where(vote_set.Crime == vote_maj_val, 1, 0)
vote_set['Duty_Free_Exports_calc'] = numpy.where(vote_set.Duty_Free_Exports == vote_maj_val, 1, 0)
vote_set['Export_Administration_Act_South_Africa_calc'] = numpy.where(vote_set.Export_Administration_Act_South_Africa == vote_maj_val, 1, 0)

#Set the index for the columns to use for the prediction
vote_cols_for_prediction = numpy.arange(18,33)

#Set the index for the column that was predicted
vote_col_to_predict = 17

#Make the portion of the training set P%
vote_training_size = math.floor(training_percent * len(vote_set))

#define which indices will be used for the training set
vote_indices = list( range( len( vote_set )))
vote_train_indices = random.sample(vote_indices, vote_training_size)
vote_test_indices = ~vote_set.index.isin( vote_train_indices )

#Split the data into the training and testing set
vote_train_set = vote_set.iloc[vote_train_indices, ]
vote_test_set = vote_set.iloc[vote_test_indices, ]


#### PREDICT VOTE DATA ####


#Calculate the models attribute weights
vote_weights = get_winnow2_weights(
  training_data_set = vote_train_set
  , cols_for_prediction = vote_cols_for_prediction
  , col_to_pred = vote_col_to_predict
)

#print(vote_weights)

#Use the model to predict what the values are
vote_test_set['Predictions_W2'] = predict_winnow2(
  testing_data_set = vote_test_set
  , cols_for_prediction = vote_cols_for_prediction
  , col_to_pred = vote_col_to_predict
  , col_weights = vote_weights
)

#Calculate the models probabilities
vote_prop_table = get_naive_bayes_probailities(
  training_data_set = vote_train_set
  , cols_for_prediction = vote_cols_for_prediction
  , col_to_pred = vote_col_to_predict
)

#print(vote_prop_table)

#Use the model to predict what the values are
vote_test_set['Predictions_NB'] = predict_naive_bayes(
  testing_data_set = vote_test_set
  , cols_for_prediction = vote_cols_for_prediction
  , col_to_pred = vote_col_to_predict
  , prob_table = vote_prop_table
)


#See % of successful predictions
print( "\r\nSuccess Rates of Predictions" )
print( "vote Winnow2: " + str( numpy.sum( vote_test_set.Predictions_W2 == vote_test_set.Predict ) / len(vote_test_set) ))
print( "vote Naive Bayes: " + str( numpy.sum( vote_test_set.Predictions_NB == vote_test_set.Predict ) / len(vote_test_set) ))



































































