# -*- coding: utf-8 -*-
"""
Course: 605.449 - Introduction to Machine Learning
Project: #2
Due: Sun Oct 7 23:59:59 2018
@author: Patrick H. Rupp

NOTES: Last Column must be class while others are numeric
"""

#### DEFINE GLOBAL REQUIREMENTS ####

import pandas
import numpy
import random
import sys
import math

##################### IMPORTANT
#Must be set properly to perform correct analysis
run_classifier = False
##################### IMPORTANT

#Specify the number of neighbors used in the calculation
number_neighbors = 8

#Define the % of training set to use
set_percent = 0.20

#Get the arguments from command prompt (given as directory containing files)
data_file = sys.argv[1]

#Read the file into a data set
data_set = pandas.read_csv( data_file )

#Make the portion of the training set P%
set_size = math.floor(set_percent * len(data_set))

#Returns list of indices for closest neighbors
def getClosestNeighborIndices(
        record          #1 record from panda data set to find the neighbors for
        , data_set      #panda data set where neighbors will reside
        , calc_col      #Column indices to use for the calculations
        , num_neighbors #number of neighbors to find for record
):
    #Get all euclidean distances of the data set from the given record
    neighbor_distances = [ numpy.linalg.norm( data_set.iloc[row_index, calc_col] - record[calc_col] ) for row_index in range(len(data_set)) ]
      
    #Get the top K indices of the sorted list by smallest distance from record
    neighbor_indices = numpy.argsort(neighbor_distances)[:num_neighbors]
    
    #Return the top indices as a list
    return( [ neighbor_indices[x] for x in range(len(neighbor_indices)) ] )
    

#K nearest neighbors classifier
def knn_classifier(
        training_set    #Data for training the model
        , testing_set   #Data to predict the classifier
        , num_neighbors #number of neighbors to use in classifier
        , use_cnn_reduce = False
):
    #Initialize list of predicted classes for test set
    predictions = []
    
    #Make a reduced training set based on Condensed Nearest Neighbors (CNN) data reduction
    reduced_training_set = training_set.copy()
    
    #true index accounting for outlier deletions from training set
    row_index = 0
    
    #Only reduce data if useful
    if( use_cnn_reduce ):

        #Go through all training records to determine if they are outliers
        for row_num in range(len(reduced_training_set)):
            
            #Only continue to remove outliers if within bounds
            if( row_index < len(reduced_training_set) ):
            
                #Get the indices for the top N closest neighbors
                neighbor_indices = getClosestNeighborIndices(
                        reduced_training_set.iloc[row_index]
                        , reduced_training_set
                        , [x for x in range(len(reduced_training_set.columns)-1)]
                        , num_neighbors+1 #it will catch itself so we must manually remove
                        )
            
                #Manually remove first index because it's the same record (distance = 0)
                del neighbor_indices[0]
                
                #Get the number of counts for each class
                class_count = reduced_training_set.iloc[neighbor_indices].groupby( reduced_training_set.columns[-1] ).agg('count').iloc[:,0]
                
                #Predict the class based on majority by taking last value in ascending sorted array
                predicted_class = class_count[ numpy.argsort( class_count ) ].index[-1]
                
                #If correctly predicted, drop the record since it is outlier otherwise increment next index
                if( reduced_training_set.iloc[row_index, -1] != predicted_class ):
                    reduced_training_set.drop( reduced_training_set.index[row_index], inplace = True )
                else:
                    row_index += 1
    
    #Go through all records that need to be classified
    for row_index in range(len(testing_set)):
    
        print(row_index)
        
        #Get the indices for the top N closest neighbors
        neighbor_indices = getClosestNeighborIndices(
                testing_set.iloc[row_index]
                , reduced_training_set
                , [x for x in range(len(reduced_training_set.columns)-1)]
                , num_neighbors
                )
        
        #Get the number of counts for each class
        class_count = reduced_training_set.iloc[neighbor_indices].groupby( reduced_training_set.columns[-1] ).agg('count').iloc[:,0]
        
        #Take the class with the highest count
        predictions.append( class_count[ numpy.argsort( class_count ) ].index[-1] )
        
    #Send back the predictions
    return( predictions )
    

#K nearest neighbors Regressor
def knn_regressor(
        training_set    #Data for training the model
        , test_set      #Data to predict the classifier
        , num_neighbors #number of neighbors to use in classifier
):
    #Initialize list of predicted classes for test set
    predictions = []

    #Go through all records that need to be classified
    for row_index in range(len(testing_set)):
    
        print(row_index)
        
        #Get the indices for the top N closest neighbors
        neighbor_indices = getClosestNeighborIndices(
                testing_set.iloc[row_index]
                , training_set
                , [x for x in range(len(training_set.columns)-1)]
                , num_neighbors
                )
        
        #Get the average of all the neighbors numerical value for prediction
        average_neighbor_pred = numpy.mean( training_set.iloc[neighbor_indices, -1] )
        
        #Add the predicted value to the output list of predicted values
        predictions.append( average_neighbor_pred )
        
    #Send back the predictions
    return( predictions )

#Determine the accuracy of the KNN Classifier accuracy
def classifierAccuracy(
        actual          #list of the actual values
        , predicted     #list of the predicted values
):
    
    #Match each predicted value with the actuals to determine how many were correct
    correct = numpy.where( [actuals[ind] == predictions[ind] for ind in range(len(predictions))] )[0]
    
    #Get the number of correct predictions
    correct_num = len([x for x in correct])
    
    #Return the percent correct of all predicted values
    return(correct_num / len(predictions))

#Determine the accuracy of the KNN Classifier accuracy
def regressorAccuracy(
        actual          #list of the actual values
        , predicted     #list of the predicted values
):
    
    #Mean Squared Error
    total_error = numpy.sum([ (actuals[ind] - predictions[ind])**2 for ind in range(len(predictions)) ])
    
    #Return the percent correct of all predicted values
    return(total_error / len(predictions))





######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################






#Seed makes the results reproducible
random.seed(32)

#define which indices will be used for the training set
data_set_indices = list( range( len( data_set )))
set_1_indices = random.sample(data_set_indices, set_size)

#Remove indices that were taken by previous set so that no set access same points
data_set_indices = [x for x in numpy.where([x not in set_1_indices for x in data_set_indices])[0]]
set_2_indices = random.sample(data_set_indices, set_size)

#Remove indices that were taken by previous set so that no set access same points
data_set_indices = [x for x in numpy.where([x not in set_2_indices for x in data_set_indices])[0]]
set_3_indices = random.sample(data_set_indices, set_size)

#Remove indices that were taken by previous set so that no set access same points
data_set_indices = [x for x in numpy.where([x not in set_3_indices for x in data_set_indices])[0]]
set_4_indices = random.sample(data_set_indices, set_size)

#Remove indices that were taken by previous set so that no set access same points
data_set_indices = [x for x in numpy.where([x not in set_4_indices for x in data_set_indices])[0]]
set_5_indices = random.sample(data_set_indices, set_size)

#Remove indices that were taken by previous set so that no set access same points
data_set_indices = [x for x in numpy.where([x not in set_5_indices for x in data_set_indices])[0]]
set_5_indices = set_5_indices + data_set_indices

#List of all the list of indices
set_manager = [set_1_indices, set_2_indices, set_3_indices, set_4_indices, set_5_indices]

#
results = []

predicts = []

#Run through 5 iterations of rotating training / testing groups
for iteration in range(len(set_manager)):
    
    #Take the first group as the test set
    testing_set_indices = set_manager[0]
    
    #Remove test set from group 
    del set_manager[0]
    
    #Collapse the subsets of indices into 1 large set
    training_set_indices = [ind for subset in set_manager for ind in subset]

    #Split the data into the training and testing set
    training_set = data_set.iloc[training_set_indices, ]
    testing_set = data_set.iloc[testing_set_indices, ]

    #Add first group to end, therefore next cycle will use the next first group as test
    #creating a "rotating" affect on the testing/training activities
    set_manager.append( testing_set_indices )

    #Switch between classifier or regressor
    if( run_classifier ):
        
        #Predict the classes that the test values fall under
        predictions = knn_classifier(training_set, testing_set, number_neighbors)
        
        #Take the actual classes that the data belongs to
        actuals = [x for x in testing_set.iloc[:, -1] ]
        results.append( classifierAccuracy(actuals, predictions) )
        
    else:
        #Predict the value using average of nearest neighbors
        predictions = knn_regressor(training_set, testing_set, number_neighbors)
        actuals = [x for x in testing_set.iloc[:, -1] ]
        results.append( regressorAccuracy(actuals, predictions) )
    
    predicts.append(predictions)
    
#indices = [item for sub in set_manager for item in sub]
#output = data_set.iloc[indices].copy() 
#output['predict'] = [item for sub in predicts for item in sub]
#output.to_csv("B:/Users/pathr/Documents/Education/JHU/DATA/605.449.81.FA18/out_forestfires.csv")


#Write to output
output_file = open("B:/Users/pathr/Documents/Education/JHU/DATA/605.449.81.FA18/output.txt", "w")
output_file.write('From Set: "' + data_file + '" \r\n')
output_file.write( ', '.join( [ str(x) for x in results] ))
output_file.close



######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################






































