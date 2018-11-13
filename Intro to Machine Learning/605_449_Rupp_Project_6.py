"""
Course: 605.449 - Introduction to Machine Learning
Project: #6
Due: Sun Nov 18 23:59:59 2018
@author: Patrick H. Rupp

NOTES: First Column must be class while others are numeric
"""

#### DEFINE GLOBAL REQUIREMENTS ####

import pandas
import numpy
import random
import sys
import math


#Define the % of training set to use
num_sets = 5

#Get the arguments from command prompt (given as directory containing files)
data_file = sys.argv[1]

#Read the file into a data set
data_set = pandas.read_csv( data_file )

#list of column names that make up the attribute names
attr_cols = [name for name in data_set.columns[1:len(data_set.columns)] ]
class_col = data_set.columns[0]



######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
##                                                   NEURAL NETWORK                                                 ##
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################


class NN_Node:
    
     #
    def __init__(
        self
        , num_array              
        , class_array           
        , node_id        #Number of hidden nodes left to create
        , learning_rate         #How fast the weights are changed each iteration
    ):

        self.node_id = node_id
        
        #If there are hidden nodes, then this will point to the next one
        self.next_node = None
        
        #Initalize the weights as small values
        self.weights = numpy.zeros( (num_array.shape[1], num_array.shape[1]) ) + 1e-3





#
class LogisticRegression:
    
    #
    def __init__(
        self
        , data_set              #Pandas Data Frame of training set
        , attr_cols             #List of attribute column names
        , class_col             #Column name of the Class of the data
        , num_hidden_layers = 0 #Number of hidden layers
        , learning_rate = 1e-6  #How fast the weights are changed each iteration
        , iterations = 1000     #Number of iterations the model will be adjusted
        , add_intercept = True  #Adds column of 1's so that the weights are by themselves
        , show_loss_step = 1e+9 #On this iteration, the loss will be printed
    ):
        #define the addition of the intercept
        self.add_intercept = add_intercept
        
        #Initialize the data frame as arrays for ease of calculations
        num_array = numpy.array( data_set[attr_cols] )
        class_array = numpy.array( data_set[class_col] )
        
        #Make a numerical array from the data frame with intercept if indicated
        if( self.add_intercept ):
            num_array = numpy.concatenate( (numpy.ones((num_array.shape[0], 1)), num_array), axis=1)
        
        #Initalize the weights as 0's
        self.weights = numpy.zeros( (num_array.shape[1], 1) )
        
        #Iterate as many times as indicated to finalize the chaning of the weights
        for iter_num in range(iterations):
            
            #Adjust the rates based on the learning rate
            self.weights -= learning_rate * self.error_gradient(num_array, class_array)
            
            #For the designated steps, display the loss
            if( iter_num % show_loss_step == 0 ):
                print( 'Iter #' + str(iter_num) )
                print( ''.join([' '] * 2) + 'loss: ' + str(self.loss_function( num_array, class_array )) )
        
        #End function
        return


    #Logistic Activation Function 'S' curve
    def sigmoid_curve(
        self
        , x_vals    #Some numeric value(s) either stand alone or as Numpy Array
    ):
        #End function
        return( 1 / (1 + numpy.exp(-x_vals)) )
    
    
    #1st derivative of the Logistic Activation Function
    def sigmoid_curve_d1(
        self
        , x_vals    #Some numeric value(s) either stand alone or as Numpy Array
    ):
        
        #Intermediate calculation step
        temp = numpy.exp(-x_vals)
        
        #End function
        return( temp / (1 + temp)**2 )
    
  
    #
    def predict_is_class(
        self
        , data_set          #Pandas Data Frame of training set
        , attr_cols         #List of attribute column name
        , threshold = 0.5   #Limit where class boundaries are separated
    ):
        
        #Initialize the data frame as arrays for ease of calculations
        num_array = numpy.array( data_set[attr_cols] )
        
        #Make a numerical array from the data frame with intercept if indicated
        if( self.add_intercept ):
            num_array = numpy.concatenate( (numpy.ones((num_array.shape[0], 1)), num_array), axis=1)
            
        #Run sigmoid curve using the weighted_set as the summation of all the 
        #attributes (0-1) with some given weights
        sigmoid_set = self.sigmoid_curve( numpy.dot( num_array, self.weights ) )
        
        #If value above threshold return 1 else 0
        output = numpy.where( sigmoid_set >= threshold, 1, 0 )
        output = [output[x][0] for x in range(output.shape[0])]
    
        #End function
        return output

######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
##                                                       ADALINE                                                    ##
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################



#
class Adaline:
    
    #
    def __init__(
        self
        , data_set              #Pandas Data Frame of training set
        , attr_cols             #List of attribute column names
        , class_col             #Column name of the Class of the data
        , learning_rate = 1e-6  #How fast the weights are changed each iteration
        , iterations = 1000     #Number of iterations the model will be adjusted
        , add_intercept = True  #Adds column of 1's so that the weights are by themselves
        , show_error_step = 10  #On this iteration, the loss will be printed
    ):
        #define the addition of the intercept
        self.add_intercept = add_intercept
        
        #Initialize the data frame as arrays for ease of calculations
        num_array = numpy.array( data_set[attr_cols] )
        class_array = numpy.array( data_set[class_col] )
        
        #Make a numerical array from the data frame with intercept if indicated
        if( self.add_intercept ):
            num_array = numpy.concatenate( (numpy.ones((num_array.shape[0], 1)), num_array), axis=1)
        
        #Initalize the weights as 0's
        self.weights = numpy.array( [0.] * num_array.shape[1] )
        
        #Iterate as many times as indicated to finalize the chaning of the weights
        for iter_num in range(iterations):
            
            #Multiply all the features for each row by the feature weights
            calculated_values = numpy.dot( num_array, self.weights )
            
            #Determine the errors between the calculated values and the actual values
            calculated_error = class_array - calculated_values
            
            #Multiply the errors by the values in each feature set to have the
            #incorrect weights have a higher error rate so that the change
            #will be greater
            self.weights += learning_rate * numpy.dot( num_array.T, calculated_error )
            
            #Calculate the mean squared error for the algorithm
            mean_squared_error = sum( calculated_error**2 ) / calculated_error.shape[0]
            
            #For the designated steps, display the loss
            if( iter_num % show_error_step == 0 ):
                print( 'Iter #' + str(iter_num) + '\tMSE: ' + str(mean_squared_error) )
        
        #End function
        return


    #
    def predict_is_class(
        self
        , data_set          #Pandas Data Frame of training set
        , attr_cols         #List of attribute column name
        , threshold = 0.0   #Limit where class boundaries are separated
    ):
        
        #Initialize the data frame as arrays for ease of calculations
        num_array = numpy.array( data_set[attr_cols] )
        
        #Make a numerical array from the data frame with intercept if indicated
        if( self.add_intercept ):
            num_array = numpy.concatenate( (numpy.ones((num_array.shape[0], 1)), num_array), axis=1)
            
        #Calculate the values and if they are above 0, then make 1 else -1
        output = numpy.where( numpy.dot( num_array, self.weights ) >= 0.0, 1, 0 )
        output = [output[x] for x in range(output.shape[0])]
    
        #End function
        return output




######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
##                                             CLASSIFIER ACCURACY                                                  ##
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################



#Determine the accuracy of the Classifier accuracy
def classifierAccuracy(
        actuals             #list of the actual values
        , predictions       #list of the predicted values
):
    #Match each predicted value with the actuals to determine how many were correct
    correct = numpy.where( [actuals[ind] == predictions[ind] for ind in range(len(predictions))] )[0]
    
    #Get the number of correct predictions
    correct_num = len([x for x in correct])
    
    #Return the percent correct of all predicted values
    return(correct_num / len(predictions))



######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################



#break a list into a number of lists
def split_lists(my_list, num_lists):
    new_list = []
    
    split_size = len(my_list) / num_lists
    
    for i in range(num_lists):
        new_list.append( my_list[int(round(i * split_size)):int(round((i+1) * split_size))] )
        
    return new_list

#define which indices will be used for the training set
data_set_indices = list( range( len( data_set )))
data_set_indices = random.sample(data_set_indices, len(data_set_indices))

#Split the indices into the defined number of sets
set_manager = split_lists(data_set_indices, num_sets)


#
results = []
predicts = []
real_vals = []

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
    
    #Depending on the type of data input, run the particular model that the 
    #data was staged for
    if('in_categories' in data_file):
        
        #Build the model used for the prediction
        model = NaiveBayes(training_set, attr_cols, class_col)
        
        #Predict the classes that the test values fall under
        predictions = model.predict_class(testing_set, attr_cols)
        
        model.print_model()
        print('\n\n')
        
    elif('in_standard' in data_file):
    
        #Build the model used for the prediction
        model = LogisticRegression(
                    training_set
                    , attr_cols
                    , class_col
                    , learning_rate = 1e-6
                    , iterations = 1000
                    , show_loss_step = 500
        )
        
        #Predict the classes that the test values fall under
        predictions = model.predict_is_class(
                    testing_set
                    , attr_cols
                    , threshold = 0.5
        )
    
    elif('in_normalized' in data_file):
     
        #Build the model used for the prediction
        model = Adaline(
                    training_set
                    , attr_cols
                    , class_col
                    , learning_rate = 1e-6
                    , iterations = 50000
                    , show_error_step = 5000
        )
        
        #Predict the classes that the test values fall under
        predictions = model.predict_is_class(
                    testing_set
                    , attr_cols
                    , threshold = 0.0
        )
  
    
    #Take the actual classes that the data belongs to
    actuals = [x for x in testing_set[class_col] ]
    results.append( classifierAccuracy(actuals, predictions) )
    
    #Store the actual and predicted values for analysis later.
    predicts.append(predictions)
    real_vals.append(actuals)
    
print(results)    

#Print the data set out with the actual vs. predicted columns
indices = [item for sub in set_manager for item in sub]
output = data_set.iloc[indices].copy() 
output['predict'] = [item for sub in predicts for item in sub]
output['actual'] = [item for sub in real_vals for item in sub]
output.to_csv( data_file.replace('in', 'out') )





"""
This program takes in a file and determines which type of model to run
based on specific strings in the file name

'categories'    = Naive Bayes

'normalized'    = Adaline

'standard'      = Logistic Regression


Files with 'in' means that they are staged to be fed into the program

Files with 'out' means that they are produced by the program
"""





















