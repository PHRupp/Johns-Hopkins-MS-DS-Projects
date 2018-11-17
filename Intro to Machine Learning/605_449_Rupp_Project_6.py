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


#Logistic Activation Function 'S' curve
def sigmoid_curve(
    x_vals    #Some numeric value(s) either stand alone or as Numpy Array
):
    #End function
    return( 1 / (1 + numpy.exp(-x_vals)) )


#1st derivative of the Logistic Activation Function
def sigmoid_curve_d1(
    x_vals    #Some numeric value(s) either stand alone or as Numpy Array
):
    
    #Intermediate calculation step
    temp = numpy.exp(-x_vals)
    
    #End function
    return( temp / (1 + temp)**2 )


#
def MSE(
    actuals
    , predictions
):
    return( (actuals - predictions)**2 )

#
def MSE_d1(
    actuals
    , predictions
):
    return( 2 * (actuals - predictions) )



class FinalLayer:
    
     #
    def __init__(
        self
        , num_array              
        , class_array           
        , node_id        #Number of hidden nodes left to create
    ):
        
        self.node_id = node_id
        
        #Initalize the weights as small values
        self.weights = numpy.zeros( (num_array.shape[1], 1) ) + 1e-3
        self.output = numpy.zeros( (class_array.shape[0], 1) )
        self.layer_vals = num_array.copy()

        #End function
        return
    
    
    #
    def feed_forward(
        self
        , num_array              
    ):
        #
        self.layer_vals = num_array.copy()
        
        #Calculate the weighted sum to feed into function for the end output
        self.output = sigmoid_curve( num_array.dot( self.weights ))
        
        #End function
        return self.output
  
    
    #
    def back_propagation(
        self
        , class_array
        , learning_rate         #How fast the weights are changed each iteration
    ):
        #number of rows
        N = class_array.shape[0]
        class_array = class_array.reshape( (N, 1) )

        #Get the weighted sum that gets fed into the activation function times elementwise the change in error
        #this is the common values that will be sent back until front of algorithm
        #change from (N,) -> (N,1)
        error_change = MSE_d1(class_array, self.output) * sigmoid_curve_d1( self.output )

        #Application of the chain rule for change in the weights
        weight_change = self.layer_vals.T.dot( error_change )
        
        #get weight change
        self.weights += weight_change * learning_rate
        
        #will help with propagation as well as changes from (N,) -> (N,k)
        #'k' being the number attributes plus a possible intercept values
        error_change = error_change.dot( self.weights.T )
        
        #the weight changes are necessary for avoiding recalculating further within the NN
        return error_change    


class HiddenLayer:
    
     #
    def __init__(
        self
        , num_array              
        , class_array           
        , node_id        #Number of hidden nodes left to create
    ):

        self.node_id = node_id
        
        #Initalize the weights as small values
        self.weights = numpy.zeros( (num_array.shape[1], num_array.shape[1]) ) + 1e-3
        self.layer_vals = num_array.copy()
        
        #If there are hidden nodes, then this will point to the next one
        #otherwise point to the final calculation layer
        if( self.node_id > 1 ):
            self.next_node = HiddenLayer(num_array, class_array, self.node_id - 1)
        else:
            self.next_node = FinalLayer(num_array, class_array, 0)
        
        #End function
        return
    
    
    #
    def feed_forward(
        self
        , num_array              
    ):
        #
        self.layer_vals = num_array.copy()
        
        #Get the new weighted sum values for all the nodes given this matrix multiplication
        #including the activation function step
        next_layer_vals = sigmoid_curve(  self.layer_vals.dot( self.weights ))
        
        #If there are more nodes, run their feed forward, else call output
        if( self.next_node.node_id == 0 ):
            return self.next_node.feed_forward( next_layer_vals )
        else:
            return self.next_node.feed_forward( next_layer_vals )
        
    #
    def back_propagation(
        self
        , class_array
        , learning_rate         #How fast the weights are changed each iteration
    ):

        #getting previous error calculation in terms of (N,k)
        #'k' being the number attributes plus a possible intercept values
        error_change = self.next_node.back_propagation( class_array, learning_rate )
        
        #elementwise multiplication of the weighted sum values entered into the activation function
        error_change = error_change * sigmoid_curve_d1( self.layer_vals.dot( self.weights ))

        #Application of the chain rule for change in the weights
        weight_change = self.layer_vals.T.dot( error_change )
        
        #adjust weights
        self.weights += weight_change * learning_rate
        
        #elementwise multiplication of the weighted sum values entered into the activation function
        error_change = error_change.dot( self.weights.T )
        
        #the weight changes are necessary for avoiding recalculating further within the NN
        return error_change 


#
class NeuralNetwork:
    
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
        , show_error_step = 1e+9#On this iteration, the loss will be printed
    ):
        #define the addition of the intercept
        self.add_intercept = add_intercept
        
        #Initialize the data frame as arrays for ease of calculations
        num_array = numpy.array( data_set[attr_cols] )
        class_array = numpy.array( data_set[class_col] )
        calc_array = numpy.zeros( class_array.shape )
        
        #Make a numerical array from the data frame with intercept if indicated
        if( self.add_intercept ):
            num_array = numpy.concatenate( (numpy.ones((num_array.shape[0], 1)), num_array), axis=1)
        
        #Initalize the weights as 0's
        if( num_hidden_layers > 0 ):
            self.next_node = HiddenLayer(num_array, class_array, num_hidden_layers)
        else:
            self.next_node = FinalLayer(num_array, class_array, 0)
        
        #Iterate as many times as indicated to finalize the chaning of the weights
        for iter_num in range(iterations):
            
            #Calculate the output values
            calc_array = self.next_node.feed_forward( num_array )
            
            #adjust the weights based on gradient descent propagated throughout all hidden layers
            self.next_node.back_propagation(class_array, learning_rate)
            
            #For the designated steps, display the loss
            if( iter_num % show_error_step == 0 ):
                mean_squared_error = MSE(class_array, calc_array).sum()/class_array.shape[0]
                print( 'Iter #' + str(iter_num) + '\tMSE: ' + str(mean_squared_error) )

        #End function
        return

    #
    def predict(
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
        
        #Calculate the values and if they are above 0, then make 1 else -1
        output = self.next_node.feed_forward( num_array )
        output = numpy.where( output >= threshold, 1, 0 )
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
   
    #Build the model used for the prediction
    model = NeuralNetwork(
        data_set
        , attr_cols
        , class_col
        , num_hidden_layers = 2
        , iterations = 10000
        , show_error_step = 100
    )
    
    #Predict the classes that the test values fall under
    predictions = model.predict(testing_set, attr_cols)
    
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
#output.to_csv( data_file.replace('in', 'out') )




"""
This program takes in a file and determines which type of model to run
based on specific strings in the file name

'categories'    = Naive Bayes

'normalized'    = Adaline

'standard'      = Logistic Regression


Files with 'in' means that they are staged to be fed into the program

Files with 'out' means that they are produced by the program
"""





















