"""
Course: 605.449 - Introduction to Machine Learning
Project: #4
Due: Sun Oct 7 23:59:59 2018
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

#Make the portion of the training set P%
set_size = math.floor( set_percent * data_set.shape[0] )

#list of column names that make up the attribute names
attr_cols = [name for name in data_set.columns[1:len(data_set.columns)] ]
class_col = data_set.columns[0]



######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
##                                                  NAIVE BAYES                                                     ##
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################

"""
This class is the portion of the Naive Bayes model implemented as a tree
in which the particular node defines a particular attribute in the 
subset of the data (aka. only 1 class)

It provides the probability of an attribute value existing within the
attribute field for a given class subset
"""
class NBAttributeNode:
    
    def __init__(
        self
        , data_set          #
        , attr_col          #List of attribute column names
        , row_nums          #Row numbers to use for the given subset
    ):
        #probability values for a given attribute value in the subset
        self.attribute_values = pandas.Series(None)
            
        #Get all the unique attribute values
        attribute_vals = list(set(data_set[attr_col][row_nums]))
        
        #Calculate the probability of each attribute value existing in the class
        for val in attribute_vals:
            
            #Get the number of instaces with this particular value in the subset
            num_instances = len( numpy.where(data_set[attr_col][row_nums] == val)[0] )
            
            #create new node with the probality of this value being in the class
            self.attribute_values.at[val] = num_instances / len( row_nums )
        
        #End function
        return
    
    #Indentation based on how deep the node is in the tree
    def print_node(self, attribute):  
        
        #Print the class information
        print(''.join([' '] * 2) + "ATTRIBUTE: '" + attribute + "'")
        
        #Print each attribute's information
        for attr_val in self.attribute_values.index:
        
            #Print the class information
            print(''.join([' '] * 3) + "VALUE: '" + attr_val + "' has P = " + str(self.attribute_values[attr_val]))
            
        #End function
        return


"""
This class is the portion of the Naive Bayes model implemented as a tree
in which the particular node defines a particular subset in the data (aka. only 1 class)

It contains all the attribute fields for the given subset in which we can
extract 
"""
class NBClassNode:
    
    def __init__(
        self
        , data_set          #Pandas Data Frame of training set
        , attr_cols         #List of attribute column names
        , row_nums          #Row numbers to use for the given subset
        , class_probability #Probability of the class existing in the training set
    ):
        self._class_probability = class_probability
        
        #sub-nodes for each attribute field
        self.attributes = pandas.Series(None)
        
        #Process each attribute in the class to find the probabilities
        for attr_col in attr_cols:
            
            #create new node with the subset for that given class' values
            self.attributes.at[attr_col] = NBAttributeNode(
                    data_set
                    , attr_col
                    , row_nums
            )
        
        #End function
        return
    
    #Indentation based on how deep the node is in the tree
    def print_node(self, class_name):  
        
        #Print the class information
        print('\n'.join([' '] * 1) + "CLASS: '" + class_name + "' has P = " + str(self._class_probability))
        
        #Print each attribute's information
        for attr_col in self.attributes.index:
            
            self.attributes[attr_col].print_node(attr_col)

        #End function
        return

"""
This class is the portion of the Naive Bayes model implemented as a tree
in which the particular node defines the root of the tree (top)

Each sub-node defines a particular class which will be subset for all the data
given that class 
"""
class NBRootNode:
    
    def __init__(
        self
        , data_set          #Pandas Data Frame of training set
        , attr_cols         #List of attribute column names
        , class_col         #Column name of the Class of the data
    ):
        #sub-nodes for each class
        self.classes = pandas.Series(None)
        
        #Process each class to find the probabilities
        for class_name in list(set(data_set[class_col])):
            
            #Get the row numbers for the subset with only the given class
            new_row_nums = [ i for i in numpy.where(data_set[class_col] == class_name)[0] ]
            new_row_nums = data_set.index[new_row_nums]
            
            #create new node with the subset for that given class' values
            self.classes.at[class_name] = NBClassNode(
                    data_set
                    , attr_cols
                    , new_row_nums
                    , len( new_row_nums ) / data_set.shape[0]
            )
            
        #End function
        return
    
    #Indentation based on how deep the node is in the tree
    def print_node(self):  
        
        for class_name in self.classes.index:
            
            self.classes[class_name].print_node(class_name)
        
        #End function
        return
       
        
        
"""
This class is the implementation of the Naive Bayes model as a tree which is
designed to be wide not deep.
            root
          /      \
      class1      class2   ...
     /     \     /     \
col 1    col2  col1   col2 ...
         /   \
attribute1   attribute2    ...

where each attribute# is the probability of that attribute existing in the
subset of the data for a given class 
"""
class NaiveBayes:
    
    #
    def __init__(
        self
        , data_set          #Pandas Data Frame of training set
        , attr_cols         #List of attribute column names
        , class_col         #Column name of the Class of the data
    ):
        #Create the tree starting with top node down to the leaves
        self._root_node = NBRootNode(
            data_set
            , attr_cols    
            , class_col
        )
        
        #End function
        return
    
    #
    def print_model(self):
        
        #Print the tree
        self._root_node.print_node()
        
        #End function
        return
    
    #Predict the class that the test data belongs to
    def predict_class(
        self
        , data_set
        , attr_cols     #List of attribute column names
    ):
        #stores the predictions
        predicted_classes = []
        
        #Run through each record and attempt to classify
        for row_num in range(len(data_set)):
            
            #Classify the given record by working down the tree with the given attributes
            predicted_classes.append( self.classify_record(data_set.iloc[row_num], attr_cols) )
        
        #End function
        return( predicted_classes )

    #Find the most probabilistic class that the given record belongs to
    def classify_record(
        self
        , record        #Pandas.Series of 1 row of a data frame for classification
        , attr_cols     #List of attribute column names
    ):
        
        max_probability = 0
        probable_class = ''
        
        #Find the probality of the record being each class and take the most likely
        for class_name in self._root_node.classes.index:
            
            #Initially set value as the probability of the class to exist
            class_probability = self._root_node.classes[class_name]._class_probability
            
            #Multiply the probability of each attribute for the given class
            for attr_col in attr_cols:
                
                #If the record exists multiply the probality of that attribute value existing in that class
                if( record[attr_col] in self._root_node.classes[class_name].attributes[attr_col].attribute_values.index ):
                    class_probability *= self._root_node.classes[class_name].attributes[attr_col].attribute_values[ record[attr_col] ]
                else:
                    class_probability *= 0.00001
            
            #If the new class is more likely than the current best, then set the new best
            if( max_probability < class_probability ):
                max_probability = class_probability
                probable_class = class_name
            
        #End function
        return probable_class
 
    


######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
##                                               LOGISTIC REGRESSION                                                ##
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################


#
class LogisticRegression:
    
    #
    def __init__(
        self
        , data_set              #Pandas Data Frame of training set
        , attr_cols             #List of attribute column names
        , class_col             #Column name of the Class of the data
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

    #
    def sigmoid_curve(
        self
        , x_vals    #Some numeric value(s) either stand alone or as Numpy Array
    ):
        #End function
        return( 1 / (1 + numpy.exp(-x_vals)) )
    
    #
    def error_gradient(
        self
        , num_array         #Numpy numerical array (represents numbers from data frame)
        , class_array       #Numpy numerical array (represents numbers from class column)
    ):
        #Dot product of numeric attributes onto weights (N = number of rows, D = number of attributes)
        #this output array is size N,1
        weighted_set = numpy.dot( num_array, self.weights )
        
        #Run sigmoid curve using the weighted_set as the summation of all the 
        #attributes (0-1) with some given weights
        sigmoid_set = self.sigmoid_curve( weighted_set )
        
        #The shape was giving issues bewteen sigmoid_set and class_array
        diff = numpy.array( [sigmoid_set[x] - class_array[x] for x in range(sigmoid_set.shape[0])] )
        
        #End function
        return numpy.dot( num_array.T, diff ) / num_array.shape[0]

    #
    def loss_function(
        self
        , num_array         #Numpy numerical array (represents numbers from data frame)
        , class_array       #Numpy numerical array (represents numbers from class column)
    ):
        
        #Dot product of numeric attributes onto weights (N = number of rows, D = number of attributes)
        #this output array is size N,1
        weighted_set = numpy.dot( num_array, self.weights )
        
        #Run sigmoid curve using the weighted_set as the summation of all the 
        #attributes (0-1) with some given weights
        sigmoid_set = self.sigmoid_curve( weighted_set )
        
        #Calculate the loss for all the points in the set
        loss = -class_array * numpy.log( sigmoid_set ) - (1 - class_array) * numpy.log( 1 - sigmoid_set )
        
        #End function
        return loss.mean()
        
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





















