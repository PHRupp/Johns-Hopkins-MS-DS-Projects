# -*- coding: utf-8 -*-
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

##################### IMPORTANT
#Must be set properly to perform correct analysis
run_classifier = False
##################### IMPORTANT

#Define the % of training set to use
set_percent = 0.10
num_sets = 10

#Get the arguments from command prompt (given as directory containing files)
data_file = sys.argv[1]

#Read the file into a data set
data_set = pandas.read_csv( data_file )

#Make the portion of the training set P%
set_size = math.floor(set_percent * len(data_set))

#list of column names that make up the attribute names
attr_names = [name for name in data_set.columns[1:len(data_set.columns)] ]
class_name = data_set.columns[0]



######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################






#
class DTNode:
    
    def __init__(
        self
        , data_set          #
        , attr_names        #
        , class_name        #
        , row_nums          #Row numbers to use for the given subset
        , lvl = 1           #Defines how deep in the tree (root = 1)
    ):
        self._attribute = None
        self._class = None
        self._lvl = lvl
        self.sub_nodes = pandas.Series(None)
        self._is_numeric = None
        self._numeric_bound = None
        
        #Get all possible classes from set
        classes = list(set(data_set[class_name][row_nums]))
        
        #
        if( len(attr_names) == 0 ):
            
            #Get a series of counts for each attribute, class pairing
            class_counts = data_set.iloc[row_nums].groupby(class_name).size()
                
            #Sort information gains to have most frequent at top
            class_counts.sort_values(ascending = False, inplace = True)
            
            #If No more attributes, then take the most frequent class
            self._class = [x for x in class_counts.index][0]
            
        else:
        
            #If there are multiple classes, keep building the tree for more details
            if( len(classes) > 1 ):
                
                #Get all information gain for each attribute field
                ser = pandas.Series(
                        [ self.get_gain_ratio( attr, class_name, data_set, row_nums ) for attr in attr_names ]
                        , attr_names
                )
                
                #Sort information gains to have most gain at top
                ser.sort_values(ascending = False, inplace = True)
                
                #Determine attribute (column) that had most information gain
                attribute = ser.index[0]
                self._attribute = attribute
                        
                #remove current attribute field from list
                attributes = list(attr_names)
                attributes.remove(attribute)
                
                #The following is in regards to categorical data
                if( not pandas.api.types.is_numeric_dtype(data_set.dtypes[attribute]) ):
                    
                    #Each attribute in the field will represent a different sub-node
                    for attr in list(set(data_set[attribute][row_nums])):
                        
                        #reduce set to those with this attribute value
                        new_row_nums = [ i for i in numpy.where(data_set[attribute][row_nums] == attr)[0] ]
                        
                        #create new node with the subset
                        self.sub_nodes.at[attr] = DTNode(data_set, attributes, class_name, new_row_nums, lvl+1)
                        
                #The following column is numeric
                elif( pandas.api.types.is_numeric_dtype(data_set.dtypes[attribute]) ):
                    
                    self._is_numeric = True
                    
                    #Use the median as the split in the data
                    self._numeric_bound = numpy.median( data_set[attribute][row_nums] )
                    
                    #Define the records that are greater or less than the numeric bounds
                    greater_than_rows = [x for x in numpy.where(data_set[attribute][row_nums] >= self._numeric_bound)[0] ]
                    less_than_rows = [x for x in numpy.where(data_set[attribute][row_nums] < self._numeric_bound)[0] ]
                    
                    #create new node with the subsets
                    self.sub_nodes.at['>= ' + str(self._numeric_bound)] = DTNode(data_set, attributes, class_name, greater_than_rows, lvl+1)
                    self.sub_nodes.at['< ' + str(self._numeric_bound)] = DTNode(data_set, attributes, class_name, less_than_rows, lvl+1)
            
            #If only 1 class, then stop the tree here and set the value as the class
            elif( len(classes) == 1 ):
                self._class = classes[0]
                
            #If no classes, then set as unknown
            else:
                self._class = 'UNKNOWN'
        return
    
    
    #Determines if the current node is a leaf node
    def is_leaf(self):
        if( self.sub_nodes.empty ):
            return( True )
        else:
            return( False )
       

    #Determines if the current node is a leaf node
    def is_numeric(self):
        if( self._is_numeric ):
            return( True )
        else:
            return( False )


    #Indentation based on how deep the node is in the tree
    def print_node(self):  

        #If the node is a leaf, print the expected class
        if( self.is_leaf() ):
            
            #Create output string for given node
            #'  CLASS: example1'
            #'    CLASS: example2'
            print(''.join([' '] * self._lvl) + "CLASS: " + self._class)
            
        #If not a leaf, then print the current node specs
        else:
            
            #Create output string for given node
            #' LVL 1: COL_NAME1 -> example1'
            #'   LVL 3: COL_NAME4 -> example4'
            output = ''.join([' '] * self._lvl) + "LVL " + str(self._lvl) + ": " + self._attribute + " -> "
            
            #Loop through all downstream edges for the given node
            for node in [x for x in self.sub_nodes.index]:
                
                #print node to output
                print( output + node )
                
                #Print the sub-node
                self.sub_nodes[node].print_node()
   
    
    #Returns
    def get_gain_ratio(
        self
        , attr_name     #Attribute Column name to find entropy of
        , class_name    #Classification Column that is being trained/predicted
        , data_set      #panda data set where neighbors will reside
        , row_nums      #Row numbers to use, if not specified use entire set
    ):
        
        #The following is in regards to categorical data
        if( not pandas.api.types.is_numeric_dtype(data_set.dtypes[attr_name]) ):
            
            #Get a series of counts for each attribute, class pairing
            attr_counts = data_set.iloc[row_nums].groupby([attr_name, class_name]).size()
            class_counts = data_set.iloc[row_nums].groupby(class_name).size()
        
        #The following column is numeric
        elif( pandas.api.types.is_numeric_dtype(data_set.dtypes[attr_name]) ):
            
            #Use the median as the split in the data
            num_bounds = numpy.median( data_set[attr_name][row_nums] )
            
            #Define the records that are greater or less than the numeric bounds
            num_categories = [x for x in numpy.where(data_set[attr_name][row_nums] >= num_bounds, 'GE', 'L')]
            
            #Create a temporary data set with numerical data as categorical
            temp_set = pandas.DataFrame({attr_name: num_categories, class_name: data_set[class_name][row_nums]})
        
            #Get a series of counts for each attribute, class pairing
            attr_counts = temp_set.groupby([attr_name, class_name]).size()
            class_counts = temp_set.groupby(class_name).size()
            
        
        #Get unique attribute values and class values
        attr_vals = [x for x in attr_counts.index.levels[0]]
        class_vals = [x for x in class_counts.index]
       
        #Get the number of values being analyized in the entire set
        set_length = len(row_nums)
        
        #Sometimes intrinsic value is '0' so this prevents division by zero
        info_gain = 1.0e-30
        intrinsic_val = 1.0e-30
        
        #Add entropy for all classes without the attributes
        for cl in class_vals:
            calc = class_counts[cl] / set_length
            info_gain -= calc * math.log(calc, 2)
        
        #Add entropy for all classes under each attribute
        for attr in attr_vals:
            
            #Get Total # of instances of each attribute across all classes
            total_attr = sum(attr_counts[attr])
            
            #Calculate Intrinsic Value for the gain ratio
            temp = total_attr / set_length
            intrinsic_val -= temp * math.log(temp, 2)
            
            attr_entropy = 0
            
            #Sum the entropy for each class for given attribute
            for cl in [x for x in attr_counts[attr].index]:
                calc = attr_counts[attr][cl] / total_attr
                attr_entropy -= calc * math.log(calc, 2)
                
            #Subtract the entropy for each attribute/class pair from Info Gain
            info_gain -= total_attr / set_length * attr_entropy
            
        #Return the top indices as a list
        return( info_gain / intrinsic_val )
        
        
#
class DecisionTree:
    
    #
    def __init__(
        self
        , data_set          #
        , attr_names  #
        , class_name       #
    ):
        #Create the tree starting with top node down to the leaves
        self._root_node = DTNode(
                data_set
                , attr_names
                , class_name
                , row_nums = [ x for x in range(len(data_set)) ]
        )
    
    #
    def print_tree(self):
        
        #Print the tree
        self._root_node.print_node()
    
    #
    def predict_class(
        self
        , data_set
        , attr_names
    ):
        #stores the predictions
        predicted_classes = []
        
        #Run through each record and attempt to classify
        for row_num in range(len(data_set)):
            
            #Classify the given record by working down the tree with the given attributes
            predicted_classes.append( self.classify(data_set.iloc[row_num], self._root_node) )
        
        return( predicted_classes )

    #
    def classify(
        self
        , record    #Pandas.Series of 1 row of a data frame for classification
        , node      #
    ):
     
        #If the node is a leaf, set the expected class
        if( node.is_leaf() ):
            
            #Create output string for given node
            return( node._class )
        
        #If not a leaf, then print the current node specs
        else:
            
            #
            if( node.is_numeric() ):
                
                #Determine the possible paths
                paths = [x for x in node.sub_nodes.index]
                
                #Verify which is the true greater path
                if( ">=" in paths[0] ):
                    greater_path = paths[0]
                    lesser_path = paths[1]
                else:
                    greater_path = paths[1]
                    lesser_path = paths[0]
                
                #Go down the appropriate path
                if( record[node._attribute] >= node._numeric_bound ):
                
                    #Continue down tree until the class is reached
                    return( self.classify( record, node.sub_nodes[greater_path] ) )
                else:
                    #Continue down tree until the class is reached
                    return( self.classify( record, node.sub_nodes[lesser_path] ) )
                    
            else:
            
                #Get the record's attribute value
                record_val = record[node._attribute]
                
                #Verify that the record's value exists in tree, and continue down that path
                if( record_val in [x for x in node.sub_nodes.index] ):
                    
                    #Continue down tree until the class is reached
                    return( self.classify( record, node.sub_nodes[record_val] ) )
                    
                #If there is no path for the given value, then return value
                else:
                    return( 'unknown' )
 

#Determine the accuracy of the Classifier accuracy
def classifierAccuracy(
        actuals          #list of the actual values
        , predictions     #list of the predicted values
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





#Predict the classes that the test values fall under
dt = DecisionTree(data_set, attr_names, class_name)





#print the decision tree
dt.print_tree()


actuals = [x for x in test_set[class_name] ]

predictions = dt.predict_class(test_set, attr_names)


print( classifierAccuracy(actuals, predictions) )















#define which indices will be used for the training set
data_set_indices = list( range( len( data_set )))

set_manager = []


#Create the specified number of data groups
for set_num in range(num_sets):
    
    set_manager.append( random.sample(data_set_indices, set_size) )
    
    data_set_indices = [x for x in numpy.where([x not in set_manager[-1] for x in data_set_indices])[0]]

#Collect the rest of the values in the last set
set_manager[-1] = set_manager[-1] + data_set_indices

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
    
    #Predict the classes that the test values fall under
    dt = DecisionTree(training_set, attr_names, class_name)
    predictions = dt.predict_class(testing_set, attr_names)
    
    #Take the actual classes that the data belongs to
    actuals = [x for x in testing_set[class_name] ]
    results.append( classifierAccuracy(actuals, predictions) )
    
    predicts.append(predictions)
    
#indices = [item for sub in set_manager for item in sub]
#output = data_set.iloc[indices].copy() 
#output['predict'] = [item for sub in predicts for item in sub]
#output.to_csv("B:/Users/pathr/Documents/Education/JHU/DATA/605.449.81.FA18/out_forestfires.csv")

dt.print_tree()

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






































