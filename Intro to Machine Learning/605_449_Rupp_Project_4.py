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
set_percent = 0.20

#Get the arguments from command prompt (given as directory containing files)
data_file = sys.argv[1]

#Read the file into a data set
data_set = pandas.read_csv( data_file )

#Make the portion of the training set P%
set_size = math.floor(set_percent * len(data_set))

#list of column names that make up the attribute names
attr_names = [name for name in data_set.columns[1:len(data_set.columns)] ]
class_name = data_set.columns[0]



#Returns
def get_info_gain(
    attr_name       #Attribute Column name to find entropy of
    , class_name    #Classification Column that is being trained/predicted
    , data_set      #panda data set where neighbors will reside
    , row_nums      #Row numbers to use, if not specified use entire set
):
    
    #Get a series of counts for each attribute, class pairing
    attr_counts = data_set.iloc[row_nums].groupby([attr_name, class_name]).size()
    class_counts = data_set.iloc[row_nums].groupby(class_name).size()
    
    #Get unique attribute values and class values
    attr_vals = list(set(data_set[attr_name][row_nums]))
    class_vals = list(set(data_set[class_name][row_nums]))
   
    #Get the number of values being analyized in the entire set
    set_length = len(row_nums)
    
    entropy = 0
    
    #Add entropy for all classes without the attributes
    for cl in class_vals:
        calc = class_counts[cl] / set_length
        entropy += -calc * math.log(calc, 2)
    
    #Add entropy for all classes under each attribute
    for attr in attr_vals:
        
        #Get Total # of instances of each attribute
        total_attr = sum(attr_counts[attr])
        
        attr_entropy = 0
        
        #Sum the entropy for each class for given attribute
        for cl in [x for x in attr_counts[attr].index]:
            calc = attr_counts[attr][cl] / total_attr
            attr_entropy += -calc * math.log(calc, 2)
            
        #Subtract the entropy for each attribute/class pair from Entropy of the set
        entropy -= total_attr / set_length * attr_entropy
    
    #Return the top indices as a list
    return( entropy )




class DTNode:
    
    def __init__(
        self
        , data_set          #
        , attribute_fields  #
        , class_field       #
        , row_nums          #Row numbers to use for the given subset
    ):
        self._attribute = None
        self._class = None
        self.sub_nodes = pandas.Series(None)
        
        #Get all possible classes from set
        classes = list(set(data_set[class_field][row_nums]))
        
        #If there are multiple classes, keep building the tree for more details
        if( len(classes) > 1 ):
            
            #Get all information gain for each attribute field
            ser = pandas.Series(
                    [ get_info_gain( attr, class_field, data_set, row_nums ) for attr in attribute_fields ]
                    , attribute_fields
            )
            
            #Sort information gains to have most gain at top
            ser.sort_values(ascending = False, inplace = True)
            
            #Determine attribute (column) that had most information gain
            attribute = ser.index[0]
            self._attribute = attribute
            
            #Each attribute in the field will represent a different sub-node
            for attr in list(set(data_set[attribute][row_nums])):
                
                #reduce set to those with this attribute value
                new_row_nums = [ i for i in numpy.where(data_set[attribute][row_nums] == attr)[0] ]
                
                #remove current attribute field from list
                attributes = list(attribute_fields)
                attributes.remove(attribute)
                
                #create new node with the subset
                self.sub_nodes.at[attr] = DTNode(data_set, attributes, class_field, new_row_nums)
        
        #If only 1 class, then stop the tree here and set the value as the class
        elif( len(classes) == 1 ):
            self._class = classes[0]
          
        #If no classes, then set as unknown
        else:
            self._class = 'UNKNOWN'
        
            

#
class DecisionTree:
    
    def __init__(
        self
        , training_set      #
        , attribute_fields  #
        , class_field       #
    ):
        
        self._root_node = DTNode(
                training_set
                , attribute_fields
                , class_field
                , row_nums = [x for x in range(len(data_set)) ]
        )




dt = DecisionTree(data_set, attr_names, class_name)



#K nearest neighbors classifier
def knn_classifier(
        training_set    #Data for training the model
        , testing_set   #Data to predict the classifier
        , num_neighbors #number of neighbors to use in classifier
        , use_cnn_reduce = False
):
    
    #Send back the predictions
    return(  )
    

#K nearest neighbors Regressor
def knn_regressor(
        training_set    #Data for training the model
        , test_set      #Data to predict the classifier
        , num_neighbors #number of neighbors to use in classifier
):
    
    #Send back the predictions
    return(  )

#Determine the accuracy of the KNN Classifier accuracy
def classifierAccuracy(
        actual          #list of the actual values
        , predicted     #list of the predicted values
):
    
    #Return the percent correct of all predicted values
    return(  )

#Determine the accuracy of the KNN Classifier accuracy
def regressorAccuracy(
        actual          #list of the actual values
        , predicted     #list of the predicted values
):
    
    #Return the percent correct of all predicted values
    return(  )





######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################





######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################






































