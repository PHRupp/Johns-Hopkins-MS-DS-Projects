"""
Course: 605.449 - Introduction to Machine Learning
Project: #6
Due: Sun Nov 18 23:59:59 2018
@author: Patrick H. Rupp

NOTES: 
    
    This program takes in a file of normalized values (0-1) and runs 4 models
    on the data. Three are neural networks with 0, 1, and 2 hidden layers while
    the fourth is a RBF network.
    
    Files with 'in' means that they are staged to be fed into the program
    
    Files with 'out' means that they are produced by the program. These files
    have extra columns containing the predicted outputs
"""

#### DEFINE GLOBAL REQUIREMENTS ####

import pandas
import numpy
import random
import sys


#Define the % of training set to use
num_sets = 5

#define number of iterations for all models
num_iterations = 5000
mse_show_error = 5000
my_learning_rate = 1e-6

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
    
    #Initialize the weights, output, and stored layer values
    #layer values being the output of the previous node
    def __init__(
        self
        , num_array              
        , class_array           
        , node_id        #Number of hidden nodes left to create
    ):
        
        self.node_id = node_id
        
        #Initalize the weights as small values
        self.weights = numpy.random.randn( num_array.shape[1], 1 )
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
        error_change = MSE_d1(class_array, self.output) * sigmoid_curve_d1( self.layer_vals.dot(self.weights) )

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
    
    #Initialize the weights, output, and stored layer values
    #layer values being the output of the previous node
    def __init__(
        self
        , num_array              
        , class_array           
        , node_id        #Number of hidden nodes left to create
    ):

        self.node_id = node_id
        
        #Initalize the weights as small values
        self.weights = numpy.random.randn( num_array.shape[1], num_array.shape[1] )
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
##                                                      RBF                                                         ##
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################

#
class RBF:
     
    #
    def __init__(
        self
        , data_set              #Pandas Data Frame of training set
        , attr_cols             #List of attribute column names
        , class_col             #Column name of the Class of the data
        , num_centers = 4       #Number of hidden layers
        , iterations = 1000     #Number of iterations the model will be adjusted
        , add_intercept = True  #Adds column of 1's so that the weights are by themselves
    ):
        #define the addition of the intercept 
        self.add_intercept = add_intercept
        self.num_centers = num_centers
        self.critical_radius = 8
        
        #Initialize the data frame as arrays for ease of calculations
        num_array = numpy.array( data_set[attr_cols] )
        class_array = numpy.array( data_set[class_col] )
        
        #Make a numerical array from the data frame with intercept if indicated
        if( self.add_intercept ):
            num_array = numpy.concatenate( (numpy.ones((num_array.shape[0], 1)), num_array), axis=1)
        
        #get the center coordinates for all the centers
        self.centers, self.critical_radius = self.kmeans(num_array, self.num_centers, iterations)
        
        #This will be used within gaussian function for the 'width' of the curve
        self.critical_radius = 1 / ( 2 * self.critical_radius**2 )
        
        #use the gaussian activation function
        gauss_vals_center_record = self.get_record_gaussians_from_centers(num_array)
        
        #get the weights with pseudoinverse algorithm
        self.weights = numpy.linalg.pinv( gauss_vals_center_record ).dot( class_array )
    
        #end function
        return
    
    def kmeans(
        self
        , num_array
        , num_centers
        , iterations
    ):
                
        #get the center coordinates for all the centers
        #self.centers = kmeans()
        random_indices = random.sample( range(num_array.shape[0]), num_centers )
        clusters_array = [num_array[i,:] for i in random_indices]
        
        #Initialize the distance matrix of each record (row) from a cluster (column)
        distance_matrix = numpy.zeros( (num_array.shape[0], num_centers) )
        
        #Adjust the cluster centers for the set amount of times
        for i in range(iterations):
        
            #Calculate the distances of all points from each cluster
            for cluster_col, cluster in enumerate(clusters_array):
                
                #Calculate the distances of all points from the cluster and store it in matrix
                distance_matrix[:, cluster_col] = numpy.linalg.norm(num_array - cluster, axis = 1)
                
            #find which cluster each record belongs based on the minimum distance of all clusters
            cluster_indices = numpy.argmin( distance_matrix, axis = 1 )
                
            #Re-calculate the positions of all the cluster centers
            for cluster_index in set(cluster_indices):
                
                #Determine which records belong to that cluster
                num_array_cluster_indices = numpy.where( cluster_indices == cluster_index )[0]
                
                #Get the new cluster center values based on the average of all the closest points in the cluster
                new_cluster_vals = numpy.mean(num_array[num_array_cluster_indices, :], axis = 0)
                
                #Update the cluster center
                clusters_array[cluster_index] = new_cluster_vals
        
        """
        The below is to calculate the average distance of each point from its cluster center        
        """
        #Calculate the distances of all points from each cluster
        for cluster_col, cluster in enumerate(clusters_array):
            
            #Calculate the distances of all points from the cluster and store it in matrix
            distance_matrix[:, cluster_col] = numpy.linalg.norm(num_array - cluster, axis = 1)
            
        #find which cluster each record belongs based on the minimum distance of all clusters
        cluster_distances = numpy.min( distance_matrix, axis = 1 )
        
        #Find the average distance each point is from it's cluster center
        avg_distance_from_clusters = cluster_distances.sum() / cluster_distances.shape[0]
        
        #end function and return list of cluster centers
        return clusters_array, avg_distance_from_clusters
    
    
    #radial basis function using gaussian distribution
    def radial_basis_gaussian(
        self
        , cluster_center
        , record_vals
    ):
        #end function
        return numpy.exp(-self.critical_radius * numpy.linalg.norm( (cluster_center - record_vals)**2 ))
    
    
    #returns distance values for each point (row) for a given centroid center (column)
    def get_record_gaussians_from_centers(
        self
        , num_array
    ):
        
        #initialize the variables that will store the gaussian values of each record (rows)
        #from each center (columns)
        gauss_vals_center_record = numpy.zeros( (num_array.shape[0], self.num_centers) )
        
        #for every center, calculate the gaussian from all the points
        for center_index, center_vals in enumerate(self.centers):
        
            #for a given center, calculate the gaussian for each point and store it
            for record_index, record_vals in enumerate(num_array):
                gauss_vals_center_record[record_index, center_index] = self.radial_basis_gaussian( center_vals, record_vals )
        
        #end function
        return gauss_vals_center_record 
        
    
    #
    def predict(
        self
        , data_set          #Pandas Data Frame of testing set
        , attr_cols         #List of attribute column name
        , threshold = 0.05  #Limit where class boundaries are separated
    ):
        
        #Initialize the data frame as arrays for ease of calculations
        num_array = numpy.array( data_set[attr_cols] )
        
        #Make a numerical array from the data frame with intercept if indicated
        if( self.add_intercept ):
            num_array = numpy.concatenate( (numpy.ones((num_array.shape[0], 1)), num_array), axis=1)
        
        #Calculate the values and if they are above 0, then make 1 else -1
        output = self.get_record_gaussians_from_centers(num_array).dot( self.weights )
        output = numpy.where( output >= threshold, 1, 0 )
        output = [output[x] for x in range(output.shape[0])]
        
        #end function and return predictions
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
results_nn_0h = []
results_nn_1h = []
results_nn_2h = []
results_rbf = []
predicts_nn_0h = []
predicts_nn_1h = []
predicts_nn_2h = []
predicts_rbf = []
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
   
    print('\nNN (hidden = 0):')
    #Build the model used for the prediction
    model_nn_0h = NeuralNetwork(
        data_set
        , attr_cols
        , class_col
        , num_hidden_layers = 0
        , iterations = num_iterations
        , show_error_step = mse_show_error
        , learning_rate = my_learning_rate
    )
   
    print('\nNN (hidden = 1):')
    #Build the model used for the prediction
    model_nn_1h = NeuralNetwork(
        data_set
        , attr_cols
        , class_col
        , num_hidden_layers = 1
        , iterations = num_iterations
        , show_error_step = mse_show_error
        , learning_rate = my_learning_rate
    )
   
    print('\nNN (hidden = 2):')
    #Build the model used for the prediction
    model_nn_2h = NeuralNetwork(
        data_set
        , attr_cols
        , class_col
        , num_hidden_layers = 2
        , iterations = num_iterations
        , show_error_step = mse_show_error
        , learning_rate = my_learning_rate
    )
    
    #Build the model used for the prediction
    model_rbf = RBF(
        data_set
        , attr_cols
        , class_col
        , num_centers = 4
        , iterations = num_iterations
    )
    print('\nRBF ran... trust me')
    
    #Predict the classes that the test values fall under
    predictions_nn_0h = model_nn_0h.predict(testing_set, attr_cols, threshold = 0.1)
    predictions_nn_1h = model_nn_1h.predict(testing_set, attr_cols, threshold = 0.1)
    predictions_nn_2h = model_nn_2h.predict(testing_set, attr_cols, threshold = 0.1)
    predictions_rbf = model_rbf.predict(testing_set, attr_cols)
    
    #Take the actual classes that the data belongs to
    actuals = [x for x in testing_set[class_col] ]
    results_nn_0h.append( classifierAccuracy(actuals, predictions_nn_0h) )
    results_nn_1h.append( classifierAccuracy(actuals, predictions_nn_1h) )
    results_nn_2h.append( classifierAccuracy(actuals, predictions_nn_2h) )
    results_rbf.append( classifierAccuracy(actuals, predictions_rbf) )
    
    #Store the actual and predicted values for analysis later.
    predicts_nn_0h.append(predictions_nn_0h)
    predicts_nn_1h.append(predictions_nn_1h)
    predicts_nn_2h.append(predictions_nn_2h)
    predicts_rbf.append(predictions_rbf)
    real_vals.append(actuals)
    
print('\nNN (hidden = 0): ', results_nn_0h)  
print('\nNN (hidden = 1): ', results_nn_1h)  
print('\nNN (hidden = 2): ', results_nn_2h)  
print('\nRBF: ', results_rbf)    

#Print the data set out with the actual vs. predicted columns
indices = [item for sub in set_manager for item in sub]
output = data_set.iloc[indices].copy() 
output['predict_nn_0h'] = [item for sub in predicts_nn_0h for item in sub]
output['predict_nn_1h'] = [item for sub in predicts_nn_1h for item in sub]
output['predict_nn_2h'] = [item for sub in predicts_nn_2h for item in sub]
output['predict_rbf']   = [item for sub in predicts_rbf for item in sub]
output['actual'] = [item for sub in real_vals for item in sub]
output.to_csv( data_file.replace('in', 'out') )

























