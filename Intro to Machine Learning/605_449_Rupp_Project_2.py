# -*- coding: utf-8 -*-
"""
Course: 605.449 - Introduction to Machine Learning
Project: #2
Due: Sat Sep 23 23:59:59 2018
@author: Patrick H. Rupp
"""

#### DEFINE GLOBAL REQUIREMENTS ####

import pandas
import numpy
import random
import sys
import statistics

#Seed makes the results reproducible
random.seed(23)

#Get the arguments from command prompt (given as directory containing files)
data_file = sys.argv[1]

#Read the file into a data set
data_set = pandas.read_csv( data_file, header = None, index_col = False )


#### DEFINE CLUSTER SILHOUETTE LOSS FUNCTION ####


"""
@returns
"""
def get_silhouette(
        data_set            #
        , cluster_set       #
        , calc_cols = None  #
        , cluster_col_name = 'cluster'
        , cluster_name = 'name'
):
    
    #Set the columns to be calculated
    calc_data_cols = [num for num in numpy.where( [(col != cluster_col_name) for col in data_set.columns] )[0]]
    
    #Average distance of all points within similar cluster (inclusive) and other clusters (exclusive) to a given point
    inclusive_avg_distance = []   
    exclusive_avg_distance = []
        
    #Calculate the distances of a given point with all the other points in the cluster
    for index in range(len(data_set)):
        
        #Define the specific cluster being reviewed
        cluster = data_set.iloc[index][cluster_col_name]
        
        #Get the indices of the data that fall within the specified cluster
        cluster_indices = [num for num in numpy.where( data_set[cluster_col_name] == cluster )[0] ]
        
        #Calculate the distances of a given point with all the other points in the cluster
        distances = [
                numpy.linalg.norm( data_set.iloc[index][calc_data_cols] - data_set.iloc[ind, calc_data_cols] ) 
                for ind in cluster_indices
        ]
        
        #Calculate the average distance disregarding the point with 0 distance to itself
        inclusive_avg_distance.append( sum(distances) / ( len(distances)-1 ) )

    #Calculate the distances of a given point with all the other points in the cluster
    for index in range(len(data_set)):

        #Define the specific cluster being reviewed
        cluster = data_set.iloc[index][cluster_col_name]
        
        #Initialize the minimum cluster average for all clusters that don't correspond to the given data point
        min_clust_avg = sys.maxsize
        
        #Go through each cluster not associated with the given point
        for cluster_num in numpy.where( cluster_set[cluster_name] != cluster )[0]:
            
            #Find cluster that will be reviewed
            temp_cluster = cluster_set.iloc[cluster_num][cluster_name]
        
            #Get the indices of the data that fall within the specified cluster
            cluster_indices = [num for num in numpy.where( data_set[cluster_col_name] == temp_cluster )[0] ]
            
            #Calculate the distances of a given point with all the other points in the cluster
            distances = [
                    numpy.linalg.norm( data_set.iloc[index][calc_data_cols] - data_set.iloc[ind, calc_data_cols] ) 
                    for ind in cluster_indices
            ]
            
            #Calculate the average distance disregarding the point with 0 distance to itself
            cluster_avg_dist = sum(distances) / ( len(distances)-1 )
            
            #Set the minimum cluster avg for the given cluster
            if( min_clust_avg > cluster_avg_dist ):
                min_clust_avg = cluster_avg_dist
            
        #Add the minimum avg distance of another's cluster's points
        exclusive_avg_distance.append( min_clust_avg )

    #Take the average of all silhouette coefficients
    cluster_silhouette = statistics.mean([
            ( 
                    (exclusive_avg_distance[ind] - inclusive_avg_distance[ind]) 
                / max(exclusive_avg_distance[ind], inclusive_avg_distance[ind])
            )
            for ind in range(len(inclusive_avg_distance))
    ])

    #Send the silhouette score
    return(cluster_silhouette)



#### DEFINE CLUSTER METHOD ####


"""
@returns
"""
def cluster_k_means(
        data_set                #
        , calc_cols = None      #list of column indices for numeric attribute fields
        , num_clusters = 5      #
        , iterations = 5        #
        , cluster_col_name = 'cluster'
        , cluster_name = 'name'
        , return_silhouette = False
):
    
    #Set the clusters initially as random points in the data set (using random indices)
    cluster_indices = random.sample( range(len(data_set)), num_clusters )
    
    #Set the columns to be calculated
    non_cluster_cols = [num for num in numpy.where( [(col != cluster_col_name) for col in data_set.columns] )[0]]
    
    #Set the cluster set as the chosen random points
    cluster_set = data_set.iloc[cluster_indices, non_cluster_cols]
    
    #If the given columns for calculation is not input, assume all but cluster field
    if( calc_cols == None ):
        calc_cols = non_cluster_cols
    
    #Distinguish the cluster records from each other and set as index
    cluster_set[cluster_name] = [str(ind) for ind in list( range( len( cluster_set )))]
    
    #Set the cluster name field for the data set
    data_set[cluster_col_name] = '-1'
    cluster_col = int( numpy.where( data_set.columns == cluster_col_name )[0] )
    
    #Re-calculate the clusters through the specified iterations
    for iteration in range(iterations):
    
        #Find the closest cluster for all the points in the set
        for row_index in range(len(data_set)):
            
            #Set the generic closest cluster info for the given data point
            min_distance = sys.maxsize
            closest_cluster = None
            
            #Find which centroid is the closest to the data point with euclidean distance
            for cluster_index in range(len(cluster_set)):
                
                #Calculate the euclidean distance for the data point to the centroid
                distance = numpy.linalg.norm( cluster_set.iloc[cluster_index][calc_cols] - data_set.iloc[row_index][calc_cols] )
                
                #if the distance is the smallest of all centroids, use it as the cluster for that point
                if( distance < min_distance ):
                    min_distance = distance
                    closest_cluster = cluster_set.iloc[cluster_index][cluster_name]
             
            #Set the closest cluster to the given data point
            data_set.iat[row_index, cluster_col] = closest_cluster
        
        #Find the new centroids as the average of all the points within the cluster
        for cluster_index in range(len(cluster_set)):
             
            #find the points that reside within the specific cluster
            cluster_points_ind = numpy.where( data_set.cluster == cluster_set.iloc[cluster_index][cluster_name] )
            
            #Set clusters values as the avg of all the points within the cluster
            cluster_set.iloc[cluster_index, calc_cols] = data_set.iloc[cluster_points_ind].mean(axis = 0, numeric_only = True)
    
    #Calculate Loss if defined by the user
    if( return_silhouette ):
    
        #Calculate the Loss
        loss = get_silhouette(
            data_set            
            , cluster_set = cluster_set
            , calc_cols = calc_cols  
            , cluster_col_name = cluster_col_name
            , cluster_name = cluster_name
        )
    else:
        #Set default loss
        loss = 0
    
    #Send the cluster data frame back
    return(loss)




#### Stepwise Forward Selection ####
cols = [col for col in range(len(data_set.columns))]

#Designate the columns that will be used for the cluster calculations
cols_for_calc = []

#Set the initial loss information
max_silhouette = -1*sys.maxsize
iterate_again = True

while( iterate_again ):
    
    #Defines column to keep. -1 for default
    col_to_keep = -1
    
    #Go through all columns until best is found
    for col in cols:
        
        print( "Processing Col: " + str( cols_for_calc + [col] ) )
        
        #Run Kmeans clustering and return a loss value
        cluster_silhouette = cluster_k_means(data_set, calc_cols = cols_for_calc + [col], return_silhouette = True)
        
        #keep the column with the lowest loss
        if( max_silhouette < cluster_silhouette ):
            max_silhouette = cluster_silhouette
            col_to_keep = col
     
    #Add new column if added value otherwise break
    if( col_to_keep == -1 ):
        break
    else:
        #Keep the column in the calculation and drop from the list to re-run
        print( "Col: " + str(col_to_keep) + " Added" )
        cols_for_calc.append(col_to_keep)
        cols.remove(col_to_keep)    
    
#Spit out the valuable columns and the cluster column
data_set.iloc[:, cols_for_calc + [len(data_set.columns)-1]].to_csv("output.DATA", index = False, header = False)






























































