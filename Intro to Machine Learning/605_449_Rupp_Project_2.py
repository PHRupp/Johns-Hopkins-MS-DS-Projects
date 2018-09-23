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
import math
import random
import sys

#Seed makes the results reproducible
random.seed(23)

#Get the arguments from command prompt (given as directory containing files)
data_file = sys.argv[1]

#Read the file into a data set
data_set = pandas.read_csv( data_file, header = None, index_col = False )

"""
@returns
"""
def cluster_k_means(
        data_set            #
        , num_clusters = 5  #
        , iterations = 5    #
):
    
    #Set the clusters initially as random points in the data set (using random indices)
    cluster_indices = random.sample( range(len(data_set)), num_clusters )
    
    #Set the cluster set as the chosen random points
    cluster_set = data_set.iloc[cluster_indices]
    
    #Distinguish the cluster records from each other and set as index
    cluster_set['name'] = [str(ind) for ind in list( range( len( cluster_set )))]
    
    #Set the columns to be calculated
    calc_cols = list( range( len( data_set.columns )))
    
    #Set the cluster name field for the data set
    data_set['cluster'] = '-1'
    cluster_col = int( numpy.where( data_set.columns == 'cluster' )[0] )
    
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
                    closest_cluster = cluster_set.iloc[cluster_index]['name']
             
            #Set the closest cluster to the given data point
            data_set.iat[row_index, cluster_col] = closest_cluster
        
        #Find the new centroids as the average of all the points within the cluster
        for cluster_index in range(len(cluster_set)):
             
            #find the points that reside within the specific cluster
            cluster_points_ind = numpy.where( data_set.cluster == cluster_set.iloc[cluster_index]['name'] )
            
            #Set clusters values as the avg of all the points within the cluster
            cluster_set.iloc[cluster_index, calc_cols] = data_set.iloc[cluster_points_ind].mean(axis = 0, numeric_only = True)
    
    #Send the cluster data frame back
    return(cluster_set)














































































