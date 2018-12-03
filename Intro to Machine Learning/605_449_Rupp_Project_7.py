"""
Course: 605.449 - Introduction to Machine Learning
Project: #7
Due: Sun Dec 9th 23:59:59 2018
@author: Patrick H. Rupp

NOTES: 
    
    
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
map_filepath = sys.argv[1]



######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
##                                                Track Manager Class                                               ##
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################


class TrackManager:
    
    #determine the types of floor values in the map
    map_floor_type = {
        '#': 0
        , 'S': 1
        , '.': 2
        , 'F': 3
    }
    
    #
    def __init__(
        self
        , map_filepath
    ):
        #
        self.map_filepath = map_filepath
        
        #open the file to store the racetrack's map contents
        with open(self.map_filepath, 'r') as map_file:
            
            #get the size of the map (1st line -> ex. "24, 32")
            map_size = map_file.readline()
            map_size_array = numpy.fromstring(map_size, dtype=int, sep=',')
            self.map_vals = numpy.zeros( (map_size_array[0], map_size_array[1]) )
            
            #go through each map position and store the value, and process special points
            for y in range(map_size_array[0]):
                for x in range(map_size_array[1]):
                    
                    #store the map's position value
                    next_val = map_file.read(1)
                    
                    #skip new line values
                    if( next_val == '\n'):
                        next_val = map_file.read(1)
                    
                    #get the type of floor that the position is
                    self.map_vals[y,x] = map_floor_type.get( next_val, -1 )
                
        #close the file connection
        map_file.close()
        
        #end function
        return
        


class RaceCar:
    
    def __init__(
        self
    ):

        return








#determine the types of floor values in the map
map_floor_type = {
    '#': 0
    , 'S': 1
    , '.': 2
    , 'F': 3
}

#open the file to store the racetrack's map contents
with open(map_filepath, 'r') as map_file:
    
    #get the size of the map (1st line -> ex. "24, 32")
    map_size = map_file.readline()
    map_size_array = numpy.fromstring(map_size, dtype=int, sep=',')
    map_vals = numpy.zeros( (map_size_array[0], map_size_array[1]) )
    
    #go through each map position and store the value, and process special points
    for y in range(map_size_array[0]):
        for x in range(map_size_array[1]):
            
            #store the map's position value
            next_val = map_file.read(1)
            
            #skip new line values
            if( next_val == '\n'):
                next_val = map_file.read(1)
            
            #get the type of floor that the position is
            map_vals[y,x] = map_floor_type.get( next_val, -1 )

#
map_file.close()



#TM = TrackManager(map_filepath)




#Set the boundaries for the velocity/acceleration values
min_x = 0
max_x = map_size_array[1] - 1
min_y = 0
max_y = map_size_array[0] - 1
min_Vx = -5
max_Vx = 5
min_Vy = -5
max_Vy = 5
min_Ax = -1
max_Ax = 1
min_Ay = -1
max_Ay = 1

#size of all positions (y,x)
all_position_states =  map_size_array[0] * map_size_array[1]

#size of all possible velocities (Vy, Vx) from -5, ..., +5 = (11,11)
all_V_states = (max_Vy - min_Vy + 1) * (max_Vx - min_Vx + 1)

#size of all possible accelerations (Ay, Ax) from
num_actions_Ay = 3
num_actions_Ax = 3
all_actions = num_actions_Ay * num_actions_Ax

#create the table of possible values for all states (rows) given actions (columns)
Q = numpy.zeros( (all_position_states * all_V_states, all_actions) )




#the breakout of this table has a unique state for each record
#a state is defined by a particular position and velocity
#unique row is defined as a unique combination of (Vy, Vx, y, x)
def get_record_by_state(
    Vy              #integer
    , Vx            #integer
    , y             #integer
    , x             #vector of integer
    , min_Vy = 0    #integer
    , max_Vy = 0    #integer
    , min_Vx = 0    #integer
    , max_Vx = 0    #integer
    , min_y = 0     #integer
    , max_y = 0     #integer
    , min_x = 0     #integer
    , max_x = 0     #integer
):
    #Find the total amount of possibilities for each bucket
    num_x = (max_x - min_x) + 1
    num_y_x = (max_y - min_y + 1) * num_x
    num_Vy_y_x = (max_Vx - min_Vx + 1) * num_y_x
    
    #Find the number of possibilities given for the input
    m_y = (y - min_y) * num_x
    m_Vx = (Vx - min_Vx) * num_y_x + m_y
    m_Vy = (Vy - min_Vy) * num_Vy_y_x + m_Vx
    
    #end of function
    return( m_Vy + (x - min_x) )


#
print( get_record_by_state( min_Vy, min_Vx, min_y, min_x ) ) 
print( get_record_by_state( min_Vy, min_Vx, min_y, max_x ) ) 
print( get_record_by_state( min_Vy, min_Vx, min_y+1, min_x ) ) 
print( get_record_by_state( min_Vy, min_Vx, max_y, max_x ) ) 
print( get_record_by_state( min_Vy, min_Vx+1, min_y, min_x ) ) 
print( get_record_by_state( min_Vy, max_Vx, max_y, max_x ) ) 
print( get_record_by_state( min_Vy+1, min_Vx, min_y, min_x ) ) 
print( get_record_by_state( max_Vy, max_Vx, max_y, max_x ) ) 



#
get_record_by_state(
    Vy = max_Vy
    , Vx = max_Vx
    , y = max_y
    , x = [0, 3, max_x] #
    , min_Vy = min_Vy
    , max_Vy = max_Vy
    , min_Vx = min_Vx
    , max_Vx = max_Vx
    , min_y = min_y
    , max_y = max_y
    , min_x = min_x
    , max_x = max_x
)






def get_reward(
    Vy
    , Vx
    , y
    , x
    , Ay
    , Ax
    , min_
):
    
    #Is the car currently on the track?
    current_car_on_track = map_vals[y,x] in [1, 2]
    
    #Calculate the new state
    new_Vy = Vy + Ay
    new_Vx = Vx + Ax
    new_y = y + new_Vy
    new_x = x + new_Vx
    
    #Adjust velocities to fit restraints
    if( new_Vx > max_Vx ):
        new_Vx = max_Vx
    elif( new_Vx < min_Vx ):
        new_Vx = min_Vx
    
    #Adjust velocities to fit restraints
    if( new_Vy > max_Vy ):
        new_Vy = max_Vy
    elif( new_Vy < min_Vy ):
        new_Vy = min_Vy
    
    #Adjust coordinates to fit within map if car went outside
    if( new_x > max_x ):
        new_x = max_x
    elif( new_x < min_x ):
        new_x = min_x
    
    #Adjust coordinates to fit within map if car went outside
    if( new_y > max_y ):
        new_y = max_y
    elif( new_y < min_y ):
        new_y = min_y
    
    #Is the car going to be on the track?
    future_car_on_track = map_vals[new_y,new_x] in [1, 2]
    
    #determine the path that the car took (can be descending or ascending)
    if( x < new_x ):
        path_x = [i for i in range(x, new_x+1, 1)]
    else: 
        path_x = [i for i in range(x, new_x+1, -1)]
       
    #determine the path that the car took (can be descending or ascending)
    if( y < new_y ):
        path_y = [i for i in range(y, new_y+1, 1)]
    else: 
        path_y = [i for i in range(y, new_y+1, -1)]
    
    #move along the x with constant y then move in y with constant x to find the walls
    path_x_has_wall = sum([map_vals[y,_x] == 0 for _x in path_x]) > 0
    path_y_has_wall = sum([map_vals[_y,new_x] == 0 for _y in path_y]) > 0
    path_has_wall = path_x_has_wall or path_y_has_wall
    
    #move along the x with constant y then move in y with constant x to find the finish
    path_x_has_finish = sum([map_vals[y,_x] == 3 for _x in path_x]) > 0
    path_y_has_finish = sum([map_vals[_y,new_x] == 3 for _y in path_y]) > 0
    path_has_finish = path_x_has_finish or path_y_has_finish
    
    #move on the track without obstruction
    if( current_car_on_track and future_car_on_track and not path_has_wall and not path_has_finish ):
        return(-1)
    
    #move from track across finish line
    if( current_car_on_track and path_has_finish ):
        return(100)
    
    #move from track to the wall
    if( current_car_on_track and path_has_wall ):
        return(-10)
    
    return None





































































