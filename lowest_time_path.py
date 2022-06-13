import os
import pandas as pd
import numpy as np
import math
import time

#define constants
INF = math.inf 
SECONDS_IN_DAY = 24*60*60

def get_list_of_vertices(schedule:pd.DataFrame)->list:
    stations_dep = set(schedule['depature_station'])#get all unique stations from departures
    stations_arrival = set(schedule['arrival_station'])#get all unique stations from arrivals
    all_stations = set().union(stations_dep,stations_arrival)#union all stations

    return list(all_stations)

def adjacency_matrix(vertices: list, schedule:pd.DataFrame):
    
    #Create matrix with infinity prices
    vertices_amount = len(vertices)
    matrix = np.zeros((vertices_amount,vertices_amount),dtype='int64')

    #create all possible groups of stations connections
    stations_connections = schedule.groupby(['depature_station','arrival_station']).agg(lambda x: ','.join(str(x)))
   
    #loop through schedule and fill all connections
    for i, row in stations_connections.iterrows():
        matrix[vertices.index(i[0]),vertices.index(i[1])] = 1

    return matrix

def time_to_seconds(t:str)->int:
    h,m,s = map(int,str(t).split(':'))
    return h*60*60+m*60+s

def fastest_path_beetwen_stations(dep_station:int,arv_station:int,cur_time:int,
                                    schedule:pd.DataFrame,is_first_train = False):
    lowest_time = INF
    train = None
    schedule = schedule.loc[(schedule['depature_station'] == dep_station) 
                            & (schedule['arrival_station'] == arv_station)]
    for i, row in schedule.iterrows():

        #convert time to seconds
        arv_t = time_to_seconds(row['arrival_time'])
        dep_t = time_to_seconds(row['depature_time'])

        #count travel time
        if arv_t >= dep_t:
            travel_time = arv_t-dep_t  
        else:
            travel_time = SECONDS_IN_DAY-dep_t+arv_t

        #count delay time
        if is_first_train:
            wait_time = 0
        elif cur_time <= dep_t:
            wait_time = dep_t-cur_time 
        else:
            wait_time =SECONDS_IN_DAY-cur_time+dep_t
        
        sum_time = travel_time+wait_time

        #check for minimum time
        if lowest_time > sum_time:
            lowest_time = sum_time
            arrive_at = arv_t
            train = row['train']

    return lowest_time,train,arrive_at

def brute_force(matrix:np.ndarray,stations:list,schedule:pd.DataFrame):
    number_of_verticies = matrix.shape[0]
    
    def recursive_search(cur_vertex,used,path,trains,cur_time = 0):
        used[cur_vertex] = True #mark current station as used
        
        if len(path) == number_of_verticies:#exit recursion if all stations is used
            return path,0,trains

        #init final values
        final_path = []
        lowest_time = INF
        used_trains = []

        #loop through all unused stations and check if Hamiltonian path exists
        for i in range(number_of_verticies):

            if not used[i] and matrix[cur_vertex][i] != 0:
                is_first_train = len(path) == 1
                travel_time,train,arrive_at = fastest_path_beetwen_stations(stations[cur_vertex],
                                                                    stations[i],cur_time,
                                                                    schedule,is_first_train)

                new_path,new_time,new_trains= recursive_search(i,
                                                used.copy(),
                                                path+[stations[i]],
                                                trains+[train],
                                                (arrive_at)%SECONDS_IN_DAY)
                new_time += travel_time
                
                if len(new_path) == number_of_verticies and lowest_time > new_time:#check if we found Hamiltonian path
                    #update final values
                    final_path = new_path
                    lowest_time = new_time
                    used_trains = new_trains

        return final_path,lowest_time,used_trains#return lowest price path

    #init final values
    optimal_path = []
    lowest_time = INF
    trains_used = []

    #loop though all stations and use them as the begining vertex
    for start_vertex in range(number_of_verticies):
        
        #Try to find Hamiltonian path
        used = [False]*number_of_verticies
        path,tim,trains = recursive_search(start_vertex,used,[stations[start_vertex]],[])

        #check if more optimal path is found
        if lowest_time > tim:
            #update values
            trains_used = trains
            optimal_path = path
            lowest_time = tim

    return optimal_path,lowest_time,trains_used
def get_pathes_beetwen_stations(dep_station,arv_station,cur_time,schedule,is_first_train):
    
    trains = []
    schedule = schedule.loc[(schedule['depature_station'] == dep_station) 
                            & (schedule['arrival_station'] == arv_station)]
    for i, row in schedule.iterrows():

        #convert time to seconds
        arv_t = time_to_seconds(row['arrival_time'])
        dep_t = time_to_seconds(row['depature_time'])

        #count travel time
        if arv_t >= dep_t:
            travel_time = arv_t-dep_t  
        else:
            travel_time = SECONDS_IN_DAY-dep_t+arv_t

        #count delay time
        if is_first_train:
            wait_time = 0
        elif cur_time <= dep_t:
            wait_time = dep_t-cur_time 
        else:
            wait_time =SECONDS_IN_DAY-cur_time+dep_t
        
        sum_time = travel_time+wait_time
        trains.append([sum_time,row['train'],arv_t])

    return trains
def full_brute_force(matrix,stations,schedule):
    number_of_verticies = matrix.shape[0]
    def recursive_search(cur_vertex,used,path,trains,cur_time = 0):
        used[cur_vertex] = True #mark current station as used

        if len(path) == number_of_verticies:#exit recursion if all stations is used
            return path,0,trains

        #init final values
        final_path = []
        lowest_time = INF
        used_trains = []
        is_first_train = len(path) == 1
        
        #loop through all unused stations and check if Hamiltonian path exists
        for i in range(number_of_verticies):

            if not used[i] and matrix[cur_vertex][i] != 0:
                
                connections = get_pathes_beetwen_stations(stations[cur_vertex],stations[i],cur_time,schedule,is_first_train)

                for voyage in connections:

                    new_path,new_time,new_trains= recursive_search(i,
                                                                    used.copy(),
                                                                    path+[stations[i]],
                                                                    trains+[voyage[1]],
                                                                    (voyage[2])%SECONDS_IN_DAY)
                    new_time += voyage[0] 

                    if len(new_path) == number_of_verticies and lowest_time > new_time:#check if we found Hamiltonian path
                        #update final values
                        final_path = new_path
                        lowest_time = new_time
                        used_trains = new_trains

        return final_path,lowest_time,used_trains#return lowest price path

    #init final values
    optimal_path = []
    lowest_time = INF
    trains_used = []

    #loop though all stations and use them as the begining vertex
    for start_vertex in range(number_of_verticies):
        
        #Try to find Hamiltonian path
        used = [False]*number_of_verticies
        path,time,trains = recursive_search(start_vertex,used,[stations[start_vertex]],[])

        #check if more optimal path is found
        if lowest_time > time:
            #update values
            trains_used = trains
            optimal_path = path
            lowest_time = time

    return optimal_path,lowest_time,trains_used
if __name__ == "__main__":
    os.system("clear")#clear console
    
    schedule = pd.read_csv("test_task_data.csv",sep=";")#read schedule from csv file
    stations = get_list_of_vertices(schedule)#import only distinct stations
    connections = adjacency_matrix(stations,schedule)#build matrix of our graph

    timer_begin = time.time()
    path,lowest_time,trains = brute_force(connections,stations,schedule)
    timer_end = time.time()
    
    h = lowest_time//(60*60)
    m = lowest_time%(60*60)//60
    s = lowest_time%60

    
    print("Price of the time:",end=" ")
    print(h,m,s,sep=":")
    print("Trains is used:",trains)
    print("Order of visiting stations:",path,'\n')
    print("Greedy alogrithm + brute force time:",timer_end-timer_begin)
    print()

    """
        For not wasting your time, I commented full_brute_force  
        Best time: 52:31:0
        Price of the time: 52:31:0
        Trains is used: [1156, 1098, 1382, 1035, 1342]
        Order of visiting stations [1981, 1937, 1902, 1929, 1921, 1909] 
        Full brute force time: 398.7164878845215
    """
    true_lowest_time = 189060
    h = true_lowest_time//(60*60)
    m = true_lowest_time%(60*60)//60
    s = true_lowest_time%60

    # timer_begin = time.time()
    # path,true_lowest_time,trains = full_brute_force(connections,stations,schedule)
    # timer_end = time.time()
    
    # h = true_lowest_time//(60*60)
    # m = true_lowest_time%(60*60)//60
    # s = true_lowest_time%60

    
    # print("Price of the time:",end=" ")
    # print(h,m,s,sep=":")
    # print("Trains is used:",trains)
    # print("Order of visiting stations:",path,'\n')
    # print("Full brute force time:",timer_end-timer_begin,'\n')

    error = round(abs(lowest_time-true_lowest_time)/true_lowest_time * 100,2)
    print("Error value: ",error,"%",sep="")
    