import os
import pandas as pd
import numpy as np
import math
import time


INF = math.inf #define positive infinity

def get_list_of_vertices(schedule:pd.DataFrame)->list:
    stations_dep = set(schedule['depature_station'])#get all unique stations from departures
    stations_arrival = set(schedule['arrival_station'])#get all unique stations from arrivals
    all_stations = set().union(stations_dep,stations_arrival)#union all stations

    return list(all_stations)

def graph_min_price_matrix(vertices: list, schedule:pd.DataFrame):
    
    #Create matrix with infinity prices
    vertices_amount = len(vertices)
    matrix_prices = np.zeros((vertices_amount,vertices_amount),dtype='float64')
    matrix_trains = np.zeros((vertices_amount,vertices_amount),dtype='int64')
    for row in range(vertices_amount):
        for col in range(vertices_amount):
            matrix_prices[row][col] = INF
            matrix_trains[row][col] = -1
    
    #create dataframe with minimal voyage prices
    prices_and_trains = schedule.loc[:,['train','price','depature_station','arrival_station']]
    min_price_trains = prices_and_trains.groupby(['depature_station','arrival_station'], as_index=False)
    min_price_trains = min_price_trains.apply(lambda x: x.nsmallest (1,['price']))
    min_price_trains = min_price_trains.reset_index(level=0, drop=True)

    #loop through schedule and fill min prices for edges
    for i, row in min_price_trains.iterrows():
        matrix_prices[vertices.index(row['depature_station']),vertices.index(row['arrival_station'])] = row['price']
        matrix_trains[vertices.index(row['depature_station']),vertices.index(row['arrival_station'])] = row['train']

    return matrix_prices,matrix_trains

def create_costs_matrix(matrix):
    C = matrix
    min_in_rows = C.min(axis=1).reshape(C.shape[0],1)
    matrix_min_in_rows = np.tile(min_in_rows,(1, C.shape[0]))
    C = C - matrix_min_in_rows
    min_in_cols = C.min(axis=0).reshape(1,C.shape[0])
    matrix_min_in_cols = np.tile(min_in_cols,(C.shape[0],1))
    C = C - min_in_cols
    R = min_in_rows.sum()+min_in_cols.sum()
    return C,R
def branch_bound(matrix_prices,matrix_trains,stations):
    

    #Choose cell with max penalty
    min_price = INF
    for k in range(matrix_prices.shape[0]):
        C,R = create_costs_matrix(matrix_prices)
        temp_stations = stations.copy()
        path = []
        start_cell = k
        
        while C.shape[0] != 2:
            min_penalty,new_C,new_cell = INF,C,1
            prev_shape = C.shape
            path.append(temp_stations[start_cell])
            temp_stations.pop(start_cell)
            for i in range(C.shape[0]):
                if i == start_cell:
                    continue

                C_temp = C
                C_temp[i][start_cell] = INF
                C_temp = np.delete(C_temp, i, 1)
                C_temp = np.delete(C_temp,start_cell,0)
                C_temp,R_temp = create_costs_matrix(C_temp)

                cost = R_temp+R+C[start_cell][i]
                if min_penalty > cost:

                    min_penalty = cost
                    new_C = C_temp
                    new_cell = i

            if prev_shape == new_C.shape:
                break

            C = new_C
            R = min_penalty
            start_cell = new_cell
            

        if C.shape[0] == 2 and min_price > R:
            min_price = R
            path.append(temp_stations[start_cell])
            temp_stations.pop(start_cell)
            path.extend(temp_stations)
            final_path = path
    print(min_price,final_path)

def brute_force(matrix_prices,matrix_trains,stations):
    number_of_verticies = matrix_prices.shape[0]

    def recursive_search(cur_vertex,used,path,trains):
        used[cur_vertex] = True#mark current station as used

        if len(path) == number_of_verticies:#exit recursion if all stations is used
            return path,0,trains

        #init final values
        final_path = []
        lowest_price = INF
        used_trains = []

        #loop through all unused stations and check if Hamiltonian path exists
        for i in range(number_of_verticies):

            if not used[i] and matrix_prices[cur_vertex][i] != INF:

                new_path,new_price,new_trains= recursive_search(i,
                                                used.copy(),
                                                path+[stations[i]],
                                                trains+[matrix_trains[cur_vertex][i]])
                new_price += matrix_prices[cur_vertex][i]

                if len(new_path) == number_of_verticies and lowest_price > new_price:#check if we found Hamiltonian path
                    #update final values
                    final_path = new_path
                    lowest_price = new_price
                    used_trains = new_trains

        return final_path,lowest_price,used_trains#return lowest price path

    #init final values
    optimal_path = []
    lowest_price = INF
    trains_used = []

    #loop though all stations and use them as the begining vertex
    for start_vertex in range(number_of_verticies):
        
        #Try to find Hamiltonian path
        used = [False]*number_of_verticies
        path,price,trains = recursive_search(start_vertex,used,[stations[start_vertex]],[])

        #check if more optimal path is found
        if lowest_price > price:
            #update values
            trains_used = trains
            optimal_path = path
            lowest_price = round(price,2)

    return optimal_path,lowest_price,trains_used
def greedy_alg(matrix_prices,matrix_trains,stations):
    
    number_of_verticies = matrix_prices.shape[0]

    #Init final values
    optimal_path = []
    lowest_price = INF
    trains_used = []
    cost_of_each_trip = []

    #loop through all stations and use them as the begining of the path
    for start_vertex in range(number_of_verticies):
        costs = []
        path,path_price = [stations[start_vertex]],0
        trains = []
        cur_vertex = start_vertex

        while len(path) != number_of_verticies:

            minimum_cost_edge,new_vertex= INF,-1

            #loop through unused stations and pick train with the lowest price
            for index,value in enumerate(matrix_prices[cur_vertex]):
                if minimum_cost_edge > value and not stations[index] in path:
                    minimum_cost_edge = value
                    new_vertex = index

            #check if did not find any path
            if new_vertex == -1:
                break

            #expand path,costs,trains
            costs.append(minimum_cost_edge)
            trains.append(matrix_trains[cur_vertex][new_vertex])
            path.append(stations[new_vertex])
            
            #update path price
            path_price+=minimum_cost_edge

            #use new station as a current stations
            cur_vertex = new_vertex
        
        #check if found more optimal path
        if new_vertex != -1 and round(path_price,2) < round(lowest_price,2):
            #update final values
            cost_of_each_trip = costs
            lowest_price = path_price
            optimal_path = path    
            trains_used = trains

    return optimal_path,round(lowest_price,2),trains_used,cost_of_each_trip
            
if __name__ == "__main__":
    os.system("clear")#clear console

    schedule = pd.read_csv("test_task_data.csv",sep=";")#read schedule from csv file
    stations = get_list_of_vertices(schedule)#import only distinct stations
    matrix_prices,matrix_trains = graph_min_price_matrix(stations,schedule)#build matrix of our graph
    
    timer_begin = time.time()
    path,price,trains,cost_of_each_trip = greedy_alg(matrix_prices,matrix_trains,stations)
    timer_end = time.time()
    
    greedy_alg_time = round(timer_end-timer_begin,5)

    print("Price of the travel:", price)
    print("Trains is used:",trains)
    print("Order of visiting stations",path)
    print("Costs of each trip:",cost_of_each_trip)

    timer_begin = time.time()
    brute_force(matrix_prices,matrix_trains,stations)
    timer_end = time.time()

    brute_force_alg_time = round(timer_end-timer_begin,5)

    print("Greedy algorithm time:",greedy_alg_time)
    print("Brute force algorith time:",brute_force_alg_time)
    