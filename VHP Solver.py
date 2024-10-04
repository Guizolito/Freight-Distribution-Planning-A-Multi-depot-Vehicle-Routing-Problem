#Assigment 1 - Group 27

#%%

k=0

from ctypes.wintypes import RGB
import math
from typing import Set
from mip import Model, xsum, minimize, BINARY
import mip
import numpy as np
import random
import time
import matplotlib.pyplot as plt

from sympy import product

def ScenarioSolver(ScenarioFile): #Entrance variable corresponds to the respective scenario file

    Ncustomers = 0
    NDepots = 0
    M = 1000000


    with open(ScenarioFile,"r") as f:
        Ncustomers = int(f.readline())
        NDepots = int(f.readline())
        CustLocations = np.zeros([Ncustomers+2,2])
        CustDemand = np.zeros([Ncustomers+2])
        LineNumber = 0
        for line in f.readlines():
            if LineNumber < Ncustomers:
                CustLocations[LineNumber][0] = float(line.split(" ")[0])  #Defines the x coordinate
                CustLocations[LineNumber][1] = float(line.split(" ")[1])  #Defines the y coordinate
            elif (LineNumber >= Ncustomers) and (LineNumber < Ncustomers*2):
                CustDemand[LineNumber-Ncustomers] = float(line)
            LineNumber += 1
        f.close()


    CustLocations[Ncustomers+1][0] = 25
    CustLocations[Ncustomers+1][1] = 5
    CustLocations[Ncustomers][0] = 1
    CustLocations[Ncustomers][1] = 50

    CustDemand[Ncustomers + 1] = 0
    CustDemand[Ncustomers] = 0

    #Calculation of distances between customers

    DistanceMatrix = np.zeros([len(CustLocations),len(CustLocations)]) #This matrix will provide the distance between each Customer with the same size as the position list

    for i in range(len(DistanceMatrix)):
        for j in range(len(DistanceMatrix)):
            if i == j:
                DistanceMatrix[i][j] = M #The distance of a customer to itself is 0
            else:
                xDistance = (int(abs(CustLocations[i][0]-CustLocations[j][0])))**2
                yDistance = (int(abs(CustLocations[i][1]-CustLocations[j][1])))**2
                DistanceMatrix[i][j] = math.sqrt(xDistance+yDistance) #Euclidean distance between i and j

                # Note that this will create a symmetric matrix across the main diagonal. This could be simplified by simply making an upper triangular matrix
          
    
    #Cost matrix depends on the model utilized and it is equal to the product between the distance matrix and the Cost per travelled unit distance
    
    

    #Initializing the model

    model = Model(solver_name="GRB")

    #Binary variables indicating if the arc between customer i and j is used or not

    arcBin = [[[model.add_var(var_type=BINARY) for k in range(8)] for j in range(len(CustLocations))] for i in range(len(CustLocations))]

    UsageMatrix = [model.add_var(var_type=BINARY) for k in range(8)]

    #Variable preventing subtours - The next customer to be visited will always be different

    SubTourPrev = [model.add_var() for i in range(Ncustomers)]

    #The following lists were used in order to simplify the objective function and reduce it to 2 sums

    V = set(range(len(CustLocations)))
    V1 = set(range(Ncustomers))
    NTrucks = set(range(len(UsageMatrix)))


    #From Table 1 - Model A:

    TruckDistCost = [0.25,0.25,0.25,0.2,0.2,0.2,0.3,0.3]
    TruckRentCost = [295,295,295,325,325,325,395,395] 
    TruckCapacity = [13,13,13,16,16,16,20,20]
    MaxRouteLength = [500,500,500,600,600,600,700,700]
    TruckTime = [0.1,0.1,0.1,0.2,0.2,0.2,0.15,0.15]
    ServiceTime = [10,10,10,9,9,9,9,9]

    #From Table 2:

    #Objective function - Minimizes rent costs and travelling costs

    model.objective = minimize(xsum(arcBin[i][j][k] * DistanceMatrix[i][j] * TruckTime[k] for k in NTrucks for j in V for i in V) + xsum(ServiceTime[k] * arcBin[i][j][k] for k in NTrucks for j in V for i in V))

    #Constraints:

    # Only leave a city once
    for i in V1:
            model += xsum(arcBin[i][j][k] for k in NTrucks for j in V) == 1

    # Only enter a city once
    for j in V1:
            model += xsum(arcBin[i][j][k] for k in NTrucks for i in V) == 1

    # Always enters and leaves a city

    for k in NTrucks:
        for h in V1:
            model += xsum(arcBin[i][h][k] for i in V) - xsum(arcBin[h][j][k] for j in V) == 0     

    # Subtour elimination

    for k in NTrucks:
        for i in V1:
            for j in V1:
                if i != j:    
                    model += SubTourPrev[i] - SubTourPrev[j] + (Ncustomers + NDepots) * arcBin[i][j][k] <= Ncustomers + NDepots - 1 

    # Starts at the depot

    for k in NTrucks:

        model += xsum(arcBin[i][j][k] for j in V1 for i in range(Ncustomers,Ncustomers + NDepots)) <= 1

    # Ends at the depot

    for k in NTrucks:

        model += xsum(arcBin[i][j][k] for i in V1 for j in range(Ncustomers,Ncustomers + NDepots)) <= 1

    # Number of trucks

    model += xsum(UsageMatrix[k] for k in range(0,3)) <= 3

    model += xsum(UsageMatrix[k] for k in range(3,6)) <= 3

    model += xsum(UsageMatrix[k] for k in range(6,8)) <= 2

    #Usage Matrix

    for k in NTrucks:

        model += xsum(arcBin[i][j][k] for j in V for i in V) <= M * UsageMatrix[k]

    # Max route length per truck

    for k in NTrucks:

        model += xsum(DistanceMatrix[i][j] * arcBin[i][j][k] for j in V for i in V) <= MaxRouteLength[k]

    # Capacity restriction

    for k in NTrucks:

        model += xsum(CustDemand[j] * arcBin[i][j][k] for j in V for i in V) <= TruckCapacity[k]
        
    # Optimize

    model.optimize(max_seconds=120)

    ModelSolution = model.objective_value
    print("Model solution: " + str(ModelSolution))
                
    

    SumK = 0
    for k in NTrucks:
        if UsageMatrix[k].x > 0.99:
            SumK +=1

    print(SumK)
    #plt.plot(CustLocations[Ncustomers][0],CustLocations[Ncustomers][1], marker = "p", color="red")
    #plt.plot(CustLocations[Ncustomers+1][0],CustLocations[Ncustomers+1][1], marker = "p", color="red")


    # Plotting
    DifferentLocation = [(i, j, k) for i in V for j in V for k in NTrucks if arcBin[i][j][k].x > 0.99]
    colors = ["w","g","k","y","m","c","blue","r","black"]
    for i, j, k in DifferentLocation:
        if k == 0:
            plt.plot([CustLocations[i][0], CustLocations[j][0]], [CustLocations[i][1], CustLocations[j][1]], c='black', linestyle="-")
        elif k == 1:
            plt.plot([CustLocations[i][0], CustLocations[j][0]], [CustLocations[i][1], CustLocations[j][1]], c='green', linestyle="-")
        elif k == 2:
            plt.plot([CustLocations[i][0], CustLocations[j][0]], [CustLocations[i][1], CustLocations[j][1]], c='orange', linestyle="-")
        elif k == 3:
            plt.plot([CustLocations[i][0], CustLocations[j][0]], [CustLocations[i][1], CustLocations[j][1]], c='yellow', linestyle="-")
        elif k == 4:
            plt.plot([CustLocations[i][0], CustLocations[j][0]], [CustLocations[i][1], CustLocations[j][1]], c='brown', linestyle="-")
        elif k == 5:
            plt.plot([CustLocations[i][0], CustLocations[j][0]], [CustLocations[i][1], CustLocations[j][1]], c='pink', linestyle="-")
        elif k == 6:
            plt.plot([CustLocations[i][0], CustLocations[j][0]], [CustLocations[i][1], CustLocations[j][1]], c='blue', linestyle="-")
        else:
            plt.plot([CustLocations[i][0], CustLocations[j][0]], [CustLocations[i][1], CustLocations[j][1]], c='red', linestyle="-")

    plt.plot(CustLocations[Ncustomers][0], CustLocations[Ncustomers][1], ls="--", c='r', marker='s')
    plt.plot(CustLocations[Ncustomers+1][0], CustLocations[Ncustomers+1][1], ls="--", c='g', marker='s')
    plt.scatter(CustLocations[:Ncustomers+2, 0], CustLocations[:Ncustomers+2, 1], ls="--")
    plt.show()

    DifferentLocation = np.array(DifferentLocation)

ScenarioSolver("C:\\Users\\gui_t\\Desktop\\Assignment 1\\Scenarios\\Scenario1.txt")
#%%





# %%
