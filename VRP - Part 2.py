#Assigment 1 - Group 27

#%%    #Code divider for VSC

#Libraries utilized

import math
from typing import Set
from mip import Model, xsum, minimize, BINARY
import mip
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sympy import product

# Time Conversion - Converts the time from minutes to its respective units

def TimeConversion(TotalTime):

    Hours = 0
    Minutes = 0
    Seconds = 0
    
    Hours = round(TotalTime / 60)

    if Hours < 0:
        Hours = 0

    Minutes = round((TotalTime / 60 - round(TotalTime / 60)) * 60)

    if Minutes < 0:
        Minutes = 0

    Seconds = round((((TotalTime / 60 - round(TotalTime / 60)) * 60)-round((TotalTime / 60 - round(TotalTime / 60)) * 60)) * 60)

    if Seconds < 0 :
        Seconds = 0

    FinalTime = str(Hours) + "h " + str(Minutes) + "min " + str(Seconds) + "s"

    return FinalTime

#Scenario Solver - Function that receives the file direction, reads it and resolves the LP model

def ScenarioSolver(ScenarioFile, ReturnOrN): #Entrance variable corresponds to the respective scenario file

    # Variable initialization 

    Ncustomers = 0
    NDepots = 0
    M = 1000000

    # File reading

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

    # Individual adding of each Depot to the Location Matrix and Demand Array

    # Depot 2
    
    CustLocations[Ncustomers][0] = 1
    CustLocations[Ncustomers][1] = 50

    CustDemand[Ncustomers] = 0

    # Depot 1

    CustLocations[Ncustomers+1][0] = 25
    CustLocations[Ncustomers+1][1] = 5

    CustDemand[Ncustomers + 1] = 0



    # Calculation of distances between customers

    DistanceMatrix = np.zeros([len(CustLocations),len(CustLocations)]) #This matrix will provide the distance between each Customer with the same size as the position list
    
    # Algorithm to calculate Euclidean distances

    for i in range(len(DistanceMatrix)):
        for j in range(len(DistanceMatrix)):
            if i == j:
                DistanceMatrix[i][j] = M #The distance of a customer to itself is 0
            else:
                xDistance = (int(abs(CustLocations[i][0]-CustLocations[j][0])))**2
                yDistance = (int(abs(CustLocations[i][1]-CustLocations[j][1])))**2
                DistanceMatrix[i][j] = math.sqrt(xDistance+yDistance) #Euclidean distance between i and j

                # Note that this will create a symmetric matrix across the main diagonal. This could be simplified by simply making an upper triangular matrix
          
    
    # Cost matrix depends on the model utilized and it is equal to the product between the distance matrix and the Cost per travelled unit distance
    
    V = set(range(len(CustLocations))) # Equivalent to N+M (Customers + Depots)
    V1 = set(range(Ncustomers))        # Equivalent to N (Only Customers)
    NTrucks = set(range(8))            # Includes all Models

    if ReturnOrN == "y":    # If we're checking for a solution with return to depot
        V2 = V
    else:                   # If we're checking for a solution with NO return to depot
        V2 = V1

    # Initializing the model

    model = Model()

    # Binary variables indicating if the arc between customer i and j is used or not by truck k

    arcBin = [[[model.add_var(var_type=BINARY) for k in NTrucks] for j in V] for i in V]

    UsageMatrix = [model.add_var(var_type=BINARY) for k in NTrucks]

    # Variable preventing subtours - The next customer to be visited will always be different

    SubTourPrev = [model.add_var() for i in V1]

    # Time variable of each truck's arrival to i

    TTime = [[model.add_var() for k in NTrucks] for i in V]

    # Keeps track of the Total Costs and Distance of each Truck

    TotalCosts = model.add_var()

    TotalDistance = [model.add_var() for k in NTrucks]

    #The following lists were used in order to simplify the objective function and reduce it to 2 sums

    #From Table 1:

    TruckDistCost = [0.25,0.25,0.25,0.2,0.2,0.2,0.3,0.3]
    TruckRentCost = [295,295,295,325,325,325,395,395] 
    TruckCapacity = [13,13,13,16,16,16,20,20]
    MaxRouteLength = [500,500,500,600,600,600,700,700]
    TruckTime = [0.1,0.1,0.1,0.2,0.2,0.2,0.15,0.15]
    ServiceTime = [10,10,10,9,9,9,9,9]
    TruckModel = ["A","A","A","B","B","B","C","C"]


    # Objective function - Minimizes delivery time and service time

    model.objective = minimize(xsum(TTime[i][k] for k in NTrucks for i in V1))

    # Parameters:

    TotalCosts = xsum(arcBin[i][j][k] * DistanceMatrix[i][j] * TruckDistCost[k] for k in NTrucks for j in V2 for i in V) + xsum(UsageMatrix[k] * TruckRentCost[k] for k in NTrucks)

    for k in NTrucks:

        TotalDistance[k] = xsum(arcBin[i][j][k] * DistanceMatrix[i][j] for j in V2 for i in V)

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

    for k in NTrucks:
        for h in range(Ncustomers,Ncustomers + NDepots):
            model += xsum(arcBin[h][i][k] for i in V) - xsum(arcBin[j][h][k] for j in V) <= 1
         

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

    model += xsum(UsageMatrix[k] for k in range(0,3)) <= 3 # From Table 1

    model += xsum(UsageMatrix[k] for k in range(3,6)) <= 3 # From Table 1

    model += xsum(UsageMatrix[k] for k in range(6,8)) <= 2 # From Table 1

    #Usage Matrix - Guarantees coherence with binary variables

    for k in NTrucks:

        model += xsum(arcBin[i][j][k] for j in V for i in V) <= M * UsageMatrix[k]

    # Max route length per truck

    for k in NTrucks:

        model += xsum(DistanceMatrix[i][j] * arcBin[i][j][k] for j in V for i in V) <= MaxRouteLength[k]

    # Capacity restriction

    for k in NTrucks:

        model += xsum(CustDemand[j] * arcBin[i][j][k] for j in V for i in V) <= TruckCapacity[k]

    # Time restriction

    for k in NTrucks:
        for i in V:
            for j in V:
                model += TTime[i][k] + ServiceTime[k] * UsageMatrix[k] + TruckTime[k] * DistanceMatrix[i][j] - TTime[j][k] <= (1-arcBin[i][j][k]) * M


    # Optimization


    model.optimize(max_seconds=120)

    ModelSolution = model.objective_value



    # Printing

    print("Results:\n")

    print("Model solution: " + str(ModelSolution))

    print("Total Costs: " + str(TotalCosts.x) + "\n")
    
    
    # Plotting

    DifferentLocation = [(i, j, k) for i in V for j in V2 for k in NTrucks if arcBin[i][j][k].x > 0.99]  # Detects travelled trips

    # Color and label scheme for each truck

    colors = ["grey","green","blue","gold","red","orange","pink","purple","black"]
    Labels = ["M.A-Truck 1","M.A-Truck 2","M.A-Truck 3","M.B-Truck 1","M.B-Truck 2","M.B-Truck 3","M.C-Truck 1","M.C-Truck 2"]

    # Plots each Customer

    for i, j, k in DifferentLocation:
        plt.plot([CustLocations[i][0], CustLocations[j][0]], [CustLocations[i][1], CustLocations[j][1]], c=colors[k], linestyle="-", label = Labels[k])
        
    # Adds Annotation over each Customer
    
    for i in V1:
        DotName = "C" + str(i+1)
        plt.annotate(DotName,(CustLocations[i][0], CustLocations[i][1]), fontweight = "bold")
    
    # Adds Depots individually
    
    plt.plot(CustLocations[Ncustomers][0], CustLocations[Ncustomers][1], ls="--", c='r', marker='s')
    plt.annotate("Depot 1",(CustLocations[Ncustomers][0], CustLocations[Ncustomers][1]), fontweight = "bold")

    plt.plot(CustLocations[Ncustomers+1][0], CustLocations[Ncustomers+1][1], ls="--", c='r', marker='s')
    plt.annotate("Depot 2",(CustLocations[Ncustomers+1][0], CustLocations[Ncustomers+1][1]), fontweight = "bold")
    
    plt.scatter(CustLocations[:Ncustomers+2, 0], CustLocations[:Ncustomers+2, 1], ls="--")
    
    # Titles, Axis names and legend processing
    
    plt.title("Mandatory return to depot", fontsize = 10, fontname="Times New Roman", fontweight = "bold")
    plt.suptitle("Scenario 3 - MD-CCVRP", fontsize = 20, fontname="Times New Roman", fontweight = "bold")
    
    plt.xlabel(" X Coordinate", fontsize = 15, color = "black", fontname="Times New Roman", fontweight = "bold")
    plt.ylabel(" Y Coordinate", fontsize = 15, color = "black", fontname="Times New Roman", fontweight = "bold")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels,handles))

    plt.legend(by_label.values(),by_label.keys())
    plt.grid() 

    # Adds the arrow for a clearer view of the route

    for i,j,k in DifferentLocation:
        xpos = (CustLocations[i][0]+CustLocations[j][0])/2
        ypos = (CustLocations[i][1]+CustLocations[j][1])/2
        xdir = CustLocations[j][0]-CustLocations[i][0]
        ydir = CustLocations[j][1]-CustLocations[i][1]
        plt.annotate("", xytext = (xpos,ypos), xy = (xpos + 0.001*xdir,ypos+0.001*ydir), arrowprops=dict(arrowstyle="->", linewidth=3,color=colors[k]), size = 15)        

    plt.show()

    # Travelled distance printing

    print("Travelled Distances:\n")

    TTDistance = 0

    for k in NTrucks:
        
        if UsageMatrix[k].x > 0.5:

            TTDistance += TotalDistance[k].x
            print("Model " + str(TruckModel[k]) + " Truck " + str(k) + "-> " + str(TotalDistance[k].x))

    print("Total distance travelled: " + str(TTDistance))



# Start of the program on the main code

ReturnOrNot = input("MD-CCVRP Solver - Please type y for MANDATORY RETURN TO DEPOT or n for NO RETURN TO DEPOT")



StartTime = time.time()
ScenarioSolver("C:\\Users\\gui_t\\Desktop\\Assignment 1\\Scenarios\\Scenario1.txt",ReturnOrNot)
EndTime = time.time()

print("Computational Time: " + str(EndTime-StartTime))
#%%


