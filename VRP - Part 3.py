# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:19:15 2022

@author: Admin
"""

#%%

from mip import Model, xsum, minimize, BINARY
import numpy as np
import matplotlib.pyplot as plt
import time

def ScenarioSolver(ScenarioFile): #Entrance variable corresponds to the respective scenario file

    N = 0 #number of customers
    K = 8 #number of vehicles
    M = 0 #number of depots
    A = 1000000
    
    with open(ScenarioFile,"r") as f:
        N = int(f.readline())
        M = int(f.readline())+1
        CustLocations = np.zeros([N+M,2])
        Q = np.zeros([N+M])
        LineNumber = 0
        for line in f.readlines():
            if LineNumber < N:
                CustLocations[LineNumber][0] = float(line.split(" ")[0])  #Defines the x coordinate
                CustLocations[LineNumber][1] = float(line.split(" ")[1])  #Defines the y coordinate
            elif (LineNumber >= N) and (LineNumber < N*2):
                Q[LineNumber-N] = float(line)
            LineNumber += 1
        f.close()

    CustLocations[N][0] = 25
    CustLocations[N][1] = 5
    CustLocations[N+1][0] = 1
    CustLocations[N+1][1] = 50
    CustLocations[N+2][0] = 25
    CustLocations[N+2][1] = 25
    
    # Adds the Depots to the Demand matrix with value 0

    Q[N] = 0
    Q[N+1] = 0
    Q[N+2] = 0

    #Calculation of distances between customers


    c = np.zeros([len(CustLocations),len(CustLocations)]) #This matrix will provide the distance between each Customer with the same size as the position list

    for i in range(len(c)):
        for j in range(len(c)):
            if i == j:
                c[i][j] = A #The distance of a customer to itself is 0
            else:
                xDistance = float(abs(CustLocations[i][0]-CustLocations[j][0]))
                yDistance = float(abs(CustLocations[i][1]-CustLocations[j][1]))
                c[i][j] = np.sqrt(xDistance**2+yDistance**2) #Euclidean distance between i and j

                # Note that this will create a symmetric matrix across the main diagonal. This could be simplified by simply making an upper triangular matrix
            
    #Initializing the model

    model = Model()

    #Binary variable indicating if the arc between customer i and j is used or not

    x = [[[model.add_var(var_type=BINARY) for k in range(K)] for j in range(N+M)] for i in range(N+M)]

    #Variable preventing subtours - The next customer to be visited will always be different

    y = [model.add_var() for i in range(N)]
    
    #Binary variable indicating if the truck k is used or not
    
    z = [model.add_var(var_type=BINARY) for k in range(K)]
    
    #Binary variable indicating if the depot k is used or not
    
    w = [model.add_var(var_type=BINARY) for i in range(M)]

    #The following lists were used in order to simplify the objective function and reduce it to 2 sums
    #From Table 1:

    C = [0.25, 0.25, 0.25,0.2, 0.2, 0.2, 0.3, 0.3] 
    R = [295, 295, 295, 325, 325, 325, 395, 395]  
    P = [13, 13, 13, 16, 16, 16, 20, 20]
    T = [500, 500, 500, 600, 600, 600, 700, 700]
    
    #Objective function - Minimizes rent costs and travelling costs

    model.objective = minimize(xsum(x[i][j][k] * c[i][j] * C[k] for k in range(K) for j in range(N) for i in range(N+M)) + xsum(z[k] * R[k] for k in range(K)) + 200 + 225 + (180+175) * w[2])

    #Constraints:

    # Only leave a customer once
    for i in range(N):
        model += xsum(x[i][j][k] for k in range(K) for j in range(N+M)) == 1

    # Only enter a customer once
    for j in range(N):
        model += xsum(x[i][j][k] for k in range(K) for i in range(N+M)) == 1

    # Subtour elimination
    for k in range(K):
        for j in range(N):
            for i in range(N):
                if i!=j:
                    model += y[i] - y[j] + (M+N)*x[i][j][k] <= N+M-1
                    
    #If enters in one customer it has to departure from there as well
    
    for k in range(K):
        for h in range(N):
            model += xsum(x[i][h][k] for i in range(N+M)) - xsum(x[h][j][k] for j in range(N+M)) == 0
    
    #Max Capacity
    
    for k in range(K):
        model += xsum(Q[i] * x[i][j][k] for j in range(N+M) for i in range(N+M)) <= P[k]
        
    #Max Route Length
    
    for k in range(K):
        model += xsum(c[i][j] * x[i][j][k] for j in range(N+M) for i in range(N+M)) <= T[k]
    
    #each truck can departure from at most one depot
    
    for k in range(K):
        model += xsum(x[i][j][k] for i in range(N, N+M) for j in range(N)) <= 1
        
    #each truck can arrive to at most one depot
    
    for k in range(K):
        model += xsum(x[i][j][k] for j in range(N, N+M) for i in range(N)) <= 1
        
    #Usage Matrix
    
    for k in range(K):
        model += xsum(x[i][j][k] for j in range(N+M) for i in range(N+M)) <= A * z[k]
    
    #Depot Matrix 1
    
    for i in range(N, N+M):
        model += xsum(x[i][j][k] for k in range (K) for j in range(N+M)) <= A * w[i-N]
        
    #Depot Matrix 2
    
    for j in range(N, N+M):
        model += xsum(x[i][j][k] for k in range (K) for i in range(N+M)) <= A * w[j-N]
        
    
    model.optimize(max_seconds=30)

    ModelSolution = model.objective_value
    print("Model solution: " + str(ModelSolution))
    
    #Printing the solution
    for i in range(N+M):
        for j in range(N+M):
            for k in range(K):
                if x[i][j][k].x >= 0.5:
                    print("Truck " + str(k+1) + " goes from customer " + str(i+1) + " to customer " + str(j+1))   


# Plotting
    DifferentLocation = [(i, j, k) for i in range(N+M) for j in range(N) for k in range(K) if x[i][j][k].x > 0.99]
    colors = ["grey","green","blue","gold","red","orange","pink","purple","black"]
    Labels = ["M.A-Truck 1","M.A-Truck 2","M.A-Truck 3","M.B-Truck 1","M.B-Truck 2","M.B-Truck 3","M.C-Truck 1","M.C-Truck 2"]

    for i, j, k in DifferentLocation:
        plt.plot([CustLocations[i][0], CustLocations[j][0]], [CustLocations[i][1], CustLocations[j][1]], c=colors[k], linestyle="-", label = Labels[k])
        
    for i in range(N):
        DotName = "C" + str(i+1)
        plt.annotate(DotName,(CustLocations[i][0], CustLocations[i][1]), fontweight = "bold")
    plt.plot(CustLocations[N][0], CustLocations[N][1], ls="--", c='r', marker='s')
    plt.annotate("Depot 1",(CustLocations[N][0], CustLocations[N][1]), fontweight = "bold")
    plt.plot(CustLocations[N+1][0], CustLocations[N+1][1], ls="--", c='r', marker='s')
    plt.annotate("Depot 2",(CustLocations[N+1][0], CustLocations[N+1][1]), fontweight = "bold")
    plt.plot(CustLocations[N+2][0], CustLocations[N+2][1], ls="--", c='blue', marker='s')
    plt.annotate("Depot 3",(CustLocations[N+2][0], CustLocations[N+2][1]), fontweight = "bold")
    plt.scatter(CustLocations[:N+2, 0], CustLocations[:N+2, 1], ls="--")
    plt.title("No return to depot", fontsize = 10, fontname="Times New Roman", fontweight = "bold")
    plt.suptitle("Scenario 3 - 3 depot MDVRP", fontsize = 20, fontname="Times New Roman", fontweight = "bold")
    plt.xlabel(" X Coordinate", fontsize = 15, color = "black", fontname="Times New Roman", fontweight = "bold")
    plt.ylabel(" Y Coordinate", fontsize = 15, color = "black", fontname="Times New Roman", fontweight = "bold")
    handles, labels = plt.gca().get_legend_handles_labels()
    
    
    by_label = dict(zip(labels,handles))
    plt.legend(by_label.values(),by_label.keys())
    plt.grid()
    for i,j,k in DifferentLocation:
        xpos = (CustLocations[i][0]+CustLocations[j][0])/2
        ypos = (CustLocations[i][1]+CustLocations[j][1])/2
        xdir = (CustLocations[j][0]-CustLocations[i][0])
        ydir = (CustLocations[j][1]-CustLocations[i][1])
        plt.annotate("", xytext = (xpos,ypos), xy = (xpos + 0.001*xdir,ypos+0.001*ydir), arrowprops=dict(arrowstyle="->", linewidth=3,color=colors[k]), size = 15)        

    plt.show()
    
    DifferentLocation = np.array(DifferentLocation)
    
StartTime = time.time()
ScenarioSolver("C:\\Users\\gui_t\\Desktop\\Assignment 1\\Scenarios\\Scenario3.txt")
EndTime = time.time()
print("Computational time: " + str(EndTime-StartTime))
#%%