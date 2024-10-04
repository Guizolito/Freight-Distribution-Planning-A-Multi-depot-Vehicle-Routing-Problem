# Multi Depot -Cumulative Capacitated Vehicle Routing Problem (+MDVRP) Solution Approach

The following project originated from a project during my studies in Transportation & Logistics Management @ University of Twente. The gurobi optimization tool was used as a solver on a Python environment.

The problem can be divided into 5 different sections:

Part 1 - MDVRP w/ Return to the Closest Depot
It is expected that each truck returns to one of the depots at the end of the route.

Part 2 - MDVRP w/ No Return to Depot
It is not expected that each truck returns to one of the depots at the end of the route.

Part 3 - MD-CCVRP w/ Return to the Closest Depot
Minimize total delivery time and it is expected that each truck returns to one of the depots at the end of the route.

Part 4 - MD-CCVRP w/ No Return to Depot
Minimize total delivery time and it is not expected that each truck returns to one of the depots at the end of the route.

Part 5 - MD-VRP w/ 3 Depots - w/ and w/o Return to Depot
Minimize total cost for transportation & depot location.
