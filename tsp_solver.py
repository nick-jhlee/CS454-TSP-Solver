"""
Created on 2020/10/02
@author: nicklee

(Description)
"""

import argparse
import csv
import time
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}

import hdbscan
from TSP_ACO import *


# Helper functions

def metric(x, y):
    """
    Calculates the Euclidean distance between two points in 2D

    :param x: point (tuple)
    :param y: point (tuple)
    :return: distance between x and y (float)
    """
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def export_csv(l) -> None:
    """
    Wrutes the input list(l) to a single column in a new csv file

    :param l: input list
    :return: None
    """
    with open('solution.csv', 'w') as f:
        writer = csv.writer(f)
        for val in l:
            writer.writerow([val])


# Solve given (sub-)TSP problem using TSP
# Hyperparameter tuning of TSP is done using PSO


# Parse arguments
parser = argparse.ArgumentParser(description='Symmetric (2D Euclidean) Metric TSP Solver\n author: nick-jhlee')
parser.add_argument('filename', type=str, nargs=1, help='Name of the .tsp file to solve')
parser.add_argument("-v", "--verbose", help="Enable verbose", action="store_true")

# parser.add_argument('solver', type=str, nargs=1, help='Type of solver to be used')

# Process the input file
args = parser.parse_args()
filename = args.filename[0]
verbose = args.verbose

"""
(assumed) .tsp format:

NAME : ~~~~~
COMMENT : ~~~~~~~~
TYPE : TSP
DIMENSION : (int)
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
num coordinate1 coordinate2
num coordinate1 coordinate2
...
"""

# Extract essential information and the coordinates of the cities
dim_cities = 2  # dimension that the cities lie in
cities = []  # coordinates of cities, as a list of lists
with open(filename) as raw_tsp:
    i = 0
    for line in raw_tsp:
        if str(line) == 'EOF':
            break
        if i == 0:
            tsp_name = line[7:].strip()
        if i == 3:
            num_cities = int(line[12:])
        if i > 5:
            tmp = [float(item.strip()) for item in line.split()]
            cities.append(tmp[1:])
        i += 1
cities = np.array(cities)  # convert to numpy array

print("Solving %s" % tsp_name)
print("number of cities: %d" % num_cities)

### Actual Solver

"""
Step 1: Clustering the cities

Use HDBSCAN(Hierarchical Density-Based Spatial Clustering of Applications with Noise)[8]

min_cluster_size = 
"""

min_cluster_size = max(int(len(cities) * 0.01), 2 * dim_cities)

clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
cluster_labels = clusterer.fit_predict(cities)

# Plot the points and the clusters

plt.scatter(cities.T[0], cities.T[1], c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plt.show()


# code from https://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/Comparing%20Clustering%20Algorithms.ipynb
def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    plt.show()


plot_clusters(cities, hdbscan.HDBSCAN, (), {'min_cluster_size': min_cluster_size})

plt.hist(cluster_labels)
plt.show()

# Divide the points into each cluster
num_clusters = max(cluster_labels) + 1
clustered_cities = []
for i in range(num_clusters):
    idx = [item == i - 1 for item in cluster_labels]
    clustered_cities.append(cities[idx])


# Step 2: (Paralell) Intracluster TSP solver for each cluster
# for cities in clustered_cities:

def solve_TSP(cities):
    tsp_aco = TSP_ACO(cities, metric)

    # Set/Initialize hyperparameters
    T = 100
    num_ants = 10
    a0 = 1
    b0 = 1
    rho0 = 0.1
    Q0 = tsp_aco.initialize_Q()

    return tsp_aco.optimize(T, num_ants, a0, b0, rho0, Q0)


print(solve_TSP(cities))

# Step 3: Intercluster TSP solver (median-based)
