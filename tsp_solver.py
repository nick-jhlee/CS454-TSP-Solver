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
import hdbscan
from TSP_ACO import *

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}


# Helper functions
def metric(x, y):
    """
    Calculates the Euclidean distance between two points in 2D

    :param x: point
    :param y: point
    :return: distance between x and y (float)
    """
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def export_csv(l) -> None:
    """
    Wrutes the input list(l) to a single column in a new csv file

    :param l: input list
    :return: None
    """
    with open('solution.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for val in l:
            writer.writerow([val])


# Solve given (sub-)TSP problem using TSP
# Hyperparameter tuning of TSP is done using PSO


# Parse arguments
parser = argparse.ArgumentParser(description='Symmetric (2D Euclidean) Metric TSP Solver\n author: nick-jhlee')
parser.add_argument('filename', type=str, nargs=1, help='Name of the .tsp file to solve')
parser.add_argument("-c", "--clustering", help="Enable clustering-based optimization", action="store_true")
parser.add_argument("-p", "--plotting", help="Enable plotting", action="store_true")
parser.add_argument("-t", "--tuning", help="Enable PSO-based hyperparameter tuning of ACO", action="store_true")
parser.add_argument("-v", "--verbose", help="Enable verbose", action="store_true")

# parser.add_argument('solver', type=str, nargs=1, help='Type of solver to be used')

# Process the input file
args = parser.parse_args()
filename = args.filename[0]
verbose = args.verbose
clustering = args.clustering
tuning = args.tuning
plotting = args.plotting

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
cities_nums = np.array(list(range(len(cities))))

print("Solving %s" % tsp_name)
print("number of cities: %d" % num_cities)

# Actual Solver

"""
Step 1: Clustering the cities

Use HDBSCAN(Hierarchical Density-Based Spatial Clustering of Applications with Noise)[8]

min_cluster_size = 
"""

min_cluster_size = max(int(len(cities) * 0.01), 2 * dim_cities)

clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
cluster_labels = clusterer.fit_predict(cities)

if plotting:
    # Plot the points and the clusters
    fig1 = plt.figure()
    fig1.scatter(cities.T[0], cities.T[1], c='b', **plot_kwds)
    frame = fig1.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)


    # code from https://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/Comparing%20Clustering%20Algorithms.ipynb
    def plot_clusters(fig, data, algorithm, args, kwds):
        start_time = time.time()
        labels = algorithm(*args, **kwds).fit_predict(data)
        end_time = time.time()
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        fig.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
        frame = fig.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        fig.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
        fig.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
        # plt.show()


    fig2 = plt.figure()
    plot_clusters(fig2, cities, hdbscan.HDBSCAN, (), {'min_cluster_size': min_cluster_size})

    plt.show()


def solve_TSP(cities, cities_nums, T, num_ants, a0, b0, rho0, tuning):
    tsp_aco = TSP_ACO(cities, cities_nums, metric)

    # Set Q0
    Q0 = tsp_aco.initialize_Q()

    # Incorporate tuning
    # Incorporate graphs
    # Incorporate min/max values for pheromones
    return tsp_aco.optimize(T, num_ants, a0, b0, rho0, Q0)


if clustering:
    # Divide the points into each cluster
    num_clusters = max(cluster_labels) + 1
    clustered_cities = []   # coordinates
    clustered_cities_nums = []   # city numbers
    for i in range(num_clusters):
        idx = [item == i - 1 for item in cluster_labels]
        clustered_cities.append(cities[idx])
        clustered_cities_nums.append(cities_nums[idx])


    # Step 2: (Paralell) Intracluster TSP solver for each cluster
    # for cities in clustered_cities:
    costs = [None for i in range(num_clusters)]
    cycles = [None for i in range(num_clusters)]
    T = 1000
    num_ants = 10
    a0, b0 = 1, 1
    rho0 = 0.1
    for i in range(num_clusters):
        costs[i], tmp = solve_TSP(np.array(clustered_cities[i]), clustered_cities_nums[i], T, num_ants, a0, b0, rho0, tuning)
        cycles[i] = tmp.tolist()

    # Step 3: Intercluster TSP solver (median-based)
    median_cities = [np.mean(cluster, axis=0) for cluster in clustered_cities]
    median_cities_nums = np.array(list(range(-1, num_clusters-1)))
    T = 100
    num_ants = 5
    a0, b0 = 1, 1
    rho0 = 0.1
    intercluster_cost, intercluster_path = solve_TSP(median_cities, median_cities_nums, T, num_ants, a0, b0, rho0, tuning)


    # Step 4: Combine the TSP paths cluster-wise
    # Refer to the report for a detailed description of this step
    def reorder_vertices(l, idx, end):
        """
        Helper function #1 for combine_clusters()
        Delete the longer edge of the two edges adjacent to l[idx], and return the resulting reordered list of vertices

        :param l: input list of cities (by numbers, not coordinates)
        :param idx: index to check
        :param end: if True, resulting list has l[idx] has the last element. Else, it is the first element.
        :return: Ordered list
        """
        if end:
            if idx == 0:
                if metric(cities[l[0]], cities[l[1]]) < metric(cities[l[0]], cities[l[-1]]):
                    l.reverse()
                    return l
                else:
                    return l[1:] + l[:1]
            elif idx == -1 or idx == len(l) - 1:
                if metric(cities[l[-1]], cities[l[-2]]) < metric(cities[l[-1]], cities[l[0]]):
                    return l
                else:
                    l = l[-1:] + l[:-1]
                    l.reverse()
                    return l
            else:
                if metric(cities[l[idx]], cities[l[idx - 1]]) < metric(cities[l[idx]], cities[l[idx + 1]]):
                    return l[idx + 1:] + l[:idx + 1]
                else:
                    l = l[idx:] + l[:idx]
                    l.reverse()
                    return l
        else:
            tmp = reorder_vertices(l, idx, True)
            tmp.reverse()
            return tmp


    def combine_path_cycle(path, cycle):
        """
        Helper function #2 for combine_clusters()
        (Locally optimally) combine path and cycle

        :param path: list of vertices of a path
        :param cycle: list of vertices of a cycle
        :return: combined (and ordered) list of vertices
        """
        x = path[-1]
        max_dist = math.inf
        y = None
        for i in cycle:
            if max_dist > metric(cities[x], cities[i]):
                y = i
        idx_y = cycle.index(y)
        return path + reorder_vertices(cycle, idx_y, False)


    def combine_clusters(cycles):
        """
        :param: cycles: list of list of vertices, each consisting a Hamiltonian cycle
        :return: a list of vertices, consisting of the wanted solution
        """
        # Combine cluster 0 and 1
        max_dist = math.inf
        x, y = None, None
        for i in cycles[0]:
            for j in cycles[1]:
                if max_dist > metric(cities[i], cities[j]):
                    x, y = i, j
        idx_x = cycles[0].index(x)
        idx_y = cycles[1].index(y)

        final_path = reorder_vertices(cycles[0], idx_x, True) + reorder_vertices(cycles[1], idx_y, False)
        if num_clusters > 2:
            for c in range(2, num_clusters):
                final_path = combine_path_cycle(final_path, cycles[c])

        return final_path


    i0 = -1
    tmp = metric(median_cities[-1], median_cities[0])
    for i in range(-1, num_clusters - 1):
        tmpp = metric(median_cities[i], median_cities[i + 1])
        if tmp < tmpp:
            tmp = tmpp
            i0 = i
    if i0 == -1:
        i0 = num_clusters - 1

    # Relabel the clusters!
    cycles = cycles[i0:] + cycles[:i0]
    # Combine the clusters
    final_path = combine_clusters(cycles)

else:
    T = 1000
    num_ants = 10
    a0, b0 = 1, 1
    rho0 = 0.1
    final_cost, final_path = solve_TSP(cities, cities_nums, T, num_ants, a0, b0, rho0, tuning)

# Step 6: 3-Opt Heuristic


# Compute the final_cost
tsp = TSP_ACO(cities, cities_nums, metric)
final_cost = tsp.length_path(final_path)

# Step 5: Export the solution to .csv, and print the resulting cost
export_csv(final_path)
print("Total length of the computed Hamiltonian Cycle: %f" % final_cost)
