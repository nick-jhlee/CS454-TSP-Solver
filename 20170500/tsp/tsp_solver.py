"""
Created on 2020/10/02
@author: nicklee

(Description)
"""

# TO-DO
# Include progress bar...
# Fix parallel process (pool2 inside pool1...)
# More intricate verbose/plotting options
# ... (Refer to the report for the detailed improvements to be done)

import argparse
import copy
import csv
import os
import timeit

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from multiprocessing import Pool
from TSP_ACO import *
from Errors import *


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
parser = argparse.ArgumentParser(description='Symmetric (2D Euclidean) Metric TSP Solver\n author: nick-jhlee'
                                             '\n\n Make sure that the .tsp (and .opt.tour) files match and have '
                                             'no duplicate coordinates!!!')
parser.add_argument('filename', type=str, nargs=1, help='Name of the .tsp file to solve')
parser.add_argument('sol_filename', type=str, nargs='?', help='Name of the .opt.tour solution file')
parser.add_argument("-p", type=int, nargs='?', help="Set the number of ants to be used for ACO"
                                                    "(if clustering is used, this only applies to "
                                                    "intracluster ACOs), Default is 10")
parser.add_argument("-topt", type=int, nargs='?', help="Choose how three_opt is applied (Default is 1)\n"
                                                       "0: not applied at all\n"
                                                       "1: applied at the end, once\n"
                                                       "2: applied at all the steps")
parser.add_argument("-cratio", type=float, nargs='?',
                    help="Choose the ratio of which the optimal number of clusters is "
                         "searched up to (Default is 1)\n")
parser.add_argument("-cpus", type=int, nargs='?',
                    help="If par option is enabled, input how many cores to use (Default is all)\n")
parser.add_argument("-c", "--clustering", help="Enable clustering-based optimization", action="store_true")
parser.add_argument("-t", "--tuning", help="Enable PSO-based hyperparameter tuning of ACO", action="store_true")
parser.add_argument("-plot", "--plotting", help="Enable plotting", action="store_true")
parser.add_argument("-v", "--verbose", help="Enable verbose", action="store_true")
parser.add_argument("-par", "--parallel", help="Enable parallel computing", action="store_true")

# Parse the arguments
args = parser.parse_args()
filename = args.filename[0]
num_ants = args.p
topt = args.topt
cratio = args.cratio
cpus = args.cpus

clustering = args.clustering
parallel = args.parallel
tuning = args.tuning
plotting = args.plotting
verbose = args.verbose

if parallel:
    print("\nParallel option is enabled")
    print("This is used for searching for the optimal number of clusters")
    print("(Windows users may experience errors. In that case, please turn off the parallel computing option)\n")

if tuning:
    print("Hyperparameter tuning option has not been implemented, yet")
    print("Proceeding without tuning option...\n")
tuning = False

if not topt:
    topt = 1
if topt == 2:
    print("\n3-Opt will be applied at every step! (both intracluster and intercluster!)")
    print("Warning: Will take up high high memory, especially if your problem instance is big")
    print("Use at caution!\n")
if topt not in {0, 1, 2}:
    print("\nWrong options for topt!")
    print("Proceeding with topt = 1\n")
    topt = 1

if not cratio:
    cratio = 1
if cratio < 0 or cratio > 1:
    print("\ncratio out of range!")
    print("Proceeding with cratio = 1\n")
    cratio = 1
if cratio:
    print("\nNumber of clusters will be searched up to %.2f%% of the problem size" % (cratio * 100))

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
os.system('pip install -r requirements.txt')
# Extract essential information and the coordinates of the cities
dim_cities = 2  # dimension that the cities lie in
cities = []  # coordinates of cities, as a list of lists
with open(filename) as raw_tsp:
    i = 0
    for line in raw_tsp:
        if str(line) == 'EOF':
            break
        if i == 0:
            line = line.split()
            tsp_name = line[-1].strip()
        if i > 5:
            tmp = [float(item.strip()) for item in line.split()]
            cities.append(tmp[1:])
        i += 1
cities = np.array(cities)  # convert to numpy array
# cities = np.unique(cities, axis=0)  # Remove any duplicate coordinates
N = len(cities)
cities_nums = np.array(list(range(N)))
tsp = TSP_ACO(cities, cities_nums, metric)

print("number of cities: %d" % N)
print("Solving %s.tsp" % tsp_name)

# Actual Solver
start_time = timeit.default_timer()


def solve_TSP(cities, cities_nums, T, num_ants, a0, b0, rho0, tuning):
    """
    Wrapper function for the main optimization.

    :param cities: np array of coordinates of the cities
    :param cities_nums: np array of the names(labels) of the cities
    :param T: maximum amount of time
    :param num_ants: number of ants to be used
    :param a0, b0, rho0: (initial) hyperparameters for the ACO instance
    :param tuning: Boolean for whether enabling PSO-based hyperparameter tuning

    :return: cost, cycle
    """
    tsp_aco = TSP_ACO(cities, cities_nums, metric)

    # Set Q0
    Q0 = tsp_aco.initialize_Q()

    # Incorporate tuning
    # Incorporate graphs
    # Incorporate min/max values for pheromones
    return tsp_aco.optimize(T, num_ants, a0, b0, rho0, Q0)


def reorder_vertices(l, idx, end):
    """
    Helper function #1 for combine_clusters()
    Delete the longer edge of the two edges adjacent to l[idx], and return the resulting reordered list of vertices

    :param l: input list of cities (by numbers, not coordinates)
    :param idx: index to check
    :param end: if True, resulting list has l[idx] has the last element. Else, it is the first element.
    :return: Ordered list
    """
    if len(l) == 1:
        return l

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
    (Locally optimally) combine all cycles

    :param: cycles: list of list of vertices, each consisting a Hamiltonian cycle
    :return: a list of vertices, consisting of the wanted solution
    """
    num_clusters = len(cycles)
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


# From https://en.wikipedia.org/wiki/3-opt
def distance(cities, x, y):
    """
    Helper function for reverse_segment_if_better

    :param cities: np array of coordinates of cities to be considered
    :param x, y: labels of the cities
    :return: distance between cities[x] and cities[y]
    """
    return metric(cities[x], cities[y])


def reverse_segment_if_better(cities, tour, i, j, k):
    """
    If reversing sub-tour would make the tour shorter, then do it.

    :param cities: np array of coordinates of cities to be considered
    :param tour: initial tour
    :param i, j, k: indices to be considered

    :return: tour after reversing some sub-tour
    """
    # Given tour [...A-B...C-D...E-F...]
    A, B, C, D, E, F = tour[i - 1], tour[i], tour[j - 1], tour[j], tour[k - 1], tour[k % len(tour)]
    d0 = distance(cities, A, B) + distance(cities, C, D) + distance(cities, E, F)
    d1 = distance(cities, A, C) + distance(cities, B, D) + distance(cities, E, F)
    d2 = distance(cities, A, B) + distance(cities, C, E) + distance(cities, D, F)
    d3 = distance(cities, A, D) + distance(cities, E, B) + distance(cities, C, F)
    d4 = distance(cities, F, B) + distance(cities, C, D) + distance(cities, E, A)

    if d0 > d1:
        tour[i:j] = list(reversed(tour[i:j]))
        return -d0 + d1
    elif d0 > d2:
        tour[j:k] = list(reversed(tour[j:k]))
        return -d0 + d2
    elif d0 > d4:
        tour[i:k] = list(reversed(tour[i:k]))
        return -d0 + d4
    elif d0 > d3:
        tmp = tour[j:k] + tour[i:j]
        tour[i:k] = tmp
        return -d0 + d3
    return 0


def all_segments(n: int):
    """
    Generate all segments combinations

    :param n: number of cities
    :return: all possible combinations of all the segments
    """
    return ((i, j, k)
            for i in range(n)
            for j in range(i + 2, n)
            for k in range(j + 2, n + (i > 0)))


def three_opt(cities, tour):
    """
    Iterative improvement based on 3 exchange.

    :param cities: np array of coordinates of cities to be considered
    :param tour: initial tour

    :return: locally optimized tour
    """
    tour_test = copy.deepcopy(tour.tolist())
    while True:
        delta = 0
        for (a, b, c) in all_segments(len(tour)):
            delta += reverse_segment_if_better(cities, tour_test, a, b, c)
        if delta >= 0:
            break
    return tour_test


if clustering:
    """
    Step 1: Clustering the cities
    Use KMeans from sklearn.cluster
    Search for the optimal size of the clusters
    """
    if verbose:
        print("\nStep 1: Clustering")

    max_clusters = math.ceil(N * cratio)
    final_clusters = list(range(2, max_clusters + 1))
    C = 1000  # number of maximum restarts for the k-means algorithm

    if verbose:
        print("Step 2: Solve intracluster TSP using ACO")
        print("Step 3: Solve intercluster TSP using ACO (with weaker setting)")
        print("Step 4: Combine the local solutions to produce a global solution")
        if topt == 2:
            print("Apply 3-Opt in each step to obtain the best possible solution")
        print("(Repeat above steps for n_clusters <= %.2f*N to find an optimal number of clusters)\n" % cratio)


    def big_wrapper(n_clusters):
        if verbose:
            print("Solving with %d clusters" % n_clusters)
        # Handles empty clusters by restarting several times!
        # If same problem persists after C times, then pass
        success = False
        for c_iter in range(C):
            kmeans = KMeans(n_clusters=n_clusters).fit(cities)
            cluster_labels = kmeans.labels_
            num_labels = len(set(cluster_labels))
            if num_labels == n_clusters:
                success = True
                break
            c_iter += 1
        if not success:
            EmptyClusterError("There is an empty cluster...")

        # if plotting:
        # Plot the points and the clusters
        # FIX!!

        # Divide the points into each cluster
        num_clusters = max(cluster_labels) + 1
        clustered_cities = []  # coordinates
        clustered_cities_nums = []  # city numbers
        for i in range(num_clusters):
            idx = [item == i for item in cluster_labels]
            clustered_cities.append(cities[idx])
            clustered_cities_nums.append(cities_nums[idx])

        # Step 2: Parallel intracluster TSP solver for each cluster
        # (Initial) Hyperparameters:
        T = 100
        global num_ants
        if not num_ants:
            num_ants = 10
        a0, b0 = 1, 1.1
        rho0 = 0.1

        if False:
            # FIX so that Pool can have Pool
            def wrapper_solve_TSP(i):
                """
                Wrapper function to be used at parallel process
                :param i:
                :return:
                """
                cost, cycle = solve_TSP(np.array(clustered_cities[i]), clustered_cities_nums[i], T, num_ants, a0, b0,
                                        rho0,
                                        tuning)
                # Run 3-opt for each cluster
                cycle = three_opt(cities, cycle)
                cost = tsp.length_path(cycle)
                return cost, cycle

            pool2 = Pool()
            costs, cycles = zip(*pool2.map(wrapper_solve_TSP, range(num_clusters)))
            pool2.close()
            cycles = [item for item in cycles]
        else:
            costs = [None for _ in range(num_clusters)]
            cycles = [None for _ in range(num_clusters)]
            for i in range(num_clusters):
                if len(clustered_cities[i]) > 2:
                    # try:
                    costs[i], tmp = solve_TSP(np.array(clustered_cities[i]), clustered_cities_nums[i], T,
                                              num_ants, a0, b0, rho0, tuning)
                    cycles[i] = tmp.tolist()
                #
                # except:
                #     print("Expected number of clusters: %d" % n_clusters)
                #     print("Current number of clusters: %d " % len(set(cluster_labels)))
                #     EmptyClusterError("Maybe an empty cluster has been passed...?")
                #     # exit(0)
                else:
                    cycles[i] = clustered_cities_nums[i].tolist()
                    if len(clustered_cities[i]) == 1:
                        costs[i] = 0
                    else:
                        costs[i] = metric(clustered_cities[i][0], clustered_cities[i][1])

        # Step 3: Intercluster TSP solver (median-based)
        median_cities = [np.mean(cluster, axis=0) for cluster in clustered_cities]
        median_cities_nums = np.array(range(num_clusters))
        T = 50
        num_ants = 5
        a0, b0 = 1, 1
        rho0 = 0.1
        intercluster_cost, intercluster_path = solve_TSP(median_cities, median_cities_nums, T, num_ants, a0, b0,
                                                         rho0,
                                                         tuning)
        if topt == 2:
            intercluster_path = three_opt(median_cities, intercluster_path)
        median_cities = np.array(median_cities)[intercluster_path].tolist()

        # Step 4: Combine the TSP paths cluster-wise
        # Refer to the report for a detailed description of this step
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
        if topt == 2:
            output1 = three_opt(cities, np.array(combine_clusters(cycles)))
        else:
            output1 = np.array(combine_clusters(cycles))
        output2 = tsp.length_path(output1)
        return output2, output1


    if parallel:
        # Parallel computation for finding the optimal number of clusters
        if cpus and cpus <= os.cpu_count():
            pool1 = Pool(cpus)
        else:
            pool1 = Pool()
        final_costs, final_paths = zip(*pool1.map(big_wrapper, range(2, max_clusters + 1)))
        pool1.close()
    else:
        final_paths = [None for i in range(0, max_clusters - 1)]
        final_costs = [None for i in range(0, max_clusters - 1)]
        for n_clusters in range(2, max_clusters + 1):
            final_costs[n_clusters - 2], final_paths[n_clusters - 2] = big_wrapper(n_clusters)

    # Find the best solution
    opt_cost = min(final_costs)
    best_idx = final_costs.index(opt_cost)
    final_path = final_paths[best_idx]
    final_cluster = final_clusters[best_idx]
    print("Done searching through all possible clusters!")
    print("Optimal number of clusters: %d" % final_cluster)

    if plotting:
        print("\nPlotting...")
        fig1 = plt.figure(1)
        plt.plot(np.array(range(2, max_clusters + 1)), final_costs, marker=".")
        plt.title("final cost vs number of clusters")
        plt.xticks(np.arange(2, max_clusters + 1, step=max(math.floor(N / 10), 1)))
        plt.xlabel('Number of clusters')
        plt.ylabel('(Sub-)optimal cost')

else:
    # If clustering is not used, do vanilla ACO
    T = 1000
    if not num_ants:
        num_ants = 10
    a0, b0 = 1, 1
    rho0 = 0.1
    final_cost, final_path = solve_TSP(cities, cities_nums, T, num_ants, a0, b0, rho0, tuning)

if topt == 1 and topt != 2:
    # If topt=1 i.e. if 3-Opt is applied only at the very end
    final_path = three_opt(cities, final_path)

# Record total running time
running_time = timeit.default_timer() - start_time

# Compute the final_cost
final_cost = tsp.length_path(final_path)

# Export the solution to .csv, and print the resulting cost
final_path = (np.array(final_path) + 1).tolist()
export_csv(final_path)

print("\n")
if verbose:
    print("Took %f sec" % running_time)
print("Total length of the computed Hamiltonian Cycle: %f" % final_cost)

if verbose:
    if args.sol_filename is not None:
        sol_filename = args.sol_filename
        """
        (assumed) .opt.tour format:

        NAME : ~~~~~~~~
        TYPE : TOUR
        COMMENT : Optimal tour for pr76 (108159)
        DIMENSION : (int)
        TOUR_SECTION
        num1
        num2
        ...
        """
        opt_path = []  # ordered list of cities, in which the optimal Hamiltonian cycle is achieved
        with open(sol_filename) as opt_tsp:
            i = 0
            for line in opt_tsp:
                if str(line) == '-1' or str(line) == '-1\n':
                    break
                if i > 4:
                    tmp = line.split()
                    opt_path.append(int(tmp[0].strip()))
                i += 1
        # print(set(opt_path) - set(final_path))
        opt_cost = tsp.length_path(np.array(opt_path) - 1)  # Optimal cost
        opt_ratio = 100 * ((final_cost / opt_cost) - 1)  # How close our algorithm is to the optimal cost
        print("This algorithm is optimal up to %f%% of the optimal cost" % opt_ratio)

if plotting:
    plt.show()
