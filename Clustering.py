#BRANDON WALSH
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

def main():

    pointsFile = None;
    k = None;

    #parse command line args
    if len(sys.argv) != 3:
        print ('error wrong number of sys args')
    else:
        pointsFile = sys.argv[2]
        k = int(sys.argv[1])


    points = []
    for line in open(pointsFile, 'r'):
        nums = line.split()
        nums = map(float, nums) # turn each string into a float
        points.append(nums)

    d = np.array(points)

    #print d

    #generate the k means
    kmeans(d,k)

# kmeans clustering algorithm
# data = set of data points
# k = number of clusters
# c = initial list of centers (if provided)
#
def kmeans(data, k, c=None):
    centers = []
    centers = randCenters(data, centers, k)
    old_centers = [[] for i in range(k)]
    loops = 0
    while not (convergeClusters(centers, old_centers, loops)):
        loops += 1
        clusters = [[] for i in range(k)]
        # assign data points to clusters
        clusters = distance(data, centers, clusters)
        # recalculate centers
        index = 0
        for cluster in clusters:
            old_centers[index] = centers[index]
            centers[index] = np.mean(cluster, axis=0).tolist()
            index += 1
    print("Num Data Points: " + str(len(data)))
    print("The cluster centers are: " + str(centers))
    print("The clusters are as follows:")
    for cluster in clusters:
        print("Cluster with a size of " + str(len(cluster)) + " starts here:")
        print(np.array(cluster).tolist())
        print("end of cluster")
        color = np.random.rand(3,1)
        for c in np.array(cluster).tolist():
            plt.scatter([c[0]],[c[1]],s=50,c=color)

    #GENERATE THE GRAPH
    #plt.plot(data,'ro')
    for c in centers:
        plt.scatter([c[0]],[c[1]],s=200,c='b')
    plt.ylabel('some numbers')
    plt.show()

    return

# Calculates euclidean distance between
# a data point and all the available cluster
# centers.
def distance(data, centers, clusters):
    for instance in data:
        # Find which centroid is the closest
        # to the given data point.
        mu_index = min([(i[0], np.linalg.norm(instance-centers[i[0]])) \
                            for i in enumerate(centers)], key=lambda t:t[1])[0]
        try:
            clusters[mu_index].append(instance)
        except KeyError:
            clusters[mu_index] = [instance]

    # If any cluster is empty then assign one point
    # from data set randomly so as to not have empty
    # clusters and 0 means.
    for cluster in clusters:
        if not cluster:
            cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())

    return clusters

# randomize initial centers
def randCenters(data, centers, k):
    for cluster in range(0, k):
        centers.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
    return centers

# check if clusters have converged
def convergeClusters(centers, old_centers, loops):
    loopCounterMax = 1000
    if loops > loopCounterMax:
        return True
    return old_centers == centers

main()
