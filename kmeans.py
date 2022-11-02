""" 
- The K-Means algorithm was used for this task to create 3 clusters and and allow points to converge.
- UN wanted to group countries into three categories based on these two metrics(Birth rate and Life Expectancy) so that they can deliver proportionate
  aid packages with different birth rates and life expectancies to those countries.
- How algorithm was approached:
    (a) Initialised a center of the points for each cluster by randomly picking points from the dataset and using these as starting values for the means.
    (b) Assigned each point from the dataset to the nearest cluster.
    (c) Computed the means for each cluster as the mean for all the points that belong to it.
    (d) Repeated step 2 and 3 until convergence has been reached.

- References :
Aden Haussmann‘K-Means Clustering for Beginners’ (11-11-2020) Available at: https://towardsdatascience.com/k-means-clustering-for-beginners-ea2256154109 (Accessed: 09-10-2022).
"""
    
#Some hints on how to start, as well as guidance on things which may trip you up, have been added to this file.
#You will have to add more code that just the hints provided here for the full implementation.
#You will also have to import relevant libraries for graph plotting and maths functions.
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.spatial.distance import cdist
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn import metrics

# ====
# Define a function that computes the distance between two data points

def calc_distance(xy_list, centers):
    distance = cdist(xy_list, centers, 'euclidean')
    return distance


# ====
# Define a function that reads data in from the csv files  
# HINT: http://docs.python.org/2/library/csv.html. 
# HINT2: Remember that CSV files are comma separated, so you should use a "," as a delimiter. 
# HINT3: Ensure you are reading the csv file in the correct mode.

def read_csv(file):
    x_values = []
    y_values = []
    countries = []
    x_label = ""
    y_label = ""
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        lines = 0
        for row in reader:
            if lines >= 1:
                #print(', '.join(row))
                x_values.append(float(row[1]))
                y_values.append(float(row[2]))
                countries.append(row[0])
                lines += 1
            else:
                x_label = row[1]
                y_label = row[2]
                #print(', '.join(row))
                lines += 1
    return x_values, y_values, x_label, y_label, countries

# ====
# Define a function that finds the closest centroid to each point out of all the centroids
# HINT: This function should call the function you implemented that computes the distance between two data points.
# HINT: Numpy has a useful method that allows you to find the index of the smallest value in an array.

def find_clusters(xy_list, no_clusters, no_iterations, rseed=2):

    # Randomly choosing clusters
    randomCentre = np.random.RandomState(rseed)
    permutation = randomCentre.permutation(xy_list.shape[0])[:no_clusters]
    centers = xy_list[permutation]
    # Calculating the distances between points to centres
    labels = cdist(xy_list, centers, 'euclidean')
    min_index = np.array([np.argmin(i) for i in labels])
    return centers, min_index

def convergence(xy_list, no_clusters, no_iterations, min_index):
    
    # Looping based on the number of iterations that the user defined
    for i in range(no_iterations):
        centroids = []
        # Looping through the clusters created.
        for index in range(no_clusters):
            # Calculating the centroid mean 
            temp_cent = xy_list[min_index == index].mean(axis=0)
            centroids.append(temp_cent)
        centroids = np.vstack(centroids)
        # Calculatiing distance after new mean is found.
        distances = cdist(xy_list, centroids, 'euclidean')
        min_index = np.array([np.argmin(i) for i in distances])
    return min_index

#====
#Write a function to visualise the clusters. (optional, but useful to see the changes and if your algorithm is working)

def plot_clusters(xy_list,no_iterations, x_label, y_label):

    # Visualising the result of our K-Means algorithm before convergence
    plt.scatter(xy_list[:, 0], xy_list[:, 1], c = min_distance, s = 50, cmap = 'viridis')
    plt.scatter(centers[:,0], centers[:,1], color='r')
    plt.title('K-Means clustering of countries by birth rate vs life expectancy')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    for i in range(no_iterations):
        # Running k_means method to show convergence.
        label = convergence(xy_list, num_clusters, i, min_distance)
        plt.scatter(xy_list[:, 0], xy_list[:, 1], c = label, s = 50, cmap = 'viridis')
        #plt.scatter(centro[:,0], centro[:,1], color='r')
        plt.title('K-Means clustering of countries by birth rate vs life expectancy')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
          
#====
# Write the initialisation procedure
# Prompting user to input the file they want to use
file = input('''Plese enter the file name you want to use: 
                    data1953.csv
                    data2008.csv
                    dataBoth.csv : ''' )
# Reading from specified file and assigning to values
x_values, y_values, x_label, y_label, countries = read_csv(file)
   
#====
# Implement the k-means algorithm, using appropriate looping for the number of iterations
# --- find the closest centroid to each point and assign the point to that centroid's cluster
# --- calculate the new mean of all points in that cluster
# --- visualise (optional, but useful to see the changes)
#---- repeat

# Getting input from user to specify the number of clusters to make
num_clusters = 0
while True:
    try:
        num_clusters = int(input("Please enter the number of Clusters : "))
        break
    except ValueError:
        print("Please enter a valid number of clusters : ")
# Getting input from user to specify the number of iterations to loop        
no_iterations = 0
while True:
    try:
        no_iterations = int(input("Please enter the number of iterations : "))
        break
    except ValueError:
        print("Please enter a valid number of clusters : ")
pca = PCA(2) 
# Now that we have our data processed, we need to combine x and y into a 2D list of (x, y) pairs
xy_list = np.vstack((x_values, y_values)).T
# Calling a function to randomly create clusters
centers, min_distance = find_clusters(xy_list, num_clusters,no_iterations)
#Plotting the graph call
plot_clusters(xy_list,no_iterations, x_label, y_label)

# ====
# Print out the results for questions
#1.) The number of countries belonging to each cluster
#2.) The list of countries belonging to each cluster
#3.) The mean Life Expectancy and Birth Rate for each cluster

print("\nNumber of countries in each cluster:")
print(Counter(min_distance))
# Getting cluster indices
clusters_indices = defaultdict(list)
for index, counter_labels in enumerate(min_distance):
    clusters_indices[counter_labels].append(index)
# Printing countries in each cluster and means
clusters = 0
while clusters < num_clusters:
    print("\nCluster " + str(clusters + 1))
    print("----------")
    for indices in clusters_indices[clusters]:
        print(countries[indices])
    print("----------")
    print("Mean life expectancy:")
    print(centers[clusters][1])
    print("Mean birth rate:")
    print(centers[clusters][0])
    clusters += 1


