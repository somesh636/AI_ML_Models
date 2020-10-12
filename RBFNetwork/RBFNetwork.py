import numpy as np 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)


def kmeans(X, k):
    """Performs k-means clustering for 1D input
    
    Arguments:
        X {ndarray} -- A Mx1 array of inputs
        k {int} -- Number of clusters
    
    Returns:
        ndarray -- A kx1 array of final cluster centers
    """
 
    # randomly select initial clusters from input data
    clusters = np.random.choice(np.squeeze(X), size=k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False
 
    while not converged:
        """
        compute distances for each cluster center to each point 
        where (distances[i, j] represents the distance between the ith point and jth cluster)
        """
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
 
        # find the cluster that's closest to each point
        closestCluster = np.argmin(distances, axis=1)
 
        # update clusters by taking the mean of all of the points assigned to that cluster
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)
 
        # converge if clusters haven't moved
        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()
 
    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)
 
    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = X[closestCluster == i]
        if len(pointsForCluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(X[closestCluster == i])
 
    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))
 
    return clusters, stds


class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds
 
        self.w = np.random.randn(self.k)
        self.b = np.random.randn(1)

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return y_pred


    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kmeans(X, self.k)
        else:
            # use a fixed std 
            self.centers, _ = kmeans(X, self.k)
            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)
 
        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # print("i: ", i)
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                # print("a: ", a)
                F = a.T.dot(self.w) + self.b
                # print("w: ", self.w)
                # print("lenght of W: ", len(self.w))
                # print("b: ", self.b)
                # print("F: ", F)
                loss = (y[i] - F).flatten() ** 2
                # print("loss: ", loss)
 
                # backward pass
                error = -(y[i] - F).flatten()
 
                # online update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error

            print('Epoch: {0}/{1} | Loss: {2:.2f}'.format(epoch+1, self.epochs , loss[0]))
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    import itertools  
    import random
    import sys 
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score

    # generating point for x1
    X_1 = []
    X_2 = []
    X = []

    # random.seed(1)
    for value in range(21):
        x1_value = -2 + (0.2 * value)
        X_1.append(x1_value)
        X_2.append(x1_value)

    X = list(itertools.product(X_1, X_2))    
    # print("\n lenght of X: ", len(X))
    # print("\n X: ", X)

    X_X = []
    X_Y = []
    target = [] 
    x1_p1_X = []
    x1_p1_Y = []
    x2_m1_X = []
    x2_m1_Y = []
    for value in range(len(X)):
        tup = X[value]
        X_X.append(tup[0])
        X_Y.append(tup[1]) 

    # Target Value Generation
    for value in range(len(X)):
        x1 = X[value]
        target_value = np.square(x1[0]) + np.square(x1[1])
        if target_value <= 1:
            target.append(1)
            x1_p1_X.append(x1[0])
            x1_p1_Y.append(x1[1])
        elif target_value > 1: 
            target.append(-1)
            x2_m1_X.append(x1[0])
            x2_m1_Y.append(x1[1])

        target_value = 0
    
    X_new = []
    for value in range(len(X_X)):
        sum = X[value]
        sum_val = sum[0] + sum[1]
        X_new.append(sum_val)
        sum_val = 0
        
        
    X_new_values = np.array(X_new)
    y = np.array(target)

    # print("length of Target: ", len(target))
    # print("Targets: \n", target)
    
    # kmeans = KMeans(n_clusters= 150).fit(X)
    # print("Clusters Centers: \n", kmeans.cluster_centers_)
    # rbfnet = RBFNet(lr=1e-2,epochs = 200, k=150, inferStds=False)
    # rbfnet.fit(X_new_values, y)

    # y_pred = rbfnet.predict(X_new_values)
    # # print("y_pred.shape: ", y_pred.shape)
    # print("y_pred: ", y_pred)
    
    # accuracy_score_rbf = accuracy_score(y, y_pred)
    # print("Accuracy_Score: ",accuracy_score_rbf*100)

    print("inputs: \n", X)
    plt.plot(x1_p1_X, x1_p1_Y, 'x', color='r')
    plt.plot(x2_m1_X, x2_m1_Y, 'o', color='g')
    plt.xlabel("X_X")
    plt.ylabel("X_Y")
    plt.title("Original Datapoints")
    plt.show()
