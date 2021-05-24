from collections import defaultdict
import itertools
import numpy as np


# This class represents a directed graph
# using adjacency list representation
class Graph:

    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices

        # default dictionary to store graph
        self.graph = defaultdict(list)

        # function to add an edge to graph

    def addEdge(self, u, v):
        self.graph[u].append(v)

    '''A recursive function to print all paths from 'u' to 'd'.
    visited[] keeps track of vertices in current path.
    path[] stores actual vertices and path_index is current
    index in path[]'''

    def printAllPathsUtil(self, u, d, visited, path, a):

        # Mark the current node as visited and store in path
        visited[u] = True
        path.append(u)

        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            print(path)
            a.append(path.copy())
            print(a)
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.graph[u]:
                if visited[i] == False:
                    self.printAllPathsUtil(i, d, visited, path, a)


        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u] = False


    # Prints all paths from 's' to 'd'
    def printAllPaths(self, s, d):

        # Mark all the vertices as not visited
        visited = [False] * (self.V)

        # Create an array to store paths
        path = []
        a = []

        # Call the recursive helper function to print all paths
        self.printAllPathsUtil(s, d, visited, path, a)
        return a


# Create a graph given in the above diagram
g = Graph(4)
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(0, 3)
g.addEdge(2, 0)
g.addEdge(2, 1)
g.addEdge(1, 3)

s = 2
d = 3
print(g)
print("Following are all different paths from % d to % d :" %(s, d))
a = g.printAllPaths(s, d)
print(a)
b = [[0 for _ in range(3)] for _ in range(4)]
print(b, len(b))


f = [i for i in range(2)] * 3
print(f)
c = list(itertools.permutations(f, 3))
d = list(itertools.combinations([0, 0, 0, 1, 1, 1], 3))
e = sorted(set(c), key=c.index)
print("c:", c,
      "\nd:", d,
      "\ne:", e)

print(e[1][1])
f = [i for i in range(2)] * 3
print(f)
print(1512 * 8 * 30 //  (10 * 1000))
index = np.arange(0, 3)
print(index[(1,2,3) != (1,2,4)])

def get_index(lst, item):
    return [index for (index, value) in enumerate(lst) if value == item]

print(get_index((1, 0, 0), 0))
