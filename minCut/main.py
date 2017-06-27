import random
import math
import copy
import time


def fileRead(afile):
    fileData = open(afile, "r")
    G = {}
    for line in fileData:
        ind = [int(i) for i in line.split()]
        G[ind[0]] = ind[1:]
    return G

def chooseEdge(G):
    u = random.choice(list(G.keys()))
    v = random.choice(list(G[u]))
    return u, v

def karger(G):
    # setting up empty list to retrieve length of the cut at the end of a loop
    length = []
    while len(G)>2:
        u,v = chooseEdge(G)

        # merge u and v into a single vertex u'
        G[u].extend(G[v])

        # Updating connectivity in the new graph
        for x in G[v]:
            G[x].remove(v)
            G[x].append(u)

        # Removing self-loops
        while u in G[u]:
            G[u].remove(u)

        # Removing v from the graph 
        del(G[v])

    # first key of G is unknown at the end so copy length into list
    for key in G.keys():
        length.append(len(G[key]))
    return length[0]

def main(afile,n):
    G = fileRead(afile)
    minCut = [float("inf")]
    for i in xrange(0,n):
        newG = copy.deepcopy(G)
        cut = karger(newG)
        if int(cut)<minCut[0]:
            minCut[0] = cut
    return minCut

if __name__ == '__main__':
    start = time.time()
    print(main("kargerMinCut.txt", 1000))
    print time.time() - start


