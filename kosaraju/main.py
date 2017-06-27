#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import random
import math
import copy
import time
from itertools import groupby
from collections import defaultdict

class Track(object):
    def __init__(self):
        self.time = 0
        self.done = {}
        self.leader = {}
        self.current_source = None
        self.visited = set()

def fileRead(afile):
    fileData = open(afile, "r")
    G = {}
    for line in fileData:
        s,t = line.split()
        s,t = int(s), int(t)
        dest_nodes = G.get(s, set())
        dest_nodes.add(t)
        G[s] = dest_nodes
    return G

def reverse_graph(G):
    edges = set()
    reverse_G = {}
    for node, dest_nodes in G.items():
        for dest_node in dest_nodes:
            edges.add((node, dest_node))
    for edge in edges:
        dest_nodes = reverse_G.get(edge[1],set())
        dest_nodes.add(edge[0])
        reverse_G[edge[1]] = dest_nodes
    return reverse_G

def dfs(G,node,track):
    track.visited.add(node)
    track.leader[node] = track.current_source
    if node in G.keys():
        for j in G[node]:
            if j not in track.visited:
                dfs(G, j, track)
    track.time = track.time + 1
    track.done[node] = track.time

def kosaraju(G):
    n = len(G)
    track = Track()
    reverse_G = reverse_graph(G)
    nodes = reverse_G.keys()
    out = {}

    # run first dfs loop on reversed graph 
    for i in reversed(xrange(1, n+1)):
        if i not in track.visited:
            track.current_source = i
            dfs(reverse_G, i, track)
    
    # sort nodes by finishing times in decreasing order
    sorted_nodes = sorted(track.done, key=track.done.get, reverse = True)
    
    # reset tracker and timer
    track.done = {}
    track.current_source = None
    track.visited = set()
    

    #run second loop on original graph with sorted nodes
    for i in sorted_nodes:
        if i not in track.visited:
            track.current_source = i
            dfs(G, i, track)

    #statistics
    lengthSCC = []
    for lead, vertex in groupby(sorted(track.leader, key=track.leader.get), key=track.leader.get):
        out[lead] = list(vertex)
        lengthSCC.append(len(out[lead]))
    lengthSCC.sort(reverse = True)
    print(lengthSCC[:5])





def main():
    G = fileRead("SCC.txt")
    kosaraju(G)
    

if __name__ == '__main__':
    start = time.time()
    main()
    print time.time() - start
