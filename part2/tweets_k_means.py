import numpy as np 
import pandas as pd 
import nltk
import json
import math
import sys
import codecs

from nltk.tokenize import RegexpTokenizer
from copy import deepcopy

tokenizer = RegexpTokenizer(r'\w+')

class tweets_kMeans:
    
    def __init__(self, k, tweets_dict, iniCentroids): 
        self.k = k
        self.tweets_dict = tweets_dict
        self.iniCentroids = iniCentroids

    def jaccardDis(self, tweet1, tweet2):
        t1 = tokenizer.tokenize(tweet1.lower())
        t2 = tokenizer.tokenize(tweet2.lower())
        t1Ut2 = len(set(t1).union(set(t2)))
        t1It2 = len(set(t1).intersection(set(t2)))
        return 1 - (t1It2 / t1Ut2)
    
    def find_centroid(self, clst):
        min_distance = sys.maxsize
        new_centroid = clst[0]
        
        for tweet_id1 in clst: 
            dist = 0
            for tweet_id2 in clst:
                dist += self.jaccardDis(self.tweets_dict.get(tweet_id1), self.tweets_dict.get(tweet_id2))
            
            if dist < min_distance:
                min_distance = dist
                new_centroid = tweet_id1
    
        return new_centroid
            
    def kmeans(self):
        centroids = []
        clusters = {}

        for i in range(self.k):
            centroids.append(self.iniCentroids[i])
            
        while True:
            for i in range(self.k):
                cluster_list = []
                clusters[i] = cluster_list
            
            for tweet_id, tweet in self.tweets_dict.items():
                min_distance = sys.maxsize
                index = 0
                for i in range(self.k):
                    dist = self.jaccardDis(tweet, self.tweets_dict.get(centroids[i]))
                    if dist < min_distance:
                        min_distance = dist
                        index = i
                
                clusters.get(index).append(tweet_id) 

            new_centroids = deepcopy(centroids) #centroids.copy()
            for i, cluster in clusters.items():
                new_centroid = self.find_centroid(cluster)
                new_centroids[i] = new_centroid
                
            convergence = True
            for i in range(self.k):
                if new_centroids[i] != centroids[i]:
                    convergence = False
                    break
            
            centroids = deepcopy(new_centroids)
            if convergence == True:
                break
               
        return centroids, clusters
    
    def SSE(self, centroids, clusters):
        SSE = 0
        for i in range(self.k):
            centroid = centroids[i]
            
            for tweet_id in clusters[i]:
                SSE += math.pow(self.jaccardDis(self.tweets_dict.get(centroid), self.tweets_dict.get(tweet_id)),2)     
        return SSE


k = int(sys.argv[1])
initialSeeds = sys.argv[2]
tweetDataset = sys.argv[3]
outputFile = sys.argv[4]  

t_dict = {}
tweets = []

with codecs.open(tweetDataset, encoding='utf8') as t:
    for line in t.readlines():
        tweets.append(json.loads(line))

for tweet in tweets:
    tweet['text'] = tweet['text'].lower()
    t_dict[tweet['id']] = tweet['text']

centroids = []
with open(initialSeeds, encoding="utf8") as file:
    initialCentroid = file.read().replace('\n', '')

initialCentroid = initialCentroid.split(',')
centroids = [int(x) for x in initialCentroid]

tweet_cluster = tweets_kMeans(k, t_dict, centroids)
new_centroids, clusters = tweet_cluster.kmeans()
SSE = tweet_cluster.SSE(new_centroids, clusters)

file = open(outputFile,'w') 
for i in range(k):
    cluster_list = clusters.get(i)
    cluster_str = ', '.join(str(e) for e in cluster_list)
    file.write(str(i) + '\t' + cluster_str + '\n\n') 
file.write('SSE is ' + str(SSE) + '\n')
file.close() 