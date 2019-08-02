import os

import nltk
import numpy as np
from gensim.models import KeyedVectors
from nltk.cluster import KMeansClusterer
from sklearn import cluster, metrics


class Cluster:
  def __init__(self, sentences:list , model_path:str, num_cluster:int):
    self.NUM_CLUSTERS = num_cluster
    self.sentences = sentences
    self.model_path = model_path
    self.model = None
    self.vectorized_sentences = list()
    self.load_the_model()
    self.vectorize_sentences()

  def sent_vectorizer(self, sent):
    '''
    Function to vectorize a sentence on all the word embeddings
    '''
    sent_vec =[]
    numw = 0
    for w in sent:
      try:
        if numw == 0:
          sent_vec = self.model[w]
        else:
          sent_vec = np.add(sent_vec, self.model[w])
        numw+=1
      except:
          pass
    return np.asarray(sent_vec) / numw

  def vectorize_sentences(self):
    '''
    Function to vectorize the sentences
    '''
    for sentence in self.sentences:
      self.vectorized_sentences.append(self.sent_vectorizer(sentence))

  def load_the_model(self):
    # loading the model
    # The reason for separating the trained vectors into KeyedVectors is because it can be 
    # mapped for lightning fast loading and sharing the vectors in RAM between processes
    self.model = KeyedVectors.load(os.path.join(self.model_path, "custom_word2vec.model"), mmap='r')
  
  def sklearn_cluster(self):
    '''
    Using sklearn clustering
    '''
    kmeans = cluster.KMeans(n_clusters=self.NUM_CLUSTERS)
    kmeans.fit(self.vectorized_sentences)
      
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_   
    print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
    print (kmeans.score(self.vectorized_sentences))
    silhouette_score = metrics.silhouette_score(self.vectorized_sentences, labels, metric='euclidean')     
    print ("Silhouette_score: ")
    print (silhouette_score)
    return labels

  def nltk_cluster(self):
    '''
    Using clustering from nltk
    '''
    kclusterer = KMeansClusterer(self.NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, avoid_empty_clusters=True, repeats=1)
    assigned_clusters = kclusterer.cluster(self.vectorized_sentences, assign_clusters=True)
    return assigned_clusters
