import argparse

from train_and_cluster.languages import languages
from train_and_cluster.load_train_cluster import LoadTrainCluster


class TextClustering:
  def __init__(self, sentences:list, model_path:str, language:str, num_of_clusters:int, type_of_clustering:str):
    self.sentences = sentences
    self.model_path = model_path
    if language not in languages.keys():
      print('Invalid language')
      exit()
    else:
      self.language = language
    self.num_of_clusters = num_of_clusters
    self.type_of_clustering = type_of_clustering
    self.ltc_obj = LoadTrainCluster(self.sentences, self.language, self.model_path)
    self.clusters = self.ltc_obj.train_cluster(self.type_of_clustering,self.num_of_clusters)
