import random

import spacy

from train_and_cluster.cluster import Cluster
from train_and_cluster.train import Train


class LoadTrainCluster:
  def __init__(self, sentences:list, language:str, model_path:str):
    self.sentences = sentences
    self.model_path = model_path
    self.tokenized_sentences = list()
    self.obj = spacy.blank(language)
    self.obj.add_pipe(self.obj.create_pipe('sentencizer'))
    self.obj.max_length = 10000
    self.get_sentence_tokens()

  def get_sentence_tokens(self):
    '''
    Function to get the tokens from the sentence
    and filter out the stop words.
    '''
    # using the blank english model for tokenising and removing keywords.
    for data in self.sentences:
      doc = self.obj(data)    
      doc_array = [str(t) for t in doc if not t.is_stop]
      self.tokenized_sentences.append(doc_array)

  def train_cluster(self, type_of_clustering:str = 'nltk', num_of_clusters:int = 10):
    '''
    Start the training of the model using Train api using 80% of input.
    Cluster the remaining 20% to check the score and the clustering.
    '''
    random.shuffle(self.tokenized_sentences)

    train_obj = Train(self.tokenized_sentences, self.model_path)
    # defaulted the number of clusters to 10.
    cluster_obj = Cluster(self.tokenized_sentences, self.model_path, num_of_clusters)
    if type_of_clustering == 'nltk':
      return cluster_obj.nltk_cluster()
    elif type_of_clustering == 'sklearn':
      return cluster_obj.nltk_cluster()
