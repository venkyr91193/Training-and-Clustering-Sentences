import os

from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import common_texts, get_tmpfile


class Train:
  def __init__(self, sentences:list, model_path:str):
    self.sentences = sentences
    self.model_path = model_path
    self.temp_path = get_tmpfile("custom_word2vec.model")
    self.train()

  def train(self):
    '''
    Function to train the word2vec model using gensim
    and give the word embeddings for each word.
    '''
    # initialize the model with few common words or token.
    # Can be preloaded to a much better pretrained word2vec model
    # But for custom specific application of documents or work,
    # loading your own corpus is better.
    model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)

    # start the training
    # can vary the epochs to be higher for more work specific tasks
    model.train(self.sentences, total_examples=len(self.sentences), epochs=1)

    # save the model
    model.wv.save(os.path.join(self.model_path, "custom_word2vec.model"))
