import sys, os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__))))
from train_and_cluster import TextClustering

def test_whole_pipe():
  sentences = ['Today is a good day.','I see an aeroplane in the sky.','Alex and Mic are best friends.','']
  clusters = TextClustering(sentences,os.path.join(os.path.dirname(os.path.dirname(__file__))),'en',2,'nltk')
  clusters = TextClustering(sentences,os.path.join(os.path.dirname(os.path.dirname(__file__))),'en',2,'sklearn')
  pass

if __name__ == "__main__":
  test_whole_pipe()
  print('All tests passed')
