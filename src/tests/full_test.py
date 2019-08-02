import sys, os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__))))
from train_and_cluster import SentenceClustering

def test_whole_pipe():
  sentences = ['Today is a good day.','I see an aeroplane in the sky.','Alex and Mic are best friends.']
  clusters = SentenceClustering(sentences,r'D:\Personal\Repos\Code','English',2,'nltk')
  print()

if __name__ == "__main__":
  test_whole_pipe()