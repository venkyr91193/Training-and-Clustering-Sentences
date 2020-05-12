import setuptools
from distutils.core import setup

setup(name='text_clustering',
      author='Venkataramana Radhakrishnan',
      author_email='venkyr91193@gmail.com',
      description='To train your custom word2vec for work specific tasks in many languages',   
      version='0.0.1',
      packages = ['train_and_cluster'],
      py_modules=['cluster','languages','load_train_cluster','train'],
      install_requires=['spacy','sklearn','nltk','gensim','numpy'],
      )