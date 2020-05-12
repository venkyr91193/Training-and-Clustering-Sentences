# Training_and_Clustering_Sentences

This API can be used to input a list of sentences in the supported languages as below and cluster the sentence accordingly.

# Supported languages

German, Greek, English, Spanish, french, Italian, Dutch and Portuguese.

# Clustering

Currently supports only NLTK and Sklearn

# To install and run

Clone the repository and navigate to src folder.

# Build and Install

    >>> python setup.py bdist_wheel
    >>> pip install .\dist\custom_train_and_clustering-0.0.1-py3-none-any.whl

# Example to run

    >>> from train_and_cluster import SentenceClustering
    >>> a = SentenceClustering(sentences_in_a_python_list,model_path,'English',num_of_clusters,'nltk')
    >>> print(a.clusters)
    >>> You will have a list of integers with the clusters for the sentence in the same order as the input.
