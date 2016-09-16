Forked from [here](https://github.com/dennybritz/cnn-text-classification-tf), based on the paper [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) by Yoon Kim (2014).

## Requirements

- Python 3
- Numpy
- Pandas
- Gensim
- TensorFlow > 0.8

This modified version has a few *important* changes:

* **Pre-trained Word Embeddings**
  * Any word embeddings supported by gensim **AND** a separate vocabulary file (with frequency counts).
      - E.g, While saving *Word2Vec* model after training, do:

```
model.train(...)
model.init_sims(replace=True)    # if no plans to train further
model.save_word2vec_format('w2v_model.bin', binary=True, fvocab='vocab.txt')
```

* The **vocab** file is important as it is used to build vocabulary for the model.
    * It typically is a file consisting of words for which embeddings are defined along with their frequency count (how many times the model has seen that word while training) in decreasing order. It looks like the following:

```
This  5000
is    4500
cool  500
!     25
😍    20
...
```
&ensp; &ensp; &ensp; &ensp; &ensp; Tensorflow's built-in *`VocabularyProcessor`* can also be used but it won't detect emojis 😭😭 which I felt were important for Sentiment Analysis. If you don't care about them, you can opt for the latter method (*not implemented*). *Also note* that inorder to exploit the former technique, embeddings for the emojis should also be given by *Word2Vec / Glove etc.* models.

* **Batch Processing of input data**
  * Data is stored in HDFS file system. So easy and efficient to retrieve in batches.

* **Added *Softmax* to output class probabilities while prediction**


# ToDo:

* [Improve](data_helpers.py) input handling.
  * Handles (reasonably) large input size (tested till `2 Million` examples) well but first loads them into memory to transform and store them (numerically) in HDFS. Better not to do that.
* Testing the trained Tensorflow model assumes *cleaned (preprocessed)* input for testing.
  * Do the cleaning before feeding it to the model for classification


## Training

* Put files containing positive examples *`pos.txt`* & negative examples separately *`neg.txt`* in `data` folder.
* Put word embeddings & the vocab file in `model` folder.
* **Adjust the model hyperparams first: [here](https://github.com/vaddina/cnn-text-classification-tf/blob/experiments/train.py#L47-L62)**

then run:
```
>>> from train import Training
>>> tr = Training()
>>> tr.train()
```

Checkpoints will be stored in `tensorboard_runs` folder.

## Prediction

* Keep sentences (one per line) in a text file for input
* Feed it and get class probabilities for each sentence


```
# <Insert checkpoint model here...>
model_path = ''
orkl = Oracle(model_path)

# Input file containing sentences to be classified (one input sentence per line)
test_file = ''

# Predict class probabilities
probabilities = orkl.predict(test_file)
print (probabilities)
```

## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
