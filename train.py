#! /usr/bin/env python

import logging
logging.basicConfig(filename="cnn.log", level=logging.INFO, format='%(levelname)s: %(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger()
log.setLevel(logging.INFO)

log.debug('loading all modules...')

import sys
from os import path
path_to_set = path.dirname( path.dirname( path.dirname( path.abspath(__file__) ) ) )
sys.path.insert(0, path_to_set)

import tensorflow as tf
import pandas as pd
pd.set_option('io.hdf.default_format','table')
import numpy as np
import os
import time
import datetime
import subprocess
from tensorflow.contrib import learn
from gensim.models import Word2Vec

from data_helpers import dathelp
dhp = dathelp()

from text_cnn import TextCNN

log.debug('all modules loaded...')

# PARAMETERS
# ==================================================

W2V_MODEL_PATH      = './model/model.bin'
W2V_VOCAB_PATH      = './model/vocab.txt'
HDFStore_PATH       = './data/h5/trans_mat.h5'
TENSORBOARD_PATH    = './tensorboard_runs/'

if not os.path.exists('./data/h5'):
    os.mkdir('./data/h5')
if not os.path.exists(TENSORBOARD_PATH):
    os.mkdir(TENSORBOARD_PATH)

EMBEDDING_DIM       = 300
FILTER_SIZES        = [2, 3, 4]
NUM_FILTERS         = 3
DEV_SPLIT_SIZE      = 0.2
DEV_BATCH_SIZE      = 400
DROPOUT_KEEP_PROB   = 0.7
L2_REG_LAMBDA       = 0
LEARNING_RATE       = 1e-3

BATCH_SIZE          = 800
NUM_EPOCHS          = 2
EVALUATE_EVERY      = 5
CHECKPOINT_EVERY    = 10

ALLOW_SOFT_PLACEMENT    = True
LOG_DEVICE_PLACEMENT    =  False

POS_FILE = './data/cnn/pos.txt'
NEG_FILE = './data/cnn/neg.txt'


def get_line(filename):
    """ generator to read lines from a file """
    for line in open(filename):
        yield line.replace('\\n', ' ')

def get_ixs(arr):
    """
        iterates through a list in a pair - only once.
        e.g., [2, 4, 5, 6] gives (2,4), (4,5) & (5,6)
    """
    for i in range(len(arr)-1):
        yield (arr[i], arr[i+1])


class Training:

    def __init__(self):
        self.store = pd.HDFStore(HDFStore_PATH)
        self.store.close()
        self.posneg_changed = self.max_document_length = None

        self.build_vocab()

    def load_w2v_model(self):
        self.model = Word2Vec.load_word2vec_format(W2V_MODEL_PATH, binary=True)
        log.debug(" done loading w2v model ")

    def get_max_document_length(self):
        """
            finds the sentence with the maximum length
            'wc' command wont detect emojis separated by space as words... I need that.. so this function...
        """

        self.store.open() if not self.store.is_open else None
        if '/sentences' in self.store.keys() and not self.posneg_changed:
            shape = self.store.sentences.shape[1]
            self.store.close()
            return shape
        else:
            log.debug('calculating max_document_length...')
            self.store.close()
            total_lines = 0
            for filename in [POS_FILE, NEG_FILE]:
                tot_lines = int(subprocess.Popen('wc -l < {}'.format(filename), shell=True,
                            stdout=subprocess.PIPE).communicate()[0].strip())
                total_lines += tot_lines

            len_sent = np.zeros(total_lines)

            i = 0
            for filename in [POS_FILE, NEG_FILE]:
                r = get_line(filename)
                try:
                    while True:
                        len_sent[i] = len(next(r).strip().split())
                        i += 1
                except StopIteration:
                    # log.debug('end of file {}'.format(filename))
                    pass

            return np.max(len_sent)

    def add_w2v_vocab(self, voc_pre):
        w2v_vocab = open(W2V_VOCAB_PATH).readlines()

        for line in w2v_vocab:
            word, count = line.strip().split()
            voc_pre.vocabulary_.add(category=word, count=int(count))

        # Freeze the vocab
        voc_pre.vocabulary_.freeze(freeze=True)

        return voc_pre

    def build_vocab(self):
        log.debug("building vocabulary")
        self.max_document_length = self.get_max_document_length()
        log.debug('got {} max length'.format(self.max_document_length))
        self.vocab_processor = self.add_w2v_vocab(learn.preprocessing.VocabularyProcessor(self.max_document_length))

    def fit_transform(self, sentences):
        log.debug('fit transforming')
        mat = np.zeros((len(sentences), self.max_document_length), dtype=int)
        for i in range(len(sentences)):
            vals = [self.vocab_processor.vocabulary_.get(x) for x in sentences[i].split()]
            mat[i][:len(vals)] = vals
        return mat

    def train(self, posneg_changed=False, embeddings_changed=False):
        """
            Trains a CNN model
            set 'posneg_changed' to True if the input files are changed. (has to be 'True' when run for the first time)
            set 'embeddings_changed' to True when word embedding model & vocab files are changed
        """

        self.store.open() if not self.store.is_open else None
        data_exists = '/sentences' in self.store.keys()
        self.store.close()

        if posneg_changed or not data_exists:
            log.debug('posneg_changed...')
            self.posneg_changed = True
            self.max_document_length = None
            self.build_vocab()
            log.debug('posneg files changed... (re)writing the transformation matrix...')
            dhp.write_transformation_matrix(POS_FILE, NEG_FILE)


        if embeddings_changed:
            log.debug('embeddings_changed.... building vocab')
            self.build_vocab()


        log.info('loading w2v model')
        self.load_w2v_model()

        global EMBEDDING_DIM
        EMBEDDING_DIM = self.model.syn0.shape[1]

        log.info("Training...")
        # ==================================================

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=ALLOW_SOFT_PLACEMENT,
              log_device_placement=LOG_DEVICE_PLACEMENT)

            try:

                sess = tf.Session(config=session_conf)
                with sess.as_default():
                    log.debug("building the layers...")
                    cnn = TextCNN(
                        sequence_length=self.max_document_length,
                        num_classes=2,
                        vocab_size=len(self.vocab_processor.vocabulary_),
                        embedding_size=EMBEDDING_DIM,
                        filter_sizes=FILTER_SIZES,
                        num_filters=NUM_FILTERS,
                        l2_reg_lambda=L2_REG_LAMBDA)

                    # Define Training procedure
                    global_step = tf.Variable(0, name="global_step", trainable=False)
                    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
                    grads_and_vars = optimizer.compute_gradients(cnn.loss)
                    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                    # Keep track of gradient values and sparsity (optional)
                    grad_summaries = []
                    for g, v in grads_and_vars:
                        if g is not None:
                            grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                            sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                            grad_summaries.append(grad_hist_summary)
                            grad_summaries.append(sparsity_summary)
                    grad_summaries_merged = tf.merge_summary(grad_summaries)

                    # Output directory for models and summaries
                    timestamp = str(int(time.time()))
                    filter_str = str(FILTER_SIZES)
                    concat_filters_str = ''.join([x.strip() for x in  filter_str.strip('[]').split(',')])
                    model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format('w2v', concat_filters_str,
                    NUM_FILTERS, int(10 * DEV_SPLIT_SIZE), int(10 * DROPOUT_KEEP_PROB), L2_REG_LAMBDA, BATCH_SIZE, int(np.log10(1/LEARNING_RATE)), timestamp)
                    out_dir = os.path.abspath(os.path.join(TENSORBOARD_PATH, "runs", model_name))
                    log.debug("Writing to {}\n".format(out_dir))

                    # Summaries for loss and accuracy
                    loss_summary = tf.scalar_summary("loss", cnn.loss)
                    acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

                    # Train Summaries
                    train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
                    train_summary_dir = os.path.join(out_dir, "summaries", "train")
                    train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

                    # Dev summaries
                    dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
                    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                    dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

                    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)

                    log.debug("saving with Saver...")
                    saver = tf.train.Saver(tf.all_variables())

                    log.debug("saving vocabulary... ")
                    self.vocab_processor.save(os.path.join(out_dir, "vocab"))

                    log.debug("Initializing all variables... ")
                    sess.run(tf.initialize_all_variables())

                    try:
                        self.store.open() if not self.store.is_open else None
                        if not embeddings_changed and '/initW' in self.store.keys() and not self.posneg_changed:
                            initW = self.store['initW'].as_matrix()
                            log.debug('succesfully loaded weight matrix')
                        else:
                            self.build_vocab()
                            initW = np.zeros((len(self.vocab_processor.vocabulary_), EMBEDDING_DIM))
                            initW[0] = np.random.uniform(-0.25, 0.25, EMBEDDING_DIM)
                            len_initW = len(initW)

                            log.debug("updating weight matrix with word2vec vectors ")
                            for i in range(1, len_initW):
                                word = self.vocab_processor.vocabulary_.reverse(i)
                                if i%10000 == 0:
                                    log.debug('loading embeddings : {} %'.format(round(100. * i/len_initW, 2)))

                                initW[i] = self.model[word]

                            self.store.open() if not self.store.is_open else None
                            self.store['initW'] = pd.DataFrame(initW, columns=np.array2string(np.arange(initW.shape[1]))[1:-1].split())

                            log.debug('succesfully stored weight matrix')

                    except Exception as e:
                        log.exception(e)
                    finally:
                        self.store.close()

                    log.debug("Assigning the final weight matrix")
                    sess.run(cnn.W.assign(initW))


                    def _train_step(x_batch, y_batch):
                        """
                        A single training step
                        """
                        feed_dict = {
                          cnn.input_x: x_batch,
                          cnn.input_y: y_batch,
                          cnn.dropout_keep_prob: DROPOUT_KEEP_PROB
                        }
                        _, step, summaries, loss, accuracy, ww = sess.run(
                            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.W],
                            feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        log.debug("TRAIN: {}: step {} ({} %), loss {:g}, acc {:g}".format(time_str, step,
                                                            round(100 * step/self.TOTAL_STEPS, 2), loss, accuracy))
                        train_summary_writer.add_summary(summaries, step)


                    def _dev_step(x_batch, y_batch, writer=None):
                        """ Evaluates model on a dev set """

                        feed_dict = {
                          cnn.input_x: x_batch,
                          cnn.input_y: y_batch,
                          cnn.dropout_keep_prob: 1.0
                        }
                        step, summaries, loss, accuracy = sess.run(
                            [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                            feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        log.debug("DEV: {}: step {} ({} %), loss {:g}, acc {:g}".format(time_str, step,
                                                            round(100 * step/self.TOTAL_STEPS, 2), loss, accuracy))
                        if writer:
                            writer.add_summary(summaries, step)

                        return accuracy, loss

                    self.store.open() if not self.store.is_open else None
                    log.debug('getting development dataset...')
                    # self.get_dev_data()
                    self.seek_index = int(self.store.labels.shape[0] * (1 - DEV_SPLIT_SIZE))
                    dev_batches = np.arange(self.seek_index, self.store.labels.shape[0], DEV_BATCH_SIZE).tolist()
                    dev_batches.append(self.store.labels.shape[0])

                    batches = np.arange(0, self.seek_index, BATCH_SIZE).tolist()
                    batches.append(self.seek_index)

                    self.TOTAL_STEPS = int(np.ceil(self.seek_index / BATCH_SIZE) * NUM_EPOCHS)

                    try:
                        log.debug('starting training in batches...')
                        for epoch in range(NUM_EPOCHS):
                            log.debug('epoch : {}'.format(epoch))
                            # Training loop. For each batch...
                            for i in range(len(batches)-1):
                                start, stop = batches[i], batches[i+1]
                                x_batch = self.store.select('sentences', 'index>={} & index<{}'.format(start, stop))
                                y_batch = self.store.select('labels',    'index>={} & index<{}'.format(start, stop))

                                log.debug(" TRAIN BATCH - ({}, {})".format(start, stop))

                                _train_step(x_batch, y_batch)
                                current_step = tf.train.global_step(sess, global_step)

                                if current_step % EVALUATE_EVERY == 0:
                                    log.debug("\n ---- Evaluation ----- ")

                                    accu_losses = []
                                    for indices_pair in get_ixs(dev_batches):
                                        start, stop = indices_pair
                                        x_dev = self.store.select('sentences', 'index>={} & index<{}'.format(start, stop))
                                        y_dev = self.store.select('labels',    'index>={} & index<{}'.format(start, stop))

                                        log.debug(" DEV BATCH - ({}, {})".format(start, stop))
                                        accuracy, loss = _dev_step(x_dev, y_dev, writer=dev_summary_writer)
                                        accu_losses.append([accuracy, loss])

                                    mean_acc_loss = np.array(accu_losses).mean(axis=0)
                                    log.debug('\n\n DEV BATCH: Average loss: {}, accuracy: {} \n\n'.format(mean_acc_loss[1], mean_acc_loss[0]))

                                if current_step % CHECKPOINT_EVERY == 0:
                                    log.debug("\n ---- checkpoint ----- ")
                                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                                    log.debug("Saved model checkpoint to {}\n".format(path))

                    except Exception as e:
                        log.exception(e)
                    finally:
                        self.store.close()

            except Exception as e:
                log.exception(e)
            finally:
                sess.close()


if __name__ == '__main__':
    tr = Training()
    tr.train(posneg_changed=False, embeddings_changed=False)
