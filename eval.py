#! /usr/bin/env python

""" predict the polarity of sentences """

import logging
logging.basicConfig(filename="eval.log", level=logging.INFO, format='%(levelname)s: %(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger()
log.setLevel(logging.INFO)

import numpy as np
import tensorflow as tf

from train import Training

ALLOW_SOFT_PLACEMENT    = True
LOG_DEVICE_PLACEMENT    =  False


class Oracle:
    """
        Class to test the results
        Input: Text file containing one input sentence per line (assumes clean input)
        Output:

    """

    def __init__(self, resurrect_model_checkpoint):
        self.tr = Training()
        self.resurrect_model = resurrect_model_checkpoint

    def predict(self, test_file, cleaned=True):
        #TODO: preprocess the raw sentences (clean(line))
        if not cleaned:
            log.warn('currently only works for cleaned sentences...')
            return

        with open(test_file) as fl:
            log.debug('reading sentences...')
            #TODO: read file in batches
            sentences = np.array([x.lower().strip() for x in fl.readlines()])

        log.debug("transforming the sentences...")
        matrix = self.tr.fit_transform(sentences)

        log.debug('building the graph')
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=ALLOW_SOFT_PLACEMENT,
              log_device_placement=LOG_DEVICE_PLACEMENT)
            session_conf.gpu_options.allow_growth=True
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                log.debug('started the session...')
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(self.resurrect_model))
                saver.restore(sess, self.resurrect_model)

                log.debug('restored session, inputting x')
                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]

                log.debug('getting dropout probs...')
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                log.debug('extracting predictions graph component...')
                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/scores").outputs[0]
                bin_predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                log.debug('output class probabilities...')
                probas = sess.run(predictions, {input_x: matrix, dropout_keep_prob: 1.0})

                # log.debug('output just binary labels (1 or 0)...')
                # binary = sess.run(bin_predictions, {input_x: matrix, dropout_keep_prob: 1.0})

                del matrix, saver, predictions, bin_predictions, input_x
                sess.close()

            sess.close()

        del graph
        return probas


if __name__ == '__main__':

    # <Insert checkpoint model here...>
    # model_path = ''
    # orkl = Oracle(model_path)
    #
    # Input file containing sentences to be classified (one input sentence per line)
    # test_file = ''
    #
    # Predict class probabilities
    # probas = orkl.predict(test_file)
    # print (probas)

    pass
