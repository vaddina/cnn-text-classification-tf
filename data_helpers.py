import logging
logging.basicConfig(filename="dhp.log", level=logging.INFO, format='%(levelname)s: %(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger()
log.setLevel(logging.INFO)

import sys
from os import path
path_to_set = path.dirname( path.dirname( path.dirname( path.abspath(__file__) ) ) )
sys.path.insert(0, path_to_set)

import pandas as pd
pd.set_option('io.hdf.default_format','table')

import numpy as np

class dathelp:
    def __init__(self):
        pass

    def write_transformation_matrix(self, pos_file, neg_file, hdf_store='./data/h5/trans_mat.h5', batch_size=5000, append=False):
        """
            Loads the ±ve sentences into an array, shuffles, "fit transforms" and stores them in a HDFS file
        """

        log.debug('calling training back from datahelpers inside function')
        from train import Training
        tr = Training()

        # Not a good idea if the data is immense... Works fine for couple of hundered thousand (1e5) lines per file...
        log.debug("loading positive file...")
        with open(pos_file) as fl:
            pos = np.array([x.strip() for x in fl.readlines()])

        log.debug("loading negative file...")
        with open(neg_file) as fl:
            neg = np.array([x.strip() for x in fl.readlines()])

        log.debug("creating labels...")
        lpos = np.tile([0, 1], (len(pos), 1))
        lneg = np.tile([1, 0], (len(neg), 1))

        sentences = np.concatenate((pos, neg))
        labels = np.concatenate((lpos, lneg))

        del pos, neg, lpos, lneg

        log.debug("generating shuffled_data, setting seed for reproducibility...")
        np.random.seed(7)
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        labels = labels[shuffle_indices]

        log.debug("storing shuffled labels in HDFStore 'labels' table ")
        store = pd.HDFStore(hdf_store)
        if append:
            store.append('labels', pd.DataFrame(labels, columns=np.array2string(np.arange(labels.shape[1]))[1:-1].split()), format='table')
        else:
            store['labels'] = pd.DataFrame(labels, columns=np.array2string(np.arange(labels.shape[1]))[1:-1].split())
        store.close()

        log.debug("shuffling sentences")
        sentences = sentences[shuffle_indices]
        batch_ixs = np.arange(0, len(sentences), batch_size).tolist()
        batch_ixs.append(len(sentences))
        tot_batches = len(batch_ixs)

        store.open()
        if not append:
            if '/sentences' in store.keys():
                log.warn('"sentences" table already exists... deleting & re-creating again.')
                del store['sentences']

        tr.build_vocab()
        log.debug("loading sentence matrices into HDFStore")
        for i in range(tot_batches-1):
            log.debug("batch {} of {} - {} %".format(i, tot_batches-1, round(100. * i/(tot_batches-1), 2) ))

            log.debug('fit_transforming from dhp')
            num_mat = tr.fit_transform(sentences[ batch_ixs[i] : batch_ixs[i+1] ])
            store.append('sentences', pd.DataFrame(num_mat, index=np.arange( batch_ixs[i], batch_ixs[i+1] ),
                        columns=np.array2string(np.arange(num_mat.shape[1]))[1:-1].split()), format='table')

        log.debug('stored sentences with dims: {}'.format(store.sentences.shape))
        store.close()
        log.debug("done...")


if __name__ == '__main__':
    write_transformation_matrix('./data/cnn/pos.txt', './data/cnn/neg.txt', batch_size=2000)
