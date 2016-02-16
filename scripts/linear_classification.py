"""
:description: Classifying locations using linear classifiers and extracted features.
"""
import argparse
from sklearn import linear_model

import data_utils

def run_classifier(feature_set, loss):
    """
    :description: Fit a classifier and check its validation accuracy.

    :type feature_set: string
    :param feature_set: which feature set to use, options are {'all', 'hist', 'gist'}

    :type loss: string
    :param loss: which loss function to use
    """
    train, y_train, val, y_val, test, y_test = data_utils.load_features_and_labels(feature_set)
    classifier = linear_model.SGDClassifier(loss, n_iter=50, alpha=1e-4,  verbose=1)
    classifier.fit(train, y_train)
    acc = classifier.score(val, y_val)
    print 'validation accuracy: {}'.format(acc)
    
def parse_args():
    """
    :description: Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='feature_set', default='all',
            help="""{'all', 'hist', 'gist'}""")
    parser.add_argument('-l', dest='loss', default='hinge',
            help="""{'hinge', 'log', 'modified_huber', 'perceptron'}""")
    args = parser.parse_args()
    return args.feature_set, args.loss

if __name__ == '__main__':
    feature_set, loss = parse_args()
    run_classifier(feature_set, loss)
