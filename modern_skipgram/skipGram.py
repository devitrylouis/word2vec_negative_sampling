from __future__ import division
import argparse
import pandas as pd

# useful stuff
from code.skipgram import SkipGram
from code.utils import text2sentences, loadPairs

authors = ['Louis De Vitry','Alami Chehboune Mohamed']
emails  = ['louis.devitry@student-cs.fr','m.alamichehboune@student-cs']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = mSkipGram.load(opts.model)
        for a,b,_ in pairs:
            print sg.similarity(a, b)
