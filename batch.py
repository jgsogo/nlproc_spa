#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import argparse
from nlproc.pos_tagger import postagger_match
from dataset.base_dataset import Dataset
from nlproc.universal_pos import UniversalPOSTagger, UniversalPOSDataset

import logging
log = logging.getLogger(__name__)

_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']


def evaluate(tagger, dataset):
    sents = dataset.tagged_sents

    for sent in sents:
        test = Dataset.untag(sent)
        ret = tagger.tag(test)

        for input, output in zip(sent, ret):
            assert input[0] == output[0], "Word has to be the same"
            print("{} <> {}".format(input, output))
    return 0


def _log_level_string_to_int(log_level_string):
    if not log_level_string in _LOG_LEVEL_STRINGS:
        message = 'invalid choice: {0} (choose from {1})'.format(log_level_string, _LOG_LEVEL_STRINGS)
        raise argparse.ArgumentTypeError(message)

    log_level_int = getattr(logging, log_level_string, logging.INFO)
    # check the logging log_level_choices have not changed from our expected values
    assert isinstance(log_level_int, int)

    return log_level_int


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run batch tests for PosTaggers.')
    parser.add_argument('--pos-tagger', dest='pos_tagger', required=True,
                        help='regex expression to select pos_taggeers to run')
    parser.add_argument('--dataset', dest='dataset',
                        help='regex expression to select datasets (will select all if not set)')
    parser.add_argument('--log-level', default='INFO', dest='log_level',
                        type=_log_level_string_to_int, nargs='?',
                        help='Set the logging output level. {0}'.format(_LOG_LEVEL_STRINGS))
    args = parser.parse_args()

    # Configure log
    root_logger = logging.getLogger()
    root_logger.setLevel(args.log_level)

    # Retrieve data to test against
    taggers = list(postagger_match(args.pos_tagger))
    datasets = list(Dataset.match(args.dataset) if args.dataset else Dataset.all())

    print("Number of taggers: {}".format(len(taggers)))
    print("Number of datasets: {}".format(len(datasets)))

    for tagger_class in taggers:
        tagger = UniversalPOSTagger(tagger_class)()
        tagger.load()

        for dataset_class in datasets:
            dataset = UniversalPOSDataset(dataset_class)()

            log.info("Evaluate tagger {!r} over dataset {!r}".format(tagger, dataset))

            start = time.time()
            metrics = evaluate(tagger, dataset)
            end = time.time()

            log.debug(" - metrics: {}".format(metrics))
            log.debug(" - elapsed_time: {}".format(end))

    """
    for tagger in taggers:
        print(tagger)

    for dataset in datasets:
        data = dataset()
        print("{}: {} tagged_sentences".format(data._id_, len(data.tagged_sents)))
        print("{}: {} tagged_sentences".format(data._id_, len(data.tagged_sents)))
    """