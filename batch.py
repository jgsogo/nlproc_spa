#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nlproc.pos_tagger import postagger_factory

import logging
log = logging.getLogger(__name__)


def load_test_sentences():
    pass

def test_sentence(tagger, sentence):
    
    return False

def run_batch(postagger_id, test_sentences):
    log.info("Run batch for {!r}".format(postagger_id))

    item = postagger_factory(id)()
    item.load()



if __name__ == '__main__':
    pass