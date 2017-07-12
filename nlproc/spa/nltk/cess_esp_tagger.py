#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.corpus import cess_esp

from nlproc.pos_tagger import postagger_register
from .ngram_tagger import NLTKNgramTagger

import logging
log = logging.getLogger(__name__)


CESS_ESP = 'cess_esp'


def cess_tagger(use_mwe, ngrams):

    class CESSTagger(NLTKNgramTagger):
        tagset = 'wsj'

        def __init__(self):
            super(CESSTagger, self).__init__(id=CESS_ESP, use_mwe=use_mwe, ngrams=ngrams)

        def get_tagged_sentences(self):
            return cess_esp.tagged_sents()

        @classmethod
        def pos_tag(cls, tokens, use_mwe=True, ngrams=2):
            item_class = cess_tagger(use_mwe=use_mwe, ngrams=ngrams)
            item = item_class()
            item.load(train=True)
            return item.tag_sents(tokens)

    return CESSTagger


CESSTagger = cess_tagger(use_mwe=True, ngrams=2)

postagger_register("{}-mwe-1grams".format(CESS_ESP))(cess_tagger(use_mwe=True, ngrams=1))
postagger_register("{}-mwe-2grams".format(CESS_ESP))(cess_tagger(use_mwe=True, ngrams=2))
postagger_register("{}-mwe-3grams".format(CESS_ESP))(cess_tagger(use_mwe=True, ngrams=3))

postagger_register("{}-nomwe-1grams".format(CESS_ESP))(cess_tagger(use_mwe=False, ngrams=1))
postagger_register("{}-nomwe-2grams".format(CESS_ESP))(cess_tagger(use_mwe=False, ngrams=2))
postagger_register("{}-nomwe-3grams".format(CESS_ESP))(cess_tagger(use_mwe=False, ngrams=3))


if __name__ == '__main__':
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)7s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(levelname)7s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    ch.setLevel(logging.DEBUG)
    log.setLevel(logging.DEBUG)

    log.addHandler(ch)

    from nlproc.pos_tagger import postagger_factory

    cess_mwe_2grams = postagger_factory("{}-nomwe-2grams".format(CESS_ESP))()
    cess_mwe_2grams.load(train=True)
    print(cess_mwe_2grams.tag(["La", "casa", "es", "azul"]))

    """
    cess_mwe_1grams = postagger_factory("{}-mwe-1grams".format(CESS_ESP))()
    cess_mwe_1grams.load(train=True)
    print(cess_mwe_1grams.tag(["La", "casa", "es", "azul"]))

    cess_mwe_3grams = postagger_factory("{}-mwe-3grams".format(CESS_ESP))()
    cess_mwe_3grams.load(train=True)
    print(cess_mwe_3grams.tag(["La", "casa", "es", "azul"]))
    """
