#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import pickle

from nltk import UnigramTagger, BigramTagger, TrigramTagger, NgramTagger
from nltk.corpus import cess_esp

log = logging.getLogger(__name__)


class BaseTagger(object):
    id = None
    ngrams = 2

    def __init__(self, id, use_mwe=True, ngrams=2):
        self.id = id
        self.use_mwe = use_mwe  # Whether to use or not multiword expressions
        self.ngrams = ngrams

    def get_ngram_tagger_filename(self, id, use_mwe, ngram):
        mwe = "" if not use_mwe else "_mwe"
        ngram_str = "{}-gram".format(ngram)
        return "{id}{mwe}_{ngram}.tagger".format(id=id, mwe=mwe, ngram=ngram_str)

    @classmethod
    def _load(cls, filename):
        with open(filename,'rb') as f:
            return pickle.load(f)

    @classmethod
    def _save(cls, filename, tagger):
        with open(filename, 'wb') as f:
            pickle.dump(tagger, f)

    @classmethod
    def _remove_mwe(cls, sentences):
        nomwe = []
        for i in sentences:
            nomwe = " ".join([j[0].replace("_"," ") for j in i])
            nomwe.append(nomwe.split())
        return nomwe

    def load(self, train=True):
        log.info("Load tagger {!r}".format(self.id))
        try:
            for i in range(1, self.ngrams+1):
                filename = self.get_ngram_tagger_filename(self.id, self.use_mwe, i)
                log.debug(" - load tagger (ngram={!r}) from filename {!r}".format(i, filename))
                self.tagger = self._load(filename)

        except IOError as e:
            if train:
                log.warn(" - file failed to load, train tagger")
                self.train(tagged_sentences=None, save=True)
            else:
                raise e

    def train(self, tagged_sentences, save=True):
        log.info("Train tagger {!r} (use_mwe={!r}) up to ngram={!r}".format(self.id, self.use_mwe, self.ngrams))

        if not self.use_mwe:
            log.debug(" - remove mwe from tagged_sentences")
            nomwe = self._remove_mwe(tagged_sentences)
            log.debug(" - pos_tag sentences without mwe")
            tagged_sentences = CESSTagger.pos_tag(nomwe, use_mwe=False, ngrams=1)  # TODO: Get this class

        backoff = None
        for i in range(1, self.ngrams+1):
            log.debug(" - train tagger (ngram={!r})".format(i))
            tagger = NgramTagger(i, tagged_sentences, backoff=backoff)
            if save:
                filename = self.get_ngram_tagger_filename(self.id, self.use_mwe, i)
                log.debug(" - save tagger (ngram={!r}) to filename {!r}".format(i, filename))
                self._save(filename, tagger)
            backoff = tagger

        self.tagger = tagger

    @classmethod
    def pos_tag(cls, tokens, use_mwe=True, ngrams=2):
        item = cls(BaseTagger.id, use_mwe=use_mwe, ngrams=ngrams)
        item.load()
        return item.tag(tokens)

    @classmethod
    def pos_tag_sents(cls, sentences, use_mwe=True, ngrams=2):
        item = cls(BaseTagger.id, use_mwe=use_mwe, ngrams=ngrams)
        item.load()
        return item.tag_sents(sentences)

    def tag(self, tokens):
        return self.tagger.tag(tokens)

    def tag_sents(self, sents):
        return self.tagger.tag_sents(sents)


class CESSTagger(BaseTagger):

    def __init__(self, use_mwe=True, ngrams=2):
        super(CESSTagger, self).__init__(id='cess', use_mwe=True, ngrams=ngrams)

    def train(self, tagged_sentences=None, save=True):
        assert(tagged_sentences==None), "Do not pass arguments to this method. Attribute is here only to match parent function"
        # Load CESS corpus.
        cess_sents = cess_esp.tagged_sents()
        super(CESSTagger, self).train(tagged_sentences=cess_sents, save=save)


if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    ch.setLevel(logging.DEBUG)
    log.setLevel(logging.DEBUG)

    cess = CESSTagger(use_mwe=True, ngrams=5)
    cess.load(train=True)
    tags = cess.tag(["La", "casa", "es", "azul"])
    print(tags)
