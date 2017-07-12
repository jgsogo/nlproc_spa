#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import logging
import pickle

from nltk import UnigramTagger, BigramTagger, TrigramTagger, NgramTagger

from nlproc.pos_tagger import POSTagger

log = logging.getLogger(__name__)


class NLTKNgramTagger(POSTagger):
    id = None
    ngrams = 2

    def __init__(self, id, use_mwe=True, ngrams=2):
        super(NLTKNgramTagger, self).__init__()
        self.id = id
        self.use_mwe = use_mwe  # Whether to use or not multiword expressions
        self.ngrams = ngrams

    def get_ngram_tagger_filename(self, id, use_mwe, ngram):
        mwe = "nomwe" if not use_mwe else "mwe"
        ngram_str = "{}grams".format(ngram)
        return "{id}_{mwe}_{ngram}.tagger".format(id=id, mwe=mwe, ngram=ngram_str)

    def get_tagged_sentences(self):
        raise NotImplementedError("NLTKNgramTagger::get_tagged_sentences")

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
            nomwe.append([j[0].replace("_"," ") for j in i])
        return nomwe

    def load(self, train=True):
        log.info("Load tagger {!r} (mwe={}, ngrams={})".format(self.id, self.use_mwe, self.ngrams))

        self.tagger = None
        for i in range(1, self.ngrams+1):
            filename = self.get_ngram_tagger_filename(self.id, self.use_mwe, i)
            log.debug(" - load tagger (ngram={!r}) from filename {!r}".format(i, filename))

            try:
                self.tagger = self._load(filename)

            except IOError as e:
                if train:
                    log.debug(" - file failed to load, train tagger")
                    self.tagger = self._train(ngrams=i, backoff=self.tagger, save=True)
                else:
                    raise e

    def _train(self, ngrams, backoff, save=True):
        log.debug(" - train tagger (ngram={!r})".format(ngrams))
        tagger = NgramTagger(ngrams, self.tagged_sentences, backoff=backoff)

        if save:
            filename = self.get_ngram_tagger_filename(self.id, self.use_mwe, ngrams)
            log.debug(" - save tagger (ngram={!r}) to filename {!r}".format(ngrams, filename))
            self._save(filename, tagger)
        return tagger

    @property
    def tagged_sentences(self):
        if not hasattr(self, '_tagged_sentences'):
            log.info("Cache tagged_sentences with mwe={!r}".format(self.use_mwe))
            tagged_sentences = self.get_tagged_sentences()

            if not self.use_mwe:
                log.debug(" - remove mwe from tagged_sentences")
                nomwe = self._remove_mwe(tagged_sentences)
                log.debug(" - pos_tag sentences without mwe")
                tagged_sentences = self.__class__.pos_tag(nomwe, use_mwe=True, ngrams=1)

            setattr(self, '_tagged_sentences', tagged_sentences)

        return getattr(self, '_tagged_sentences')

    def train(self, save=True):
        log.info("Train tagger {!r} (use_mwe={!r}) up to ngram={!r}".format(self.id, self.use_mwe, self.ngrams))

        backoff = None
        for i in range(1, self.ngrams+1):
            tagger = self._train(ngrams=i, backoff=backoff, save=save)
            backoff = tagger

        self.tagger = tagger

    @classmethod
    def pos_tag(cls, tokens, use_mwe=True, ngrams=2):
        raise NotImplementedError()

    """
    @classmethod
    def pos_tag_sents(cls, sentences, use_mwe=True, ngrams=2):
        item = cls(id=NLTKNgramTagger.id, use_mwe=use_mwe, ngrams=ngrams)
        item.load()
        return item.tag_sents(sentences)
    """

    def tag(self, tokens):
        return self.tagger.tag(tokens)

    def tag_sents(self, sents):
        return self.tagger.tag_sents(sents)


