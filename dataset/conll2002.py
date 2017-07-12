# -*- coding: utf-8 -*-

from nltk.corpus import conll2002
from dataset.base_dataset import Dataset


@Dataset.register('conll2002')
class Conll2002Dataset(Dataset):
    tagset = 'es-eagles'

    @property
    def tagged_sents(self):
        if not hasattr(self, '_tagged_sents'):
            setattr(self, '_tagged_sents', conll2002.tagged_sents())
        return getattr(self, '_tagged_sents')

    def get_tagged_sentences(self):
        return conll2002.tagged_sents(fileids=['esp.testa', 'esp.testb'])

