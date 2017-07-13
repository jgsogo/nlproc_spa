# -*- coding: utf-8 -*-

from nltk.corpus import universal_treebanks
from dataset.base_dataset import Dataset

@Dataset.register('universal_treebanks')
class UniversalTreebanksDataset(Dataset):
    tagset = 'universal'

    @property
    def tagged_sents(self):
        if not hasattr(self, '_tagged_sents'):
            setattr(self, '_tagged_sents', self.get_tagged_sentences())
        return getattr(self, '_tagged_sents')

    @classmethod
    def get_tagged_sentences(cls):
        return universal_treebanks.tagged_sents(fileids=['std/es/es-universal-train.conll',])

