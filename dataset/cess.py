# -*- coding: utf-8 -*-

from nltk.corpus import cess_esp
from dataset.base_dataset import Dataset


@Dataset.register('cess_esp')
class CESSDataset(Dataset):
    tagset = 'es-eagles'

    @property
    def tagged_sents(self):
        if not hasattr(self, '_tagged_sents'):
            setattr(self, '_tagged_sents', self.get_tagged_sentences())
        return getattr(self, '_tagged_sents')

    @classmethod
    def get_tagged_sentences(cls):
        return cess_esp.tagged_sents()
