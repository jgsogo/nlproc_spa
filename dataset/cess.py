# -*- coding: utf-8 -*-

from nltk.corpus import cess_esp
from dataset.base_dataset import Dataset


@Dataset.register('cess')
class CESSDataset(Dataset):
    tagset = 'es-eagles.map'

    @property
    def tagged_sents(self):
        if not hasattr(self, '_tagged_sents'):
            setattr(self, '_tagged_sents', cess_esp.tagged_sents())
        return getattr(self, '_tagged_sents')

    def get_tagged_sentences(self):
        return cess_esp.tagged_sents()