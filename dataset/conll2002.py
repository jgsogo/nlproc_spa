# -*- coding: utf-8 -*-

from nltk.corpus import conll2002
from dataset.base_dataset import Dataset


@Dataset.register('conll2002')
class Conll2002Dataset(Dataset):
    tagset = 'en-ptb'

    @property
    def tagged_sents(self):
        if not hasattr(self, '_tagged_sents'):
            setattr(self, '_tagged_sents', self.get_tagged_sentences())
        return getattr(self, '_tagged_sents')

    @classmethod
    def get_tagged_sentences(cls):
        tags = set()
        for it in conll2002.tagged_sents():
            for i in it:
                tags.add(i[1])
            yield it
        print(sorted(tags))
        exit()
        #return conll2002.tagged_sents(fileids=['esp.testa', 'esp.testb', 'esp.train'])[:2]

