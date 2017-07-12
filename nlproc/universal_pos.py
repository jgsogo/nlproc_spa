# -*- coding: utf-8 -*-

import os
from nltk.tag import tagset_mapping

import logging
log = logging.getLogger(__name__)


class UniversalPOSMixin(object):
    missing = set()

    def __init__(self, tagset, *args, **kwargs):
        super(UniversalPOSMixin, self).__init__(*args, **kwargs)
        self.mapping = tagset_mapping(tagset, 'universal')

    def universal_pos(self, tag):
        try:
            return self.mapping[tag]
        except KeyError as e:
            self.missing.add(tag)
            # log.error("Tag {!r} not found in UniversalPOS mapping for {!r}".format(tag, self.tagset))
            return None


def UniversalPOSTagger(cls):

    class UniversalPOSWrapper(UniversalPOSMixin, cls):

        def __init__(self, *args, **kwargs):
            super(UniversalPOSWrapper, self).__init__(cls.tagset, *args, **kwargs)

        def tag(self, *args, **kwargs):
            r = super(UniversalPOSWrapper, self).tag(*args, **kwargs)
            for w, tag in r:
                yield (w, self.universal_pos(tag.lower()) if tag else None)

        def tag_sents(self, *args, **kwargs):
            r = super(UniversalPOSWrapper, self).tag_sents(*args, **kwargs)
            return r

    return UniversalPOSWrapper


def UniversalPOSDataset(cls):

    class UniversalPOSWrapperDataset(UniversalPOSMixin, cls):

        def __init__(self, *args, **kwargs):
            super(UniversalPOSWrapperDataset, self).__init__(cls.tagset, *args, **kwargs)

        @property
        def tagged_sents(self):
            for sent in super(UniversalPOSWrapperDataset, self).tagged_sents:
                yield [(w, self.universal_pos(tag.lower())) for w, tag in sent]

    return UniversalPOSWrapperDataset
