# -*- coding: utf-8 -*-

import os
import logging

log = logging.getLogger(__name__)

me = os.path.abspath(os.path.dirname(__file__))


def load(map_file):
    data = {}
    with open(os.path.join(me, "..", "universal_pos_tags", map_file), 'r') as f:
        for line in f:
            i, o = line.split()
            data[i.lower()] = o
    return data


class UniversalPOSMixin(object):

    def __init__(self, tagset, *args, **kwargs):
        super(UniversalPOSMixin, self).__init__(*args, **kwargs)
        self.mapping = load(tagset)

    def universal_pos(self, tag):
        try:
            return self.mapping[tag]
        except KeyError as e:
            log.error("Tag {!r} not found in UniversalPOS mapping for {!r}".format(tag, self.tagset))
            return ''


def UniversalPOSTagger(cls):

    class UniversalPOSWrapper(UniversalPOSMixin, cls):

        def __init__(self, *args, **kwargs):
            super(UniversalPOSWrapper, self).__init__(cls.tagset, *args, **kwargs)

        def tag(self, *args, **kwargs):
            r = super(UniversalPOSWrapper, self).tag(*args, **kwargs)
            for w, tag in r:
                yield (w, self.universal_pos(tag.lower()))

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
