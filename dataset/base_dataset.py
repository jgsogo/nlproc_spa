# -*- coding: utf-8 -*-

import re
from nltk.tag.util import untag


class Dataset:
    _id_ = None
    _datasets_ = {}

    def __str__(self):
        return self._id_

    @property
    def tagged_sents(self):
        raise NotImplementedError()

    @classmethod
    def untag(cls, sentence):
        return untag(sentence)

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._datasets_[name] = subclass
            subclass._id_ = name
            return subclass
        return decorator

    @classmethod
    def get(cls, name):
        try:
            return cls._datasets_[name]
        except KeyError:
            raise ValueError(name, "Unknown entity")

    @classmethod
    def match(cls, regex_expr):
        for key, item in cls._datasets_.items():
            if re.match(regex_expr, key):
                yield item

    @classmethod
    def all(cls):
        return cls._datasets_.values()