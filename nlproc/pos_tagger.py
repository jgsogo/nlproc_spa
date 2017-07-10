#!/usr/bin/env python
# -*- coding: utf-8 -*-


class POSTagger(object):
    _id_ = None
    _classes_ = {}

    def __init__(self, lang='spa'):
        self.lang = lang
        assert self.lang == 'spa', "Only spanish language for now"

    def load(self, train=True):
        raise NotImplementedError()

    def train(self, save=True):
        raise NotImplementedError()

    def tag(self, tokens):
        raise NotImplementedError()

    def tag_sents(self, sents):
        raise NotImplementedError()

    @classmethod
    def factory(cls, class_name):
        try:
            return cls._classes_[class_name]
        except KeyError:
            raise ValueError(class_name, "Unknown entity")

    @classmethod
    def register(cls, class_name):
        def decorator(subclass):
            cls._classes_[class_name] = subclass
            subclass._id_ = class_name
            return subclass
        return decorator


postagger_factory = POSTagger.factory
postagger_register = POSTagger.register
