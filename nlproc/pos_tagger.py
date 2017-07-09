#!/usr/bin/env python
# -*- coding: utf-8 -*-


class POSTagger(object):

    def __init__(self, lang='spa'):
        self.lang = lang
        assert(self.lang == 'spa', "Only spanish language for now")

    def list(self):
