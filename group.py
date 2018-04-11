#!/usr/bin/env python3
# -_- coding:utf-8 -_-


import nltk


class ParallelCorpus:

    def __init__(self, filename1, filename2, keyword):
        self.lines = filter(lambda x: keyword in x[0], zip(open(filename1), open(filename2)))

    def test(self):
        pass


def main():
    p = ParallelCorpus('testsets/devset/UNv1.0.devset.zh',
                       'testsets/devset/UNv1.0.devset.es', '来')
