#!/usr/bin/env python3
# -_- coding:utf-8 -_-


import nltk
import jieba.posseg


class ParallelCorpus:

    def __init__(self, filename1, filename2, keyword):
        self.keyword = keyword
        self.lines = filter(lambda x: keyword in x[0], zip(open(filename1), open(filename2)))
        self.results = {}

    def __str__(self):
        return str(self.results)

    __repr__ = __str__

    def rough_group(self):
        for line in self.lines:
            for p in jieba.posseg.cut(line[0]):
                if self.keyword in p.word:
                    self.results.setdefault(p.flag, []).append(line)

    def output(self, filename):
        for k in self.results.keys():
            print('%s: %d' % (k, len(self.results[k])))
        with open(filename, 'w') as f:
            f.write(str(self))


# a:        形容词
# b:        区别词
# c:        连词
# d:        副词
# e:        叹词
# g:        语素字
# h:        前接成分
# i:        习用语
# j:        简称
# k:        后接成分
# m:        数词
# n:        普通名词
# nd:       方位名词
# nh:       人名
# ni:       机构名
# nl:       处所名词
# ns:       地名
# nt:       时间词
# nz:       其他专名
# o:        拟声词
# p:        介词
# q:        量词
# r:        代词
# u:        助词
# v:        动词
# wp:       标点符号
# ws:       字符串
# x:        非语素字

def main():
    p = ParallelCorpus('testsets/devset/UNv1.0.devset.zh',
                       'testsets/devset/UNv1.0.devset.es', '来')
    p.rough_group()
    p.output('test.txt')

if __name__ == '__main__':
    main()
