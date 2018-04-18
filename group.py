#!/usr/bin/env python3
# -_- coding:utf-8 -_-

import json
import operator
import nltk
import jieba.posseg


meanings_lai = ['①动从另外的地方到说话人这里（跟“去”或“往”相对）。',
                '②形未来的。',
                '③动用在动词后，表示动作朝着说话人所在的地方。',
                '④名从过去到说话时为止的一段时间。',
                '⑤助用在“十” “百” “千”等整数或数量短语后面，表示概数，通常略小于那个数目。',
                '⑥动（事情、问题等）来到；发生。',
                '⑦动用在动词性短语（或介词短语）与动词（或动词性短语）之间，表示前者是方法、态度，后者是目的。',
                '⑧动a. 用在动词性短语后面，表示来做某事。b. 用在动词性短语前面，表示要做某事。',
                '⑨动表示做某个动作（代替意义具体的动词）。',
                '⑩动跟“得”或“不”连用，表示能够或不能够。']


class ParallelCorpus:

    def __init__(self, filename1, filename2, keyword):
        self.keyword = keyword
        self.lines = filter(lambda x: keyword in x[0], zip(open(filename1), open(filename2)))
        self.pairs = {}

    def __str__(self):
        return str(self.pairs)

    __repr__ = __str__

    def rough_group(self):
        for line in self.lines:
            for p in jieba.posseg.cut(line[0]):
                if self.keyword in p.word:
                    self.pairs.setdefault(p.flag, []).append((*line, p.word))

    @staticmethod
    def feature_simplecut(data, key):
        word = data[2]
        if word == key:
            rest = None
        else:
            idx = word.index(key)
            rest = word[:idx] + word[min(idx+1, len(word)):]
        return {'rest_of_phrase': rest}, data[3]

    @classmethod
    def feature_adjacent(cls, data, key):
        result = cls.feature_simplecut(data, key)[0]
        pairlist = jieba.posseg.lcut(data[0])
        wordlist = list(map(operator.attrgetter('word'), pairlist))
        flaglist = list(map(operator.attrgetter('flag'), pairlist))
        idx = wordlist.index(data[2])
        result['previous'] = flaglist[idx-1]
        result['next'] = flaglist[idx+1]
        result['previous_word'] = wordlist[idx-1]
        result['next_word'] = wordlist[idx+1]
        return result, data[3]

    def verb_group(self):
        featureset = [self.feature_adjacent(data, self.keyword)
                      for data in json.load(open('tagged.json'))]
        trainset, testset = featureset, featureset[50:]
        classifier = nltk.NaiveBayesClassifier.train(trainset)
        print(classifier.show_most_informative_features(5))
        return nltk.classify.accuracy(classifier, testset)

    def output(self, filename):
        for k in self.pairs.keys():
            print('%s: %d' % (k, len(self.pairs[k])))
        with open(filename, 'w') as f:
            json.dump(self.pairs, f, ensure_ascii=False)


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
    # p.rough_group()
    # p.output('test.json')
    print(p.verb_group())

if __name__ == '__main__':
    main()
