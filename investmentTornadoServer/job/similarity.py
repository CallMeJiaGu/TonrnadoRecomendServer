#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import array
import gensim
import os
import sys
import codecs
from mycut import FilterCut
from corpora import CorporaWithTitle
from gensim.similarities.index import AnnoyIndexer
from DBUtil import DB
import time
import ConfigParser


class Convert2Vec(object):
    def __init__(self, wm):
        self.wm = wm

    def text2v(self, text, cuttor):
        tokens = cuttor.fltcut(text)
        if len(tokens) == 0:
            return None
        else:
            return self.tokens2v(tokens)

    def tokens2v(self, tokens):
        # assert len(tokens) > 0
        vectors = [self.wm[w] for w in tokens if w in self.wm]
        if len(vectors) == 0: return [0.0 for i in range(self.wm.vector_size)]
        return array(vectors).mean(axis=0)


class Corparo2Vec(object):
    def __init__(self, wm):
        self.wm = wm
        self.c2v = Convert2Vec(wm)

    def genvec(self, corparo_with_title):
        for title, tokens in corparo_with_title:
            v = self.c2v.tokens2v(tokens)
            yield title, v


def getRowCount(f):
    output = os.popen('wc -l ' + f)
    s = output.read()
    output.close()
    return s.split()[0]


def saveVecs(infile, target, wm):
    corp2v = Corparo2Vec(wm)
    corp_with_title = CorporaWithTitle(infile)
    rows = getRowCount(infile)
    with codecs.open(target, 'w', 'utf-8') as f:
        f.write('%s %s%s' % (rows, wm.vector_size, os.linesep))
        for title, vec in corp2v.genvec(corp_with_title):
            l = u'%s %s %s' % (title, ' '.join(['%.6f' % v for v in vec]), os.linesep)
            f.write(l)


class Recommander(object):
    def __init__(self, vec_file, pap, pat, pro):
        # self.wm = gensim.models.KeyedVectors.load_word2vec_format(vec_file,binary=True)
        self.wm = gensim.models.word2vec.Word2Vec.load_word2vec_format(vec_file, binary=True)
        self.paper_index = AnnoyIndexer()
        self.paper_index.load(pap)
        self.patent_index = AnnoyIndexer()
        self.patent_index.load(pat)
        self.project_index = AnnoyIndexer()
        self.project_index.load(pro)
        self.t2v = Convert2Vec(self.wm)
        self.cuttor = FilterCut()
        self.db = DB()
        self.featureIndex = self.buildFeatureIndex()

    def buildFeatureIndex(self):
        paperFeature = open("/testdata400/data/recommender/data0828/feature/paper_feature.txt", 'r')
        patentFeature = open("/testdata400/data/recommender/data0828/feature/patent_feature.txt", 'r')
        projectFeature = open("/testdata400/data/recommender/data0828/feature/project_feature.txt", 'r')
        featureIndex = {}
        featureIndex['paper'] = self.loadFeature(paperFeature)
        featureIndex['patent'] = self.loadFeature(patentFeature)
        featureIndex['project'] = self.loadFeature(projectFeature)
        return featureIndex

    def loadFeature(self, file):
        file = file.readlines()
        index = {}
        index['field'] = {}
        index['type'] = {}
        index['province'] = {}
        index['unit'] = {}
        for line in file:
            feature = line.split('\t')
            if feature[1] not in index['field']:
                index['field'][feature[1]] = []
            index['field'][feature[1]].append(feature[0])
            if feature[2] not in index['type']:
                index['type'][feature[2]] = []
            index['type'][feature[2]].append(feature[0])
            if feature[3] not in index['province']:
                index['province'][feature[3]] = []
            index['province'][feature[3]].append(feature[0])
            if feature[4] not in index['unit']:
                index['unit'][feature[4]] = []
            index['unit'][feature[4]].append(feature[0])
        return index

    # 过滤论文，项目，专利
    def filter(self, typee, topDocs, filterParams, topN):
        topDocIds = [i for i, j in topDocs]
        if not (filterParams[0] == '' or filterParams[
            0] == '-1' or typee == 'project'):  # field, 项目没有type，不用过滤，参数为空字符串或者-1表示不过滤
            if filterParams[0] not in self.featureIndex[typee]['field']:
                topDocIds = []
            topDocIds = list(set(topDocIds).intersection(self.featureIndex[typee]['field'][filterParams[0]]))
        if not (filterParams[1] == '' or filterParams[1] == '-1'):  # type
            if filterParams[1] not in self.featureIndex[typee]['type']:
                topDocIds = []
            topDocIds = list(set(topDocIds).intersection(self.featureIndex[typee]['type'][filterParams[1]]))
        if not (filterParams[2] == '' or filterParams[2] == '-1'):  # province
            if filterParams[2] not in self.featureIndex[typee]['province']:
                topDocIds = []
            topDocIds = list(set(topDocIds).intersection(self.featureIndex[typee]['province'][filterParams[2]]))
        if not (filterParams[3] == '' or filterParams[3] == '-1'):  # unit
            if filterParams[3] not in self.featureIndex[typee]['unit']:
                topDocIds = []
            topDocIds = list(set(topDocIds).intersection(self.featureIndex[typee]['unit'][filterParams[3]]))
        result = []
        for i in topDocs:
            if i[0] in topDocIds:
                result.append(i)
            if len(result) == topN:
                break
        return result

    # 不过滤地区，且返回全部满足的文档，而不仅仅是topn个文档
    # def filterForExpert(self, typee, topDocs, filterParams):
    #     topDocIds = [i for i,j in topDocs]
    #     if not (filterParams[0] == '' or filterParams[
    #         0] == '-1' or typee == 'project'):  # field, 项目没有type，不用过滤，参数为空字符串或者-1表示不过滤
    #         if filterParams[0] not in self.featureIndex[typee]['field']:
    #             topDocIds = []
    #         topDocIds = list(set(topDocIds).intersection(self.featureIndex[typee]['field'][filterParams[0]]))
    #     if not (filterParams[1] == '' or filterParams[1] == '-1'):  # type
    #         if filterParams[1] not in self.featureIndex[typee]['type']:
    #             topDocIds = []
    #         topDocIds = list(set(topDocIds).intersection(self.featureIndex[typee]['type'][filterParams[1]]))
    #     if not (filterParams[3] == '' or filterParams[3] == '-1'):  # unit
    #         if filterParams[3] not in self.featureIndex[typee]['unit']:
    #             topDocIds = []
    #         topDocIds = list(set(topDocIds).intersection(self.featureIndex[typee]['unit'][filterParams[3]]))
    #     result = []
    #
    #     topDocsMap = {}
    #     for i in range(len(topDocs)):
    #         topDocsMap[topDocs[i][0]]=topDocs[i][1]
    #     for id in topDocIds:
    #         listTemp = [id,topDocsMap[id]]
    #         result.append(listTemp)
    #     return result

    def most_similar_paper(self, text, topn=10):
        vec = self.t2v.text2v(text, self.cuttor)
        return self.paper_index.most_similar(vec, topn)

    def most_similar_patent(self, text, topn=10):
        vec = self.t2v.text2v(text, self.cuttor)
        return self.patent_index.most_similar(vec, topn)

    def most_similar_project(self, text, topn=10):
        vec = self.t2v.text2v(text, self.cuttor)
        return self.project_index.most_similar(vec, topn)

    def getSimExpertsIds(self, topDocs):
        expertInfoOut = {}
        expertMap = {}
        authorSeqWeiht = [1.0, 0.85, 0.7, 0.5]
        for typee in topDocs:
            order = {}
            order[typee] = {}
            k = 0
            for i, j in topDocs[typee]:
                order[typee][i] = k
                k = k + 1
            ids = [i for i, j in topDocs[typee]]
            docExpertIds = self.db.getAuthors(typee, ids)
            for id in docExpertIds:
                if not self.db.idInDB(typee, id):
                    print "docId:" + id + "is not in db"
                    continue
                expertIds = docExpertIds[id]
                qs = 1.0
                sim = qs
                for i, j in topDocs[typee]:
                    if i == id:
                        sim = j * sim
                        break
                for i in range(len(expertIds)):
                    if i >= 4:  # 一个成果考虑4个作者
                        break
                    if expertIds[i] not in expertInfoOut:
                        expertInfoOut[expertIds[i]] = []
                    expertInfoOut[expertIds[i]].append([typee + str(order[typee][id]), sim * authorSeqWeiht[i], i])
                    if expertIds[i] not in expertMap:
                        expertMap[expertIds[i]] = []
                    expertMap[expertIds[i]].append(sim * authorSeqWeiht[i])
        return expertMap, expertInfoOut

    # 从成果提取专家，有些专家在不过滤省份时排在前，但过滤省份后排在后，为避免此情况，先不过滤成果的地区，
    # 从这些不过滤地区的成果中提取专家，再按地区过滤专家，若不足topN，再在过滤地区的成果中找剩余的专家
    #
    # 这个函数需要重构，但是八成需求会改，所以先不重构了
    def most_similar_expert(self, topPapers, topPatents, topProjects, filterParams, expertTopN):
        file = open("config.ini", 'r')
        config = ConfigParser.ConfigParser()
        config.readfp(file)
        LEN = int(config.get('global', 'len'))  # 对于一个专家要计算多少他的成果
        COE = float(config.get('global', 'coe'))  # 对于一个专家，从第二个的成果相似度乘的系数
        topDocs = {}
        topDocs['paper'] = self.filter('paper', topPapers, filterParams, 50)
        topDocs['patent'] = self.filter('patent', topPatents, filterParams, 50)
        topDocs['project'] = self.filter('project', topProjects, filterParams, 15)
        expertMap, expertInfoOut = self.getSimExpertsIds(topDocs)  # 专家id为key，各项成果的相似度list为value
        expertScoreMap = {}  # 专家为key，评分为value
        for expert in expertMap:
            expertMap[expert].sort(reverse=True)
            sim = expertMap[expert][0]
            for i in range(1, len(expertMap[expert])):
                if i >= LEN:
                    break
                sim = sim + COE * expertMap[expert][i]
            expertScoreMap[expert] = sim
        result = sorted(expertScoreMap.items(), key=lambda item: item[1], reverse=True)[0:expertTopN]
        out = []
        for i in result:
            if i[0] in expertInfoOut:
                out.append({i[0]: expertInfoOut[i[0]]})
                # out[i[0]]=expertInfoOut[i[0]]
        self.printOut(out, LEN)
        return result

    def printOut(self, out, l):
        name = str('log/' + time.strftime("%Y-%m-%d %H-%M-%S" + ".txt", time.localtime()))
        print name
        output = open(name, 'w')
        for expert in out:
            for i in expert:
                list = expert[i]
                expert[i] = sorted(list, key=lambda doc: doc[1], reverse=True)[0:l]
        for expert in out:
            for i in expert:
                # print i  # 作者id
                output.write(i + '\n')
                list = expert[i]  # list为doc信息
                docOrder = ''
                for j in list:
                    docOrder = docOrder + j[0] + '                  '
                # print docOrder
                output.write(docOrder + '\n')
                sim = ''
                for j in list:
                    sim = sim + str(j[1]) + '             '
                # print sim
                output.write(sim + '\n')
                expertOrder = ''
                for j in list:
                    expertOrder = expertOrder + str(j[2]) + '                            '
                # print expertOrder
                output.write(expertOrder + '\n')
                output.write("\n")
        output.close()

    # def most_similar_expert(self, text, topDocs):
    #     expertMap = self.getSimExpertsIds(topDocs)  # 专家id为key，各项成果的相似度list为value
    #     expertScoreMap = {}  # 专家为key，评分为value
    #     for expert in expertMap:
    #         expertMap[expert].sort(reverse=True)
    #         sim = expertMap[expert][0]
    #         for i in range(1, len(expertMap[expert])):
    #             if i >= 4:
    #                 break
    #             sim = sim + 0.04 * expertMap[expert][i]
    #         expertScoreMap[expert] = sim
    #     return sorted(expertScoreMap.items(), key=lambda item: item[1], reverse=True)

    def get_model(self):
        return self.wm

    def get_cuttor(self):
        return self.cuttor


if __name__ == '__main__':
    recmder = Recommander('/testdata/data/wm.bin', '/testdata/data/ind/paper.ind', '/testdata/data/ind/patent.ind',
                          '/testdata/data/ind/project.ind')
    while 1:
        print "please intpu your demand:"
        txt = raw_input()
        topn = int(raw_input())
        if txt == "000":
            break
            #   print txt
        result = recmder.most_similar_paper(txt, topn)
        for a, b in result:
            print "a:" + a
