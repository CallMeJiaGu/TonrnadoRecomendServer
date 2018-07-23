#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import array
import gensim
import os
import codecs
from mycut import FilterCut
from corpora import CorporaWithTitle
from gensim.similarities.index import AnnoyIndexer
from DBUtil import DB
import time
import ConfigParser
from annoy import AnnoyIndex
import logging.config
from redisUtil import RedisUtil
import utils

logging.config.fileConfig('logger.conf')
logger = logging.getLogger('recommServerLog')

field_list = ['A1', 'C1', 'A4', 'A2', 'A9', 'C2', 'A5', 'A3', 'Z1', 'Z9']
unit_types = ['01', '02', '03', '99']  # 大专院校, 科研院所, 企业
unit_types_project = ['01', '02', '99']


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
        assert len(tokens) > 0
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

        self.paper_index = self.load_index_investment(pap)
        self.patent_index = self.load_index_investment(pat)
        self.project_index = self.load_index_investment(pro)

        self.t2v = Convert2Vec(self.wm)
        self.cuttor = FilterCut()
        # self.redis = RedisUtil()
        self.featureIndex = self.buildFeatureIndex_investment()

    def load_index_investment(self,path):
        index = AnnoyIndexer()
        for parent, dirnames, filenames in os.walk(path):
            for filename in filenames:
                # 生成的B.ind.d 是不能加载进来的，只能加载B.ind
                if len(filename.split('.')) == 2:
                    logger.info(u'文件名为%s ,路径为：%s' % (str(filename.split('.')[0]) , os.path.join(parent, filename)) )
                    index = AnnoyIndexer()
                    index.load(os.path.join(parent, filename))
        return index

    def field_deliver_investment(self,typee, text, topn):
        try:
            vec = self.t2v.text2v(text, self.cuttor)
            results = []
            if typee == "paper":
                results = results + self.paper_index.most_similar(vec,topn)
            if typee == "patent":
                results = results + self.patent_index.most_similar(vec,topn)
            if typee == "project":
                results = results + self.project_index.most_similar(vec,topn)
            return sorted(dict(results).items(), key=lambda x: x[1], reverse=True)
        except Exception, e:
            logger.info(u'出现异常 %s'% e.message )
            return []

    def buildFeatureIndex_investment(self):
        logger.info(u'开始导入特征文件...')
        paperFeature = open("/home/tdlab/investment_recomender/data_filter/feature/paper.txt", 'r')
        patentFeature = open("/home/tdlab/investment_recomender/data_filter/feature/patent.txt", 'r')
        projectFeature = open("/home/tdlab/investment_recomender/data_filter/feature/project.txt", 'r')
        featureIndex = {}
        featureIndex['paper'] = self.loadFeature_inverstment(paperFeature, 'paper')
        featureIndex['patent'] = self.loadFeature_inverstment(patentFeature, 'patent')
        featureIndex['project'] = self.loadFeature_inverstment(projectFeature, 'project')
        logger.info(u'导入特征文件完成')
        return featureIndex

    def loadFeature_inverstment(self, file, typee):
        try:
            file = file.readlines()
            index = {}
            index['field'] = {}
            index['unit'] = {}
            index['unitType'] = {}
            index['proviceCode'] = {}
            for line in file:
                feature = line.split('\t')
                # field
                if feature[1] not in index['field']:
                    index['field'][feature[1]] = []
                index['field'][feature[1]].append(feature[0])
                # unitType
                if feature[2] not in index['unitType']:
                    index['unitType'][feature[2]] = []
                index['unitType'][feature[2]].append(feature[0])
                # proviceCode
                if feature[3] not in index['proviceCode']:
                    index['proviceCode'][feature[3]] = []
                index['proviceCode'][feature[3]].append(feature[0])
                # unit
                if feature[4] not in index['unit']:
                    index['unit'][feature[4]] = []
                index['unit'][feature[4]].append(feature[0])
            return index
        except Exception,e:
            logger.info(u'出现异常 %s %s' % (str(e), e.message))

    def filter_investment(self, typee, topDocs, filterParams, topN):
        try:
            topDocIds = [i for i, j in topDocs]
            # filter field
            if not (filterParams[2] == '' or filterParams[2] == '-1'):
                if filterParams[2] not in self.featureIndex[typee]['field']:
                    topDocIds = []
                else:
                    topDocIds = list(set(topDocIds).intersection(self.featureIndex[typee]['field'][filterParams[2]]))
            # filter unit
            if not (filterParams[3] == '' or filterParams[3] == '-1'):
                if filterParams[3] not in self.featureIndex[typee]['unit']:
                    topDocIds = []
                else:
                    topDocIds = list(set(topDocIds).intersection(self.featureIndex[typee]['unit'][filterParams[3]]))
            # filter provice
            if not (filterParams[0] == '' or filterParams[0] == '-1'):
                if filterParams[0] not in self.featureIndex[typee]['proviceCode']:
                    topDocIds = []
                else:
                    topDocIds = list(set(topDocIds).intersection(self.featureIndex[typee]['proviceCode'][filterParams[0]]))
            # filter unitType
            if not (filterParams[1] == '' or filterParams[1] == '-1'):
                if filterParams[1] not in self.featureIndex[typee]['unitType']:
                    topDocIds = []
                else:
                    topDocIds = list(
                        set(topDocIds).intersection(self.featureIndex[typee]['unitType'][filterParams[1]]))
            result = []
            for i in topDocs:
                if i[0] in topDocIds:
                    result.append(i)
                if len(result) == topN:
                    break
            return result
        except Exception,e:
            logger.info(u'出现异常 %s %s' % (str(e),e.message))


    def load_index(self, path):
        index = {}
        nn = 0
        for field in field_list:
            logger.info(u'---------field：' + field)
            index[field] = {}
            for unit_type in unit_types:
                index[field][unit_type] = {}
                for parent, dirnames, filenames in os.walk(path + field + '/' + unit_type + '/'):
                    for filename in filenames:
                        # if len(filename.split('.')) == 2 and (
                        #         'A5' in os.path.join(parent, filename) or 'project' in os.path.join(parent, filename)):
                        if len(filename.split('.')) == 2:
                            nn = nn + 1
                            logger.info(u'创建AnnoyIndexer %s：field=%s,unit_type=%s,province=%s' % (
                                nn, field, unit_type, filename.split('.')[0]))
                            index[field][unit_type][str(filename.split('.')[0])] = AnnoyIndexer()
                            index[field][unit_type][str(filename.split('.')[0])].load(os.path.join(parent, filename))
        return index

    def buildFeatureIndex(self):
        logger.info(u'开始导入特征文件...')
        if utils.get_host_ip() == '10.1.13.49':
            paperFeature = open("/home/tdlab/recommender/data180526/feature/paper_feature180526.txt", 'r')
            patentFeature = open("/home/tdlab/recommender/data180526/feature/patent_feature180526.txt", 'r')
            projectFeature = open("/home/tdlab/recommender/data180526/feature/project_feature180526.txt", 'r')
        else:
            paperFeature = open("/data/Recommender/data_filter/feature/paper_feature180526.txt", 'r')
            patentFeature = open("/data/Recommender/data_filter/feature/patent_feature180526.txt", 'r')
            projectFeature = open("/data/Recommender/data_filter/feature/project_feature180526.txt", 'r')
        featureIndex = {}
        featureIndex['paper'] = self.loadFeature(paperFeature, 'paper')
        featureIndex['patent'] = self.loadFeature(patentFeature, 'patent')
        featureIndex['project'] = self.loadFeature(projectFeature, 'project')
        logger.info(u'导入特征文件完成')
        return featureIndex

    def loadFeature(self, file, typee):
        file = file.readlines()
        index = {}
        index['unit'] = {}
        if typee == 'paper':
            index['journalQuality'] = {}
        if typee == 'patent':
            index['patentType'] = {}
        if typee == 'project':
            index['projectType'] = {}
        for line in file:
            feature = line.split('\t')
            if feature[4] not in index['unit']:
                index['unit'][feature[4]] = []
            index['unit'][feature[4]].append(feature[0])
            if typee == 'paper':
                if feature[7] not in index['journalQuality']:
                    index['journalQuality'][feature[7]] = []
                index['journalQuality'][feature[7]].append(feature[0])
            if typee == 'patent':
                if feature[7] not in index['patentType']:
                    index['patentType'][feature[7]] = []
                index['patentType'][feature[7]].append(feature[0])
            if typee == 'project':
                if feature[7] not in index['projectType']:
                    index['projectType'][feature[7]] = []
                index['projectType'][feature[7]].append(feature[0])
        return index

    # 获取某专家的所有成果对于需求txt的按相似度排序，论文专利项目分别返回前topN
    def expertDocsSort(self, expertId, txt, topN):
        vec = self.t2v.text2v(txt, self.cuttor)
        annoy = AnnoyIndex(200)
        count = 0
        annoy.add_item(count, vec)
        count = count + 1
        db = DB()
        papers = db.getPapers(expertId)
        for p in papers:
            p[3] = self.t2v.text2v(p[1] + p[2], self.cuttor)
            if p[3] is not None:
                annoy.add_item(count, p[3])
                p[3] = annoy.get_distance(0, count)
                count = count + 1
        papers = sorted(papers, key=lambda p: p[3])
        papersFormated = []
        for p in papers:
            if len(papersFormated) == topN:
                break
            map = {}
            if p[0] is not None:
                map['paperId'] = p[0].encode('utf8')
            else:
                map['paperId'] = p[0]
            if p[1] is not None:
                map['name'] = p[1].encode('utf8')
            else:
                map['name'] = p[1]
            if p[4] is not None:
                map['authors'] = p[4].encode('utf8')
            else:
                map['authors'] = p[4]
            if p[5] is not None:
                map['journalName'] = p[5].encode('utf8')
            else:
                map['journalName'] = p[5]
            if p[6] is not None:
                map['year'] = p[6].encode('utf8')
            else:
                map['year'] = p[6]
            papersFormated.append(map)

        count = 0
        annoy.unload()
        annoy.add_item(count, vec)
        count = count + 1
        patents = db.getPatents(expertId)
        for p in patents:
            p[3] = self.t2v.text2v(p[1] + p[2], self.cuttor)
            if p[3] is not None:
                annoy.add_item(count, p[3])
                p[3] = annoy.get_distance(0, count)
                count = count + 1
        patents = sorted(patents, key=lambda p: p[3])
        patentsFormated = []
        for p in patents:
            if len(patentsFormated) == topN:
                break
            map = {}
            if p[0] is not None:
                map['patentId'] = p[0].encode('utf8')
            else:
                map['patentId'] = p[0]
            if p[4] is not None:
                map['publicationNo'] = p[4].encode('utf8')
            else:
                map['publicationNo'] = p[4]
            if p[1] is not None:
                map['name'] = p[1].encode('utf8')
            else:
                map['name'] = p[1]
            if p[5] is not None:
                map['inventors'] = p[5].encode('utf8')
            else:
                map['inventors'] = p[5]
            if p[6] is not None:
                map['applicant'] = p[6].encode('utf8')
            else:
                map['applicant'] = p[6]
            if p[7] is not None:
                map['year'] = p[7].encode('utf8')
            else:
                map['year'] = p[7]
            patentsFormated.append(map)

        count = 0
        annoy.unload()
        annoy.add_item(count, vec)
        count = count + 1
        projects = db.getProjects(expertId)
        for p in projects:
            p[3] = self.t2v.text2v(p[1] + p[2], self.cuttor)
            if p[3] is not None:
                annoy.add_item(count, p[3])
                p[3] = annoy.get_distance(0, count)
                count = count + 1
        projects = sorted(projects, key=lambda p: p[3])
        projectsFormated = []
        for p in projects:
            if len(projectsFormated) == topN:
                break
            map = {}
            if p[0] is not None:
                map['projectId'] = p[0].encode('utf8')
            else:
                map['projectId'] = p[0]
            if p[1] is not None:
                map['name'] = p[1].encode('utf8')
            else:
                map['name'] = p[1]
            if p[4] is not None:
                map['member'] = p[4].encode('utf8')
            else:
                map['member'] = p[4]
            if p[5] is not None:
                map['unit'] = p[5].encode('utf8')
            else:
                map['unit'] = p[5]
            if p[6] is not None:
                map['year'] = p[6].encode('utf8')
            else:
                map['year'] = p[6]
            if p[7] is not None:
                map['type'] = p[7].encode('utf8')
            else:
                map['type'] = p[7]
            projectsFormated.append(map)
        result = {}
        result['papers'] = papersFormated
        result['patents'] = patentsFormated
        result['projects'] = projectsFormated
        return result

    # 过滤论文，项目，专利
    def filter(self, typee, topDocs, filterParams, topN):
        topDocIds = [i for i, j in topDocs]
        if not (filterParams[3] == '' or filterParams[3] == '-1'):  # unit
            if filterParams[3] not in self.featureIndex[typee]['unit']:
                topDocIds = []
            else:
                topDocIds = list(set(topDocIds).intersection(self.featureIndex[typee]['unit'][filterParams[3]]))
        if typee == 'paper' and not (filterParams[4] == '' or filterParams[4] == '-1'):  # journalQuality
            origin_doc = topDocIds
            topDocIds = []
            for param in filterParams[4]:
                topDocIds = topDocIds + list(
                    set(origin_doc).intersection(self.featureIndex[typee]['journalQuality'][param]))
        if typee == 'patent' and not (filterParams[4] == '' or filterParams[4] == '-1'):  # patentType
            if filterParams[4] not in self.featureIndex[typee]['patentType']:
                topDocIds = []
            else:
                topDocIds = list(set(topDocIds).intersection(self.featureIndex[typee]['patentType'][filterParams[4]]))
        if typee == 'project' and not (filterParams[4] == '' or filterParams[4] == '-1'):  # projectType
            if filterParams[4] not in self.featureIndex[typee]['projectType']:
                topDocIds = []
            else:
                topDocIds = list(set(topDocIds).intersection(self.featureIndex[typee]['projectType'][filterParams[4]]))
        result = []

        for i in topDocs:
            if i[0] in topDocIds:
                result.append(i)
            if len(result) == topN:
                break
        return result

    # 支持多领域推荐， 使用field_deliver将不同领域结果拼起来
    def field_deliver(self, typee, text, topn, fields, u_type, province):
        if fields == '-1':
            field_split = field_list
        else:
            field_split = fields.split(',')
        results = []
        for field in field_split:
            if typee == 'paper':
                results = results + self.most_similar_paper(text, topn, field, u_type, province)
            elif typee == 'patent':
                results = results + self.most_similar_patent(text, topn, field, u_type, province)
        return sorted(dict(results).items(), key=lambda x: x[1], reverse=True)

    def most_similar_paper(self, text, topn, field, u_type, province):
        try:
            results = []
            vec = self.t2v.text2v(text, self.cuttor)
            if province == '-1':
                if u_type == '-1':
                    for unit_type in unit_types:
                        for key in self.paper_index[field][unit_type]:
                            results = results + self.paper_index[field][unit_type][key].most_similar(vec, topn)
                else:
                    for key in self.paper_index[field][u_type]:
                        results = results + self.paper_index[field][u_type][key].most_similar(vec, topn)
            else:
                if u_type == '-1':
                    for unit_type in unit_types:
                        results = results + self.paper_index[field][unit_type][province].most_similar(vec, topn)
                else:
                    results = self.paper_index[field][u_type][province].most_similar(vec, topn)
            return results
        except:
            return []

    def most_similar_patent(self, text, topn, field, u_type, province):
        try:
            results = []
            vec = self.t2v.text2v(text, self.cuttor)
            if province == '-1':
                if u_type == '-1':
                    for unit_type in unit_types:
                        for key in self.patent_index[field][unit_type]:
                            results = results + self.patent_index[field][unit_type][key].most_similar(vec, topn)
                else:
                    for key in self.patent_index[field][u_type]:
                        results = results + self.patent_index[field][u_type][key].most_similar(vec, topn)
            else:
                if u_type == '-1':
                    for unit_type in unit_types:
                        results = results + self.patent_index[field][unit_type][province].most_similar(vec, topn)
                else:
                    results = self.patent_index[field][u_type][province].most_similar(vec, topn)
            return results
        except:
            return []

    def most_similar_project(self, text, topn, field, u_type, province):
        try:
            results = []
            vec = self.t2v.text2v(text, self.cuttor)
            if province == '-1':
                if u_type == '-1':
                    for unit_type in unit_types_project:
                        for key in self.project_index[field][unit_type]:
                            results = results + self.project_index[field][unit_type][key].most_similar(vec, topn)
                else:
                    for key in self.project_index[field][u_type]:
                        results = results + self.project_index[field][u_type][key].most_similar(vec, topn)
            else:
                if u_type == '-1':
                    for unit_type in unit_types_project:
                        results = results + self.project_index[field][unit_type][province].most_similar(vec, topn)
                else:
                    results = self.project_index[field][u_type][province].most_similar(vec, topn)
            return sorted(dict(results).items(), key=lambda x: x[1], reverse=True)
        except:
            return []

    def getSimExpertsIds(self, topDocs):
        expertInfoOut = {}
        expertMap = {}
        authorSeqWeiht = [1.0, 0.85, 0.7, 0.5]
        db_1 = time.time()
        for typee in topDocs:
            order = {}
            order[typee] = {}
            k = 0
            for i, j in topDocs[typee]:
                order[typee][i] = k
                k = k + 1
            ids = [i for i, j in topDocs[typee]]

            try:
                docExpertIds = self.redis.getAuthors(ids)  # 使用Redis获取信息
                # if len(docExpertIds) == 0:
                #     logger.info(u'Redis失效，使用sql查询')
                #     docExpertIds = self.get_author_by_sql(typee, ids)
            except:
                logger.info(u'连接不上Redis，使用sql查询')
                docExpertIds = self.get_author_by_sql(typee, ids)
            for id in docExpertIds:
                # if not db.idInDB(typee, id):
                #     print "docId: " + id + " is not in db"
                #     continue
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
        db_2 = time.time()
        logger.info(u'time in 数据库 ' + str(db_2 - db_1))
        return expertMap, expertInfoOut

    # 从成果提取专家，有些专家在不过滤省份时排在前，但过滤省份后排在后，为避免此情况，先不过滤成果的地区，
    # 从这些不过滤地区的成果中提取专家，再按地区过滤专家，若不足topN，再在过滤地区的成果中找剩余的专家
    #
    # 这个函数需要重构，但是八成需求会改，所以先不重构了
    def most_similar_expert(self, topPapers, topPatents, topProjects, expertTopN):
        file = open("config.ini", 'r')
        config = ConfigParser.ConfigParser()
        config.readfp(file)
        LEN = int(config.get('global', 'len'))  # 对于一个专家要计算多少他的成果
        COE = float(config.get('global', 'coe'))  # 对于一个专家，从第二个的成果相似度乘的系数
        topDocs = {}

        topDocs['paper'] = topPapers
        topDocs['patent'] = topPatents
        topDocs['project'] = topProjects

        expertMap, expertInfoOut = self.getSimExpertsIds(topDocs)  # 专家id为key，各项成果的相似度list为value
        # expertMap, expertInfoOut = {}, {}  # 专家id为key，各项成果的相似度list为value
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
        # self.printOut(out,LEN)
        return result

    def get_author_by_sql(self, typee, ids):
        db = DB()
        return db.getAuthors(typee, ids)  # 使用MySQL获取信息

    # 仅根据论文得到专家，由上面的most_similar_expert函数复制修改来的，以后可以重构
    def most_similar_expert_paper(self, topPapers, filterParams, expertTopN):
        file = open("config.ini", 'r')
        config = ConfigParser.ConfigParser()
        config.readfp(file)
        LEN = int(config.get('global', 'len'))  # 对于一个专家要计算多少他的成果
        COE = float(config.get('global', 'coe'))  # 对于一个专家，从第二个的成果相似度乘的系数
        topDocs = {}
        topDocs['paper'] = self.filter('paper', topPapers, filterParams, 50)
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
        # self.printOut(out,LEN)
        return result

    # 仅根据专利得到专家，由上面的most_similar_expert函数复制修改来的，以后可以重构
    def most_similar_expert_patent(self, topPatents, filterParams, expertTopN):
        file = open("config.ini", 'r')
        config = ConfigParser.ConfigParser()
        config.readfp(file)
        LEN = int(config.get('global', 'len'))  # 对于一个专家要计算多少他的成果
        COE = float(config.get('global', 'coe'))  # 对于一个专家，从第二个的成果相似度乘的系数
        topDocs = {}
        topDocs['patent'] = self.filter('patent', topPatents, filterParams, 50)
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
        # self.printOut(out,LEN)
        return result

    # 仅根据项目得到专家，由上面的most_similar_expert函数复制修改来的，以后可以重构
    def most_similar_expert_project(self, topProjects, filterParams, expertTopN):
        file = open("config.ini", 'r')
        config = ConfigParser.ConfigParser()
        config.readfp(file)
        LEN = int(config.get('global', 'len'))  # 对于一个专家要计算多少他的成果
        COE = float(config.get('global', 'coe'))  # 对于一个专家，从第二个的成果相似度乘的系数
        topDocs = {}
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
        # self.printOut(out,LEN)
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
