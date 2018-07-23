#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tornado.ioloop
import tornado.web
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import tornado.httpserver
import logging.config
import utils
import json
import time

logging.config.fileConfig('logger.conf')
logger = logging.getLogger('recommServerLog')
logger.info('系统启动...')

import similarity

if utils.get_host_ip() == '10.1.13.49':
    recmder = similarity.Recommander('/home/tdlab/recommender/data_filter/wm.bin',
                                     '/home/tdlab/recommender/data_filter/ind/paper/',
                                     '/home/tdlab/recommender/data_filter/ind/patent/',
                                     '/home/tdlab/recommender/data_filter/ind/project/')
else:
    recmder = similarity.Recommander('/data/Recommender/data_filter/wm.bin',
                                     '/data/Recommender/data_filter/ind/paper/',
                                     '/data/Recommender/data_filter/ind/patent/',
                                     '/data/Recommender/data_filter/ind/project/')
TOPN = 65  # 先取大量数据，在这数据上再做筛选，该TOPN并不是返回的数量

# /recommend/all.do
class AllHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(cpu_count())

    @run_on_executor
    def get(self):
        begin = time.time()
        txt = self.get_query_argument('words')
        expertTopN = int(self.get_query_argument('expertTopN'))
        docTopN = int(self.get_query_argument('docTopN'))
        unit_type = self.get_query_argument('type')
        province = self.get_query_argument('province')
        field = self.get_query_argument('field')
        unit = self.get_query_argument('unit')
        logger.info(u'#####收到请求，参数：txt=%s,field=%s,type=%s,province=%s,unit=%s' % (
            txt, field, unit_type, province, unit))
        ann_1 = time.time()
        topPapers = recmder.field_deliver('paper', txt, TOPN, field, unit_type, province)
        ann_2 = time.time()
        topPatents = recmder.field_deliver('patent', txt, TOPN, field, unit_type, province)
        ann_3 = time.time()
        topProjects = recmder.most_similar_project(txt, TOPN, 'Z9', unit_type, province)
        ann_4 = time.time()
        logger.info(u'time in ann ' + str(ann_4 - ann_1))
        logger.info(u'time in paper ann ' + str(ann_2 - ann_1))
        logger.info(u'time in patent ann ' + str(ann_3 - ann_2))
        logger.info(u'time in project ann ' + str(ann_4 - ann_3))
        if unit != '-1':
            filterParams = ['-1', '-1', '-1', unit, '-1']
            topPapers = recmder.filter('paper', topPapers, filterParams, docTopN)
            topPatents = recmder.filter('patent', topPatents, filterParams, docTopN)
            topProjects = recmder.filter('project', topProjects, filterParams, docTopN)
            filter_time = time.time()
            logger.info(u'time in filter' + str(filter_time - ann_4))
        expert_1 = time.time()
        experts = recmder.most_similar_expert(topPapers[0:50], topPatents[0:50], topProjects[0:15], expertTopN)
        expert_2 = time.time()
        logger.info(u'time in 拼人' + str(expert_2 - expert_1))
        result = {}
        result["papers"] = [i for i, j in topPapers[0:docTopN]]
        result["patents"] = [i for i, j in topPatents[0:docTopN]]
        result["projects"] = [i for i, j in topProjects[0:docTopN]]
        result["experts"] = [i for i, j in experts]
        end = time.time()
        logger.info(u'time in total ' + str(end - begin))
        self.write(json.dumps(result))


# /recommend/paper.do
class PaperHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(cpu_count())

    @run_on_executor
    def get(self):
        n = int(self.get_query_argument('docTopN'))
        txt = self.get_query_argument('words')
        logger.info(u'#####收到请求，参数：%s,%s' % (n, txt))
        unit_type = self.get_query_argument('type')
        province = self.get_query_argument('province')
        field = self.get_query_argument('field')
        unit = self.get_query_argument('unit')
        journalQuality = self.get_query_argument('journalQuality')
        TOPN = 20
        try:
            ann_1 = time.time()
            l = recmder.field_deliver('paper', txt, TOPN, field, unit_type, province)
            ann_2 = time.time()
            logger.info(u'time in paper ann ' + str(ann_2 - ann_1))
        except:
            l = []
        filterParams = ['-1', '-1', '-1', unit, journalQuality]
        begin = time.time()
        l = recmder.filter('paper', l, filterParams, n)
        end = time.time()
        logger.info(u'time in filter ' + str(end - begin))
        self.write(utils.l2m_str(l))


# /recommend/patent.do
class PatentHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(cpu_count())

    @run_on_executor
    def get(self):
        n = int(self.get_query_argument('docTopN'))
        txt = self.get_query_argument('words')
        logger.info(u'#####收到请求，参数：%s,%s' % (n, txt))
        unit_type = self.get_query_argument('type')
        province = self.get_query_argument('province')
        field = self.get_query_argument('field')
        unit = self.get_query_argument('unit')
        patentType = self.get_query_argument('patentType')
        TOPN = 20
        try:
            ann_1 = time.time()
            l = recmder.field_deliver('patent', txt, TOPN, field, unit_type, province)
            ann_2 = time.time()
            logger.info(u'time in patent ann ' + str(ann_2 - ann_1))
        except:
            l = []
        filterParams = ['-1', '-1', '-1', unit, patentType]
        for i in range(len(filterParams)):
            if filterParams[i] == '-1':
                filterParams[i] = ''
        begin = time.time()
        l = recmder.filter('patent', l, filterParams, n)
        end = time.time()
        logger.info(u'time in filter' + str(end - begin))
        self.write(utils.l2m_str(l))


# /recommend/project.do
class ProjectHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(cpu_count())

    @run_on_executor
    def get(self):
        n = int(self.get_query_argument('docTopN'))
        txt = self.get_query_argument('words')
        unit_type = self.get_query_argument('type')
        province = self.get_query_argument('province')
        unit = self.get_query_argument('unit')
        projectType = self.get_query_argument('projectType')
        TOPN = 15
        try:
            ann_1 = time.time()
            l = recmder.most_similar_project(txt, TOPN, 'Z9', unit_type, province)
            ann_2 = time.time()
            logger.info(u'time in project ann ' + str(ann_2 - ann_1))
        except:
            l = []
        filterParams = ['Z9', '-1', '-1', unit, projectType]
        for i in range(len(filterParams)):
            if filterParams[i] == '-1':
                filterParams[i] = ''
        begin = time.time()
        l = recmder.filter('project', l, filterParams, n)
        end = time.time()
        logger.info(u'time in filter' + str(end - begin))
        self.write(utils.l2m_str(l))


# /recommend/paperAndExpert.do
class PaperAndExpertHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(cpu_count())

    @run_on_executor
    def get(self):
        txt = self.get_query_argument('words')
        expertTopN = int(self.get_query_argument('expertTopN'))
        docTopN = int(self.get_query_argument('docTopN'))
        unit_type = self.get_query_argument('type')
        province = self.get_query_argument('province')
        field = self.get_query_argument('field')
        filterParams = []
        filterParams.append(field)
        filterParams.append(unit_type)
        filterParams.append(province)
        filterParams.append(self.get_query_argument('unit'))
        filterParams.append('-1')
        logger.info(u'#####收到请求，参数：txt=%s,field=%s,type=%s,province=%s,unit=%s' % (
            txt, filterParams[0], filterParams[1], filterParams[2], filterParams[3]))
        topPapers = recmder.field_deliver('paper', txt, TOPN, field, unit_type, province)
        # filteredPapers = recmder.filter('paper', topPapers, filterParams, docTopN)
        experts = recmder.most_similar_expert_paper(topPapers[0:80], filterParams, expertTopN)
        result = {}
        result["papers"] = [i for i, j in topPapers[0:docTopN]]
        result["patents"] = []
        result["projects"] = []
        result["experts"] = [i for i, j in experts]
        self.write(json.dumps(result))


# /recommend/patentAndExpert.do
class PatentAndExpertHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(cpu_count())

    @run_on_executor
    def get(self):
        txt = self.get_query_argument('words')
        expertTopN = int(self.get_query_argument('expertTopN'))
        docTopN = int(self.get_query_argument('docTopN'))
        unit_type = self.get_query_argument('type')
        province = self.get_query_argument('province')
        field = self.get_query_argument('field')
        filterParams = []
        filterParams.append(field)
        filterParams.append(unit_type)
        filterParams.append(province)
        filterParams.append(self.get_query_argument('unit'))
        filterParams.append('-1')
        logger.info(u'#####收到请求，参数：txt=%s,field=%s,type=%s,province=%s,unit=%s' % (
            txt, filterParams[0], filterParams[1], filterParams[2], filterParams[3]))
        topPatents = recmder.field_deliver('patent', txt, TOPN, field, unit_type, province)
        # filteredPatents = recmder.filter('patent', topPatents, filterParams, docTopN)
        experts = recmder.most_similar_expert_patent(topPatents[0:80], filterParams, expertTopN)
        result = {}
        result["papers"] = []
        result["patents"] = [i for i, j in topPatents[0:docTopN]]
        result["projects"] = []
        result["experts"] = [i for i, j in experts]
        self.write(json.dumps(result))


# /recommend/projectAndExpert.do
class ProjectAndExpertHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(cpu_count())

    @run_on_executor
    def get(self):
        txt = self.get_query_argument('words')
        expertTopN = int(self.get_query_argument('expertTopN'))
        docTopN = int(self.get_query_argument('docTopN'))
        unit_type = self.get_query_argument('type')
        province = self.get_query_argument('province')
        field = self.get_query_argument('field')
        filterParams = []
        filterParams.append(field)
        filterParams.append(unit_type)
        filterParams.append(province)
        filterParams.append(self.get_query_argument('unit'))
        filterParams.append('-1')
        logger.info(u'#####收到请求，参数：txt=%s,field=%s,type=%s,province=%s,unit=%s' % (
            txt, filterParams[0], filterParams[1], filterParams[2], filterParams[3]))
        topProjects = recmder.most_similar_project(txt, TOPN, 'Z9', unit_type, province)
        # filteredProjects = recmder.filter('project', topProjects, filterParams, docTopN)
        experts = recmder.most_similar_expert_project(topProjects[0:80], filterParams, expertTopN)
        result = {}
        result["papers"] = []
        result["patents"] = []
        result["projects"] = [i for i, j in topProjects[0:docTopN]]
        result["experts"] = [i for i, j in experts]
        self.write(json.dumps(result))


# /recommend/cut.do
class CutHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(cpu_count())

    @run_on_executor
    def get(self):
        txt = self.get_query_argument('txt')
        tokens = recmder.get_cuttor().fltcut(txt)
        logger.info(u'收到请求,文本：' + txt)
        self.write(json.dumps(tokens))


# /recommend/expertDocsSort.do
class ExpertDocsHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(cpu_count())

    @run_on_executor
    def get(self):
        txt = self.get_query_argument('demandTxt')
        expertsIds = self.get_query_argument('experts').strip().split(',')
        topN = int(self.get_query_argument('topN'))
        result = {}
        for expertId in expertsIds:
            r = recmder.expertDocsSort(expertId, txt, topN)
            result[expertId] = r
        self.write(json.dumps(result))


# /recommend/is_contain.do
class ContainHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(cpu_count())

    @run_on_executor
    def get(self):
        w = self.get_query_argument('w')
        wm = recmder.get_model()
        self.write(json.dumps(w in wm))


# /recommend/topnwords.do
class TopWordHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(cpu_count())

    @run_on_executor
    def get(self):
        try:
            n = int(self.get_query_argument('n'))
            w = self.get_query_argument('w')
            logger.info(u'#####收到请求，参数：%s,%s' % (n, w))
            r = recmder.get_model().most_similar(w, topn=n)
            l = [w for w, s in r]
        except:
            l = []
        self.write(json.dumps(l))


def make_app():
    return tornado.web.Application([
        tornado.web.url(r"/recommend/all.do", AllHandler),
        tornado.web.url(r"/recommend/paper.do", PaperHandler),
        tornado.web.url(r"/recommend/patent.do", PatentHandler),
        tornado.web.url(r"/recommend/project.do", ProjectHandler),
        tornado.web.url(r"/recommend/paperAndExpert.do", PaperAndExpertHandler),
        tornado.web.url(r"/recommend/patentAndExpert.do", PatentAndExpertHandler),
        tornado.web.url(r"/recommend/projectAndExpert.do", ProjectAndExpertHandler),
        tornado.web.url(r"/recommend/expertDocsSort.do", ExpertDocsHandler),
        tornado.web.url(r"/recommend/cut.do", CutHandler),
        tornado.web.url(r"/analysis/is_contain.do", ContainHandler),
        tornado.web.url(r"/analysis/topnwords.do", TopWordHandler),
    ])


if __name__ == '__main__':
    logger.info(u'Number of threads: %s' % cpu_count())
    app = make_app()
    app.listen(8640)
    logger.info(u'Server run on port 8640')
    tornado.ioloop.IOLoop.current().start()
