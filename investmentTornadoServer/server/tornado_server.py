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
                                     '/home/tdlab/investment_recomender/data_filter/ind/paper/',
                                     '/home/tdlab/investment_recomender/data_filter/ind/patent/',
                                     '/home/tdlab/investment_recomender/data_filter/ind/project/')
else:
    recmder = similarity.Recommander('/data/Recommender/data_filter/wm.bin',
                                     '/data/Recommender/data_filter/ind/paper/',
                                     '/data/Recommender/data_filter/ind/patent/',
                                     '/data/Recommender/data_filter/ind/project/')

TOPN = 10000
class AllHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(cpu_count())

    @run_on_executor
    def get(self):
        # get request argument
        txt = self.get_query_argument('words')
        unit_type = self.get_query_argument('type')
        province = self.get_query_argument('province')
        field = self.get_query_argument('field')
        unit = self.get_query_argument('unit')
        docTopN = int(self.get_query_argument('docTopN'))
        expertTopN = int(self.get_query_argument('expertTopN'))
        logger.info(u'#####收到请求，参数：txt=%s,field=%s,type=%s,province=%s,unit=%s' % (
            txt, field, unit_type, province, unit))

        # get most similar result
        topPapers = recmder.field_deliver_investment('paper', txt, TOPN)
        topProjects = recmder.field_deliver_investment('project', txt, TOPN)
        topPatents = recmder.field_deliver_investment('patent', txt, TOPN)

        # filter result for unit，provice,unit_type,filed
        if unit != '-1' or province != '-1' or unit_type != '-1' or field != '-1':
            filterParams = [province, unit_type, field, unit]
            topPapers = recmder.filter_investment('paper', topPapers, filterParams, docTopN)
            topPatents = recmder.filter_investment('patent', topPatents, filterParams, docTopN)
            topProjects = recmder.filter_investment('project', topProjects, filterParams, docTopN)

        # get most similar by paper,patent,project
        experts = recmder.most_similar_expert(topPapers[0:50], topPatents[0:50], topProjects[0:15], expertTopN)

        result = {}
        result["papers"] = [i for i, j in topPapers[0:20]]
        result["patents"] = [i for i, j in topPatents[0:20]]
        result["projects"] = [i for i, j in topProjects[0:20]]
        result["experts"] = [i for i, j in experts]
        self.write(json.dumps(result))

def make_app():
    return tornado.web.Application([
        tornado.web.url(r"/recommend/all.do", AllHandler),
    ])

if __name__ == '__main__':
    print u'Number of threads: %s' % cpu_count()
    app = make_app()
    app.listen(8648)
    print u'Server run on port 8648'
    tornado.ioloop.IOLoop.current().start()
