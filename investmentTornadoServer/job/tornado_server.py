#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tornado.ioloop
import tornado.web
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import tornado.httpserver



class TestHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(cpu_count())

    @run_on_executor
    def get(self):
        words = self.get_query_argument('words')
        self.write(words)

def make_app():
    return tornado.web.Application([
        tornado.web.url(r"/recommend/test.do", TestHandler),
    ])

if __name__ == '__main__':
    print u'Number of threads: %s' % cpu_count()
    app = make_app()
    app.listen(8640)
    print u'Server run on port 8640'
    tornado.ioloop.IOLoop.current().start()
