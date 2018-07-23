#!/usr/bin/env python
# -*- coding: utf-8 -*-

import MySQLdb


class DB(object):
    def __init__(self):
        print "create connection!!!!!!"
        self.conn = MySQLdb.connect(host='10.1.13.29', user='root', passwd='tdlabDatabase', db='techpooldata',
                                    port=3306, charset='utf8')
        self.tables = {'paper': 'expert_paper_join', 'patent': 'expert_patent_join', 'project': 'expert_project_join'}
        self.columns = {'paper': 'PAPER_ID', 'patent': 'PATENT_ID', 'project': 'PROJECT_ID'}

    def getConnection(self):
        try:
            self.conn.ping()
        except:
            self.conn = MySQLdb.connect(host='10.1.13.29', user='root', passwd='tdlabDatabase', db='techpooldata',
                                        port=3306, charset='utf8')
            print 'reconnection'
        return self.conn

    # 返回以docId为key，以authors为value的map
    def getAuthors(self, typee, ids):
        if len(ids) ==0:
            return {}
        authorIds = {}
        sql = "select " + self.columns[typee] + ",EXPERT_ID from " + self.tables[typee] + " where " + self.columns[
            typee] + " in("
        for i in range(len(ids)):
            sql = sql + "'" + ids[i] + "'"
            if i != len(ids) - 1:
                sql = sql + ','
        sql = sql + ') order by ' + self.columns[typee] + ',expert_role'
        cur = self.getConnection().cursor()
        cur.execute(sql)
        results = cur.fetchall()
        cur.close()
        for line in results:
            if line[0] not in authorIds:
                authorIds[line[0]] = []
            authorIds[line[0]].append(line[1])
        return authorIds

    '''
   def getAuthors(self, typee, ids):
       cur=self.conn.cursor()
       authorIds = {}
       for id in ids:
           authorIds[id]=[]
           sql = "select EXPERT_ID, expert_role from "+ self.tables[typee] +" where "+ self.columns[typee]+ " ='"+id+"' order by expert_role"
           cur.execute(sql)
           results = cur.fetchall()
           for i in results:
               authorIds[id].append(i[0])
       cur.close()
       return authorIds
   '''

    # 返回id是否在数据库中，因为有时数据不同步
    def idInDB(self, typee, id):
        cur = self.getConnection().cursor()
        sql = "select * from " + self.tables[typee] + " where " + self.columns[typee] + " ='" + id + "'"
        count = cur.execute(sql)
        cur.close()
        return count > 0

    def __del__(self):
        self.conn.close()
