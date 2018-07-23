#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Wu Yuanchao <151050012@hdu.edu.cn>

from mycut import FilterCut
import corpora
import similarity
from gensim.similarities.index import AnnoyIndexer
import gensim
import codecs
import os
import logging
logger = logging.getLogger()


raws = ['/home/tdlab/investment_recomender/data_filter/corpora/paper/', '/home/tdlab/investment_recomender/data_filter/corpora/patent/',
        '/home/tdlab/investment_recomender/data_filter/corpora/project/']
cuts = ['/home/tdlab/investment_recomender/data_filter/cut/paper', '/home/tdlab/investment_recomender/data_filter/cut/patent/',
        '/home/tdlab/investment_recomender/data_filter/cut/project/']
vecs = ['/home/tdlab/investment_recomender/data_filter/vec_new/paper/', '/home/tdlab/investment_recomender/data_filter/vec_new/patent/',
        '/home/tdlab/investment_recomender/data_filter/vec_new/project/']
indx = ['/home/tdlab/recommender/data_filter/ind/paper/', '/home/tdlab/recommender/data_filter/ind/patent/',
        '/home/tdlab/recommender/data_filter/ind/project/']
wm_file = '/home/tdlab/recommender/data_filter/wm.bin'
basic_dir = '/home/tdlab/recommender/data_filter/dict/basic'
userdict_file = '/home/tdlab/recommender/data_filter/dict/userdict.txt'


def build_dictionary():
    basicset = corpora.load_words(basic_dir)
    print('basic loaded.')
    # stoplist = corpora.load_words(stop_dir)
    # print('stop list loaded.')
    # finalset = basicset - stoplist
    finalset = basicset
    with codecs.open(userdict_file, 'w', 'utf-8') as f:
        for w in finalset:
            f.write(w + os.linesep)


def cut():
    from mycut import FilterCut
    cuttor = FilterCut()
    # for r, c in zip(raws, cuts):
    #     corpora.process_rawcorpora(r, c, cuttor)
    for raw in raws:
        # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for parent, dirnames, filenames in os.walk(raw):
            for filename in filenames:  # 输出文件信息
                origin_path = os.path.join(parent, filename)
                target_path = os.path.join(parent, filename).replace('corpora', 'cut').split('.')[0] + '.cut'
                logger.info('origin path:' + origin_path)
                logger.info('target path:' + target_path)
                if not os.path.exists(parent.replace('corpora', 'cut')):
                    os.makedirs(parent.replace('corpora', 'cut'))
                corpora.process_rawcorpora(origin_path, target_path, cuttor)


def build_model():
    paper_cuted = cuts[0]
    sentences = corpora.CorporaWithoutTitle(paper_cuted)
    # sentences-语料库 size-维度 window-词共现窗口长度 sg-skipgram cbow二选一 min_count-最小词频 workers-线程
    model = gensim.models.word2vec.Word2Vec(sentences, size=200, window=4, sg=1, min_count=0, workers=4)
    model.save_word2vec_format(wm_file, binary=True)


def gen_vec():
    wm = gensim.models.word2vec.Word2Vec.load_word2vec_format(wm_file, binary=True)
    # for c, v in zip(cuts, vecs):
    #     similarity.saveVecs(c, v, wm)
    for cut in cuts:
        # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for parent, dirnames, filenames in os.walk(cut):
            for filename in filenames:  # 输出文件信息
                origin_path = os.path.join(parent, filename)
                target_path = os.path.join(parent, filename).replace('cut', 'vec_new').split('.')[0] + '.vec'
                logger.info('origin path:' + origin_path)
                logger.info('target path:' + target_path)
                if not os.path.exists(parent.replace('cut', 'vec_new')):
                    os.makedirs(parent.replace('cut', 'vec_new'))
                similarity.saveVecs(origin_path, target_path, wm)


def build_index():
    # for v, i in zip(vecs, indx):
    #     model = gensim.models.word2vec.Word2Vec.load_word2vec_format(v, binary=False)
    #     index = AnnoyIndexer(model, 100)
    #     index.save(i)
    for vec in vecs:
        # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for parent, dirnames, filenames in os.walk(vec):
            for filename in filenames:  # 输出文件信息
                origin_path = os.path.join(parent, filename)
                target_path = os.path.join(parent, filename).replace('vec_new', 'ind').split('.')[0] + '.ind'
                logger.info('origin path:' + origin_path)
                logger.info('target path:' + target_path)
                if not os.path.exists(parent.replace('vec_new', 'ind')):
                    os.makedirs(parent.replace('vec_new', 'ind'))
                model = gensim.models.word2vec.Word2Vec.load_word2vec_format(origin_path, binary=False)
                index = AnnoyIndexer(model, 100)
                index.save(target_path)


if __name__ == '__main__':
    # 构建词袋
    # build_dictionary()
    # 分词
    # cut()
    # 训练词向量
    # build_model()
    # 文档向量化
    # gen_vec()
    # 生成索引
    build_index()
