# TonrnadoRecomendServer
基于python实现的一个推荐功能，主要是gensim词向量训练、AnnoyIndex实现快速查找向量间的距离。tornado实现服务化。

如何实现推荐功能整体的技术栈以及思路整理到下面的博客中：[如何做一个推荐系统](https://www.callmejiagu.com/2018/07/21/%E5%A6%82%E4%BD%95%E5%81%9A%E4%B8%80%E4%B8%AA%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F/)

investmentTornadoServer/job/jobs.py 主要包含词向量、文档向量化、生成索引等步骤，这几步推荐系统是预处理过程
investmentTornadoServer/server/server.py 提供推荐功能
