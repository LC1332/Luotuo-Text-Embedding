# Luotuo Embedding(骆驼嵌入): Generative Text Embedding Model distilled from OpenAI API

Luotuo Embedding(骆驼嵌入) is a text embedding model, which developed by 李鲁鲁, 冷子昂, 陈启源, 蒟蒻, HS, 641, 年等.



## A Quick Start

+ Evaluation Code

+ Training Code

## Trained Models


## Contributors

The author order is in random, we detailly record the contribution here.

李鲁鲁 proposed the project, and do the starting effort on the preliminary experiment of embedding and designed the test and visualization.

陈启源 and HS implemented the data collecting server and collect all data.

冷子昂 and 641 developed the training framework and trained the GLM based Embedding model.

蒟蒻 developed an individual training framework and trained BERT based Embedding model.

## TODO

Embedding的测试需求

+ query-Answer

- 必须包括 周鸿祎 在360 ChatGPT展示大会的语料内容（ 这部分我可以让我老婆去看一下那个大会）

- 然后就是正常的新闻语料，找个10000个的base就可以

- 例子尽量使用周鸿祎展示的例子

- 额外再给一个别的例子就可以

+ 用户自由输入，进行文档retrieve

这里可以录一个视频

+ 聚类+词云展示

我的colab里面已经有基本的例子

把我的词云加上stop words过滤

+ t-SNE展示

base语料里面，找3个高频词。

相关的文章形成3个类，展示特征的t-SNE

这里鲁叔有个特别的t-SNE设计

+ 下游分类任务展示

展示10句A类语料，10句B类语料

然后给定新的句子，判断A类还是B类

+ openAI原假设验证

query和base画对角线热图 


