# Luotuo Embedding: Generative Text Embedding Model distilled from OpenAI API

冷子昂 *,  Liu Siyi *, 陈启源 *,  蒟蒻 *, HS *, 641 *, 年 *, 李鲁鲁 *

#### 作者的footnote

作者的顺序是完全随机的，每个人的具体贡献如下:

李鲁鲁提议了整个项目，并完成了初始的实验验证，以及编写了最终的技术报告。

陈启源和HS建立了数据收集的架构，并完成了数据的收集

冷子昂 and 641 实现了训练的代码，并完成了基于GLM的text embedding训练

蒟蒻独立建立了另一套使用BERT等模型的训练代码。

年完成了实验的部分

孙骜完成了CoT的实验部分

## 摘要



## 引言

近年来，随着大型语言模型的发展，出现了以ChatGPT为首的很多新型应用。并且围绕着OpenAI开放的文本分类、文本Embedding和文本生成的几个接口，涌现了诸如NotionAI, Cursor, Copilot, ChatPDF等多种多样化的应用。

伴随着Meta开源LLaMA语言模型，并且结合对话型的指令调优(alpaca)，以及low rank adaptation的训练技术(alpaca-lora)。在最近的半年中，出现了大量的开源模型训练项目。他们往往可以利用较少的训练资源进行训练，并适配到多种不同的目标语言(葡萄牙、日本、中文)，以及垂直任务中(CoT)。

这些开源项目往往聚焦在文本生成，对于文本理解的处理，follow [GPT3的论文], 往往采用in-context-learning的形式。这是由于Decoder-based的语言模型，使用这种处理方式是更直接的，不需要额外的训练。不过，一方面在很多GPT3之后出现的应用中，很多开发者选择调用OpenAI的embedding接口，来进行语言理解的任务；另一方面，由于feature based的学习方法，能够使用更结构化的数据，并且使用更经典的机器学习架构，更便于传统文本理解研究者来使用。


TODO: 这里补充OpenAI的图在一起

<p align="center">
    <img src="image/CSEcompare.png" height="350">
</p>


+ embed_compare_fig1 , 根据 [OpenAIembedding论文] 的假设，在文本中间切开，前半部分和后半部分的文本embedding应该出现较强的相关性。在这个初步的实验中，我们调用了 [字节的数据集] 中的100条数据，并且将其从最接近中间的句号切开，分别在三个模型上测试了前半部分文档和后半部分文档的Embedding的余弦相似度。 可以发现，在OpenAI的API接口中，文本的相关性得到了有效的体现，即使前文和后文没有太多的重复词汇，也能够在对角线上体现很高的相关性。而BERT，由于采用的是文本dropout训练的方式，对于文本前后文的关联，刻画没有那么好的描写。而我们的模型实现了接近OpenAI API的效果。

对此，我们希望能够从现有的开源模型出发，获取一个较好的文本嵌入特征，来对支撑文本检索、文本分类等下游任务的识别。所有的代码包括测试代码、训练代码以及所有训练数据都会逐步清理和开源，供社区使用。我们希望我们的工作能够更好地推动语言模型，特别是中文语言模型的发展。

需要注意的是，从Decoder based的大语言模型中，得到一个Embedding特征，是非琐碎的。[OpenAIembedding论文]使用了[多少数据]规模的语料，在GPT3上进行训练，才获得了state-of-the-art级别的效果。但是无论是数据的部分还是GPT3的模型都是非开源的。这使得之后从Decoder模型中获取text embedding，只有少数的研究和公开代码[CSE那个代码]。本文意在利用较小的计算资源，从已经开源的一些中-英文模型出发，蒸馏得到一个较强的Embedding，以提供给下游更多的应用去使用。

本文的主要贡献是

+ 本文实现了一个通用的Text Embedding开源模型，可以有效支持更多的下游自然语言理解的任务。目前支持中文较长文本的Embedding，未来将进一步支持更多的跨语言任务。

+ 本文提出了一套蒸馏的同时进行自学习的方案，可以同步从一个较大的语言模型中，学习text的embedding。

+ 本文在CoT等实验中，验证了我们提出的Text Embedding的有效性。

## Embedding的特点和相关工作

作为一个self-contain的报告，我们在这里描述一下Text Embedding的基本目标和相关的工作。


+ Text Embedding的训练目标

这里chenqy补充一下？从基础n-gram的embedding，讲到bert，再讲到GPT3那个？

然后讲一下有监督和自学习的loss，我们这里主要是启用了自学习的loss。

图\ref{embed_compare_fig1}中可以看到，我们使用 [字节那个数据集] 中的一部分数据，验证了[openAI那个paper] 的假设。在OpenAI实际给出的API中，前半段的文本和后半段的文本有很强的相关性。因此在本文中，我们主要考虑与 [字节那个数据集] 相同的自监督loss。

+ Text Embedding的下游应用

<p align="center">
    <img src="image/tSNEchallenge.png" height="350">
</p>

A t-SNE visualization for proposed Embedding. Embedding有着多样化的下游应用，如数据可视化、聚类、搜索、分类等等。

考虑到有很多的监督应用，我们这里主要展示了搜索、文本可视化，以及一个AutoCoT的应用。更多的下游应用我们计划在之后的版本进行补充。


## 模型的训练

在本来[OpenAIembedding论文]的工作中，使用了[多少数据]规模的语料。而在我们的工作中，为了在更可支付的计算资源下得到模型，我们希望考虑在单卡A100的情况下，能够在7天级别内训练得到的模型。因此我们考虑了对OpenAI的[embedding模型名字]模型的输出进行蒸馏。本章节的组织如下，sec_loss_function 介绍了我们Loss的构造，sec_back_bone 介绍了我们使用的三种不同的模型大小。

TODO:在这张图左边增加两个size的bert，形成a,b,c

<p align="center">
    <img src="image/modelArch.jpg" height="350">
</p>

### 损失函数的构造

我们的损失函数由三项构成，Ldistill，LCSE。



+ Distill Loss

+ Weighted In-batch Contrastive Loss

这个loss基于标准的contrastive learning。每一个input instance是一对相似的文本sent_1, sent_2。然后在一个batch_size=N 的batch中，sent_1 与sent_2作为一个positive example，然后sent_1 与batch内的其他任意文本作为negative example，以此类推。最后的loss是加权过后的每一个sent_i与batch内其他文本的cross_entropy的总和。


### HardNegative Mining与KL散度Loss

在很多关于自学习的中提到[陈启源找一下文献]，仅仅使用简单的Contrastive Learning，对于负样本的学习太过于简单。在文本的Embedding训练中我们可以这样理解，如果两个文本中出现了一些完全topic完全不相关的单词，就很容易让两个文本去区分开。因此，[陈启源找一下文献]考虑在自相关学习中引入Hard Negative Mining的手段。

当然，有很多时候，多个文本语料可能是直接相关的，所以当我们把Hard Negative的样本放在一起的时候，很多Pesudo的Negative样本并不是真正的负样本，如果再使用之前提到的CSE损失函数学习，很容易出现Overfitting。

这个时候，结合OpenAI的输出，我们使用一个KL散度作为Loss，具体来说，在训练的时候，我们试图将Hard Negative尽可能抽样在一起。然后对于一个Batch，如果OpenAI得到的相似度矩阵是P，模型当前得到的矩阵是Q，对于这两个矩阵的第n行，p_n和q_n，我们求KL散度

KL(p_n* ||  q_n* ) = sum_i p_ni log(p_ni / q_ni )

当然，对于列我们也求KL散度，用KL(p_*n ||  q_*n )来表示。

最后，我们将每一列和每一行的KL散度求平均，得到了KL散度Loss

L_KL = 1/2n * sum_n ( KL(p_n* ||  q_n* )  + KL(p_*n ||  q_*n ) )

在第一个版本中，我们先没有使用这个KL散度的实现，我们将在后续放出的模型中使用这个Loss。


### BERT作为Backbone
大部分现有的embedding model都是使用Transformer Encoder architecture作为model的backbone（Bert，Roberta）。使用这种架构的好处在于bidirectional encoder能够掌握前后文的信息，从而更好的学习文字的表征。这篇工作同样使用BERT模型作为我们embedding model的backbone。任意一段文本被输入进BERT模型后，会先被BertTokenizer tokenize成input_ids，然后放入embedding层使它向量化，然后通过encoder学习他的表征，最后使用encoder最后一层的[CLS]的hidden states作为这段文本的表征。


### 利用ChatGLM-6b对embedding初始化

不同于之前的embedding model随机（或normal）初始化BERT的embedding layer，[OpenAIembedding论文]的工作提出了一种使用大语言模型初始化encoder的方法。Following their work，我们将input_ids输入进ChatGLM-6b并取他最后一层的hidden states作为我们encoder的input。换言而之，我们抛弃了原生的embedding层，并使用GLM去生成embedding，然后把GLM生成的embedding作为encoder的input去训练，并在训练过程中freeze了GLM的参数。此举旨在利用GLM大的参数量和parametric knowledge去初始化一个较为好的embedding。我们希望这个embedding可以一定程度的保留GLM的知识，并基于它去继续学习。







### 训练的细节

## 数据
CNewSum

## 结果
