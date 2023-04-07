# Luotuo Embedding: Generative Text Embedding Model distilled from OpenAI API

Liu Siyi *, 冷子昂 *, 陈启源 *, 蒟蒻 *, HS *, 641 *, 年 *, 李鲁鲁 *

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

然而，为了处理

## 训练

## 数据

## 结果