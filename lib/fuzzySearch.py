import jieba
from typing import Optional
import openai
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import re
import ast
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from argparse import Namespace

class FuzzySearch():
    '''Class FuzzySearch
    Method
        __init__
            input:
                data: pandas.DataFrame
                    包含sentence和embed两列，其中sentence为string，embed为float列表
                openaiAPIKey: [optional] string
                    如果要使用openai embedding模型计算用户query embedding，则需要提供openai api key
                source: [optional] string in ["openai", "local"]
                    如果使用openai，将调用openai embedding api计算用户query的embedding，如果使用local，则调用model指定的模型计算用户query的embedding
                model: [optional] string
                    指定用于计算用户query embedding的模型，默认为"silk-road/luotuo-bert"
                stop_words_path: [optional] string
                    指定stop words
        search
            input: 
                openaiAPIKey: [optional] string
                    如果要使用openai embedding模型计算用户query embedding，则需要提供openai api key
                source: [optional] string in ["openai", "local"]
                    如果使用openai，将调用openai embedding api计算用户query的embedding，如果使用local，则调用model指定的模型计算用户query的embedding

        '''
    def __init__(self, data, openaiAPIKey = None, source = "local", model = "silk-road/luotuo-bert", stop_words_path = "../data/stop_word.txt"):
        
        self.data = data
        self.openaiAPIKey = openaiAPIKey
        self.source = source
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model_args = Namespace(do_mlm=None, pooler_type="cls", temp=0.05, mlp_only_train=False, init_embeddings_model=None)
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True, model_args=self.model_args)
        self.embed = np.array(data["embed"].tolist())
        with open(stop_words_path, "r", encoding="utf-8") as f:
            self.stop_words = f.read().splitlines()

    def text_to_embed(self, text: str):
        if self.source is not None and self.source not in ["openai", "local"]:
            raise ValueError("source只能是'openai'或'local'")
        if self.source == "openai":
            if self.openaiAPIKey is None:
                raise ValueError("openaiAPIKey不能为空")
            openai.api_key = self.openaiAPIKey
            response = openai.Embedding.create(
                input=text,
                engine="text-embedding-ada-002")
            return response["data"][0]["embedding"]
        elif self.source == "local":
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            return self.model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output.detach().numpy()[0] 

    # 创建函数获取搜索结果
    def get_search_results(self, query):
        
        user_embed = self.text_to_embed(query)
        user_embed = np.array(user_embed).astype(float)

        # use cosine similarity to find the most similar sentence from test_X
        cosine_similarity = np.dot(self.embed, user_embed) / (np.linalg.norm(self.embed, axis=1) * np.linalg.norm(user_embed))
        cosine_similarity = cosine_similarity.reshape(-1, 1)

        # find the five most similar sentence
        top5 = np.argsort(cosine_similarity, axis=0)[-5:]
        top5 = top5.reshape(-1, 1)
        top5 = top5[::-1]

        
        result = []
        for i in top5:
            result.append(self.data["sentence"][i[0]])
        return result

        

    # 分词
    def tokenize(self, text):
        text = jieba.cut_for_search(text)
        return text

    # 移除停用词
    def remove_stop_words(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    # 高亮文本
    def highlight_text(self, text, words):
        for word in words:
            text = text.replace(word, f"<mark>{word}</mark>")
        return text

    # 主程序
    def highlighted_result(self, query):

        # 分词
        tokens = self.tokenize(query)
        # 移除停用词
        filtered_tokens = self.remove_stop_words(tokens)
        # 获取搜索结果
        search_results = self.get_search_results(query)
        # 对比并高亮
        highlighted_results = [self.highlight_text(result, filtered_tokens).replace(r"</mark><mark>", "") for result in search_results]

        for i in range(len(highlighted_results)):
            print(f"第{i+1}个结果：\n{highlighted_results[i]}\n")

        return highlighted_results

    def adjust_boundaries(self, text, start, end):
        while start > 0 and text[start:start+6] != '<mark>':
            start -= 1

        while end < len(text) and text[end-7:end] != '</mark>':
            end += 1

        return start, end

    def adjust_end(self, text, start, end):
        pattern = r'<mark>'
        # ther could be more than one match, so we need to find the last one
        n = 0

        for mat in re.finditer(pattern, text[start:end + 7]):
            n += 1
            if n == 2:
                return mat.end() + start - 6
    
        return end

    def get_highlighted_substr(self, text, num_chars=20):
        pattern = r'<mark>([^<]+)</mark>'
        highlighted_substrings = []

        last_end = 0
        
        for match in re.finditer(pattern, text):
            sep = ' ... '
            if match:
                start = match.start(1)
                end = match.end(1)
                # Adjust start and end index to include full <mark> tags if cut off
                start, end = self.adjust_boundaries(text, start, end)

                start -= num_chars
                end += num_chars
                start = max(0, start)
                end = min(len(text), end)
                # Adjust start index to avoid overlapping substrings
                if start < last_end:
                    start = last_end
                    sep = ''
                # Adjust end index to include full </mark> tag

                end = self.adjust_end(text, start, end)

                highlighted_substrings.append(sep + text[start:end])
                last_end = end

        return ''.join(highlighted_substrings).strip(" ... ")


    def show_highlightes(self, highlight):
        #只显示高亮前后的文本
        highlight_print = [''] * len(highlight)
        for i in range(len(highlight)):
            highlight_print[i] = self.get_highlighted_substr(highlight[i], 30)
            if highlight_print[i] != '':
                display(HTML(highlight_print[i]))
    

    def search(self, source="local", openaiAPIKey=None):
        self.source = source
        if openaiAPIKey is not None:
            self.openaiAPIKey = openaiAPIKey
        user_input = input(prompt = "请输入您想搜索的问题，例如 '虎扑报道马刺的保罗-加索尔与球队正式签订协议，有哪些相关的新闻？'")
        highlight = self.highlighted_result(user_input)
        self.show_highlightes(highlight)