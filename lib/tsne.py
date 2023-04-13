import re
import openai
from IPython.display import Markdown, display
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from openTSNE import TSNE
from datasets import load_dataset

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import plotly.graph_objs as go

from matplotlib.lines import Line2D

from scipy.spatial.distance import pdist, squareform
from networkx import Graph
from networkx.algorithms.matching import max_weight_matching
import networkx as nx

class TSNE_Plot():
    def __init__(self, sentence, embed, label = None, N = 3):
        self.N = N
        self.test_X = pd.DataFrame({"text": sentence, "embed": [np.array(i) for i in embed]})
        self.test_y = pd.DataFrame({'label':label}) if label is not None else pd.DataFrame({"label": self.cluster()})
        
    
    def cluster(self):
        from sklearn.cluster import KMeans
        n_components = min(50, len(self.test_X))
        pca = PCA(n_components=n_components)
        compact_embedding = pca.fit_transform(np.array(self.test_X["embed"].tolist()))
        kmeans = KMeans(n_clusters=self.N)
        kmeans.fit(compact_embedding)
        labels = kmeans.labels_
        return labels
    
    def generate_colormap(self, n_labels):
        #创建一个均匀分布的颜色映射
        color_norm = mcolors.Normalize(vmin=0, vmax=len(n_labels) - 1)
        # 使用 plt.cm 中预先定义的colormap，你可以自由选择其他colormap如"hsv", "hot", "cool", "viridis"等
        scalar_map = plt.cm.ScalarMappable(norm=color_norm, cmap='jet') 

        colormap = {}
        for label in range(len(n_labels)):
            # 将颜色值转换为十六进制
            color_hex = mcolors.to_hex(scalar_map.to_rgba(label))
            colormap[n_labels[label]] = color_hex
        return colormap

    def divide_hex_color_by_half(self, hex_color):
        if len(hex_color) > 0 and hex_color[0] == "#":
            hex_color = hex_color[1:]

        red_hex, green_hex, blue_hex = hex_color[0:2], hex_color[2:4], hex_color[4:6]
        
        red_half = int(red_hex, 16) // 10 + (255-25)
        green_half = int(green_hex, 16) // 10 + (255-25)
        blue_half = int(blue_hex, 16) // 10 + (255-25)
        
        half_hex_color = "#{:02x}{:02x}{:02x}".format(red_half, green_half, blue_half)
        return half_hex_color

    def calculate_distance_matrix(self, df):
        coord = np.array(df[["x", "y", "label"]])
        return squareform(pdist(coord, "euclidean"))


    def create_weighted_graph(self, distance_matrix):
        size = distance_matrix.shape[0]
        G = Graph()

        for i in range(0, size, 2):
            for j in range(1, size, 2):
                if distance_matrix[i][j] > 0:
                    G.add_edge(i, j, weight=distance_matrix[i][j])

        return G

    def show_text(self, show_sentence, text):
        sentence = []
        for i in range(len(text)):
            if i in show_sentence:
                s = text[i][:10] + "..." + text[i][-10:]
                sentence.append(s)
            else:
                sentence.append("")
        return sentence

    def format_data(self, show_sentence, embd_data, labels):
        sentence = self.show_text(show_sentence, self.test_X["text"])
        X, Y = np.split(embd_data, 2, axis=1)
        n = len(self.test_X)
        # initialize the sentence position to be left, but make sure the sentence in show_sentence is distributed on both right and left equally    
        sentence_pos = ["left" for i in range(n)]
        for i in range(len(show_sentence)):
            if i % 2 == 0:
                sentence_pos[show_sentence[i]] = "right"

        data = {
            "x": X.flatten(),
            "y": Y.flatten(),
            "label": labels,
            "sentence" : sentence,
            "sentence_pos" : sentence_pos,
            "size" : [20 if i in show_sentence else 10 for i in range(n)]
        }
        df = pd.DataFrame(data)
        return df

    def calculate_tsne(self):   
        embed = np.array(self.test_X["embed"].tolist())
        n_components = min(50, len(self.test_X))
        pca = PCA(n_components=n_components)
        compact_embedding = pca.fit_transform(embed)
        tsne = TSNE(
            perplexity=30,
            metric="cosine",
            n_jobs=8,
            random_state=42,
            verbose=True,
        )
        embedding_train = tsne.fit(compact_embedding)
        embedding_train = embedding_train.optimize(n_iter=1000, momentum=0.8)
        return embedding_train

    def random_sentence(self, N):
        #多次随机可能会影响可视化结果
        n = len(self.test_y)
        show_sentence = [np.random.randint(0, n) for i in range(N)]

        # 确保每个标签至少有一个句子，用在show_sentence中最多的标签的句子来补充
        label_count = self.test_y["label"].value_counts()
        max_label = label_count.index[0]
        max_count = label_count[0]
        for i in range(max_count):
            for j in range(len(label_count)):
                if label_count[j] == i:
                    show_sentence.append(self.test_y[self.test_y["label"] == label_count.index[j]].index[0])
        return list(set(show_sentence))

    def update_sentence_positions(self, df, max_match):
        df["sentence_pos"] = "right"
        for a, b in max_match:
            df.loc[a, "sentence_pos"] = "left"
            df.loc[b, "sentence_pos"] = "left"

    def plot(self, df):
        min_x, max_x = df['x'].min()-1, df['x'].max()+2
        fig = go.Figure()
        fig = go.Figure(layout=go.Layout(
            autosize=False,  # 禁止图像自动调整大小
            height=800,  # 您可以根据需要调整这个值
            width=1500,  # 您可以根据需要调整这个值
            # plot_bgcolor="#262626",
        ))
        
        label_colors = self.generate_colormap(df['label'].unique())

        line_legend_group = "lines"

        # 为每个类别的点创建散点图
        for label, color in label_colors.items():
            mask = df["label"] == label
            fig.add_trace(go.Scatter(x=df[mask]['x'], y=df[mask]['y'], mode='markers', 
                                    marker=dict(color=color, size=df[mask]['size']),  # 添加 size 参数
                                    showlegend=True, legendgroup=line_legend_group,
                                    name = "label " + str(label)) 
                                    )

        sentence_counts = {'left': 0, 'right': 0}
        sentences_at_each_side = {
            'left': df[df['sentence'] != ''][df['sentence_pos'] == 'left'].shape[0],
            'right': df[df['sentence'] != ''][df['sentence_pos'] == 'right'].shape[0]
        }
        max_sentence_count = max(sentences_at_each_side.values())

        # Compute the similarity matrix
        distance_matrix = self.calculate_distance_matrix(df)

        # Create graph
        G = self.create_weighted_graph(distance_matrix)

        # Calculate maximum weight matching
        max_match = max_weight_matching(G, maxcardinality=True)

        # Update sentence_pos based on the max_match
        self.update_sentence_positions(df, max_match)

        # Update sentence_pos based on the max_match
        df["sentence_pos"] = "right"
        for a, b in max_match:
            if b > a:
                df.loc[a, "sentence_pos"] = "left"
                df.loc[b, "sentence_pos"] = "left"
        
        for x, y, label, sentence, pos in zip(df.x, df.y, df.label, df.sentence, df.sentence_pos):
            if not sentence:
                continue
            
            if pos == "left":
                x_offset = min_x
                sentence_counts["left"] += 1
                y_offset = sentence_counts['left'] * (2 * max_sentence_count) / sentences_at_each_side['left'] - max_sentence_count
            else:
                x_offset = max_x
                sentence_counts["right"] += 1
                y_offset = sentence_counts['right'] * (2 * max_sentence_count) / sentences_at_each_side['right'] - max_sentence_count
            
            sentence_annotation = dict(
                x=x_offset,
                y=y + y_offset,
                xref="x",
                yref="y",
                text=sentence,
                showarrow=False,
                xanchor="right" if pos == 'left' else 'left',
                yanchor='middle',
                font=dict(color="black"),
                bordercolor=label_colors.get(label, "black"),
                borderpad=2,
                bgcolor=self.divide_hex_color_by_half(label_colors.get(label, "black"))
            )
            fig.add_annotation(sentence_annotation)

            x_start = x - 1 if pos == 'left' else x + 1
            x_turn = x - 0.5 if pos == 'left' else x + 0.5
            y_turn = y

            fig.add_trace(go.Scatter(x=[x_offset, x_start, x_turn, x], y=[y + y_offset, y + y_offset, y_turn, y], mode='lines', 
                                    line=dict(color=label_colors.get(label, "black")), showlegend=False, legendgroup=line_legend_group))

        # 取消坐标轴的数字
        fig.update_xaxes(tickvals=[])
        fig.update_yaxes(tickvals=[])

        fig.show()
    
    def tsne_plot(self, n = 20):
        # 计算t-SNE，返回降维后的数据，每个元素为一个二维向量
        embedding_train = self.calculate_tsne()

        # 随机抽取显示文本, n为抽取的数量，show_sentence为一个列表，每个元素为显示文本的索引
        show_sentence = self.random_sentence(n)

        # 格式化数据，输出为一个pandas的DataFrame，包含x, y, label, sentence, sentence_pos, size
        # x, y为降维后的坐标，label为类别，sentence为显示的文本，sentence_pos为文本的位置("left", "right")，size为被选中文本的大小
        df = self.format_data(show_sentence, embedding_train, self.test_y['label'])
        sorted_df = df.sort_values('y').reset_index(drop=True)

        # 绘制图像
        self.plot(sorted_df)