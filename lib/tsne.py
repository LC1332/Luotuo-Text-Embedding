import numpy as np
import pandas as pd
from openTSNE import TSNE
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

class TSNE_Plot():
    '''Class TSNE_Plot
    Method
    __init__
        input:
            sentence: List[string]
                用于展示的语句，应当是一个字符串的list
            embed: array
                用于计算相似度的embedding，应当是np.array(float), 或者tensor(float)等可以直接进行矩阵乘法的array
            label: [optional] List
                用于标注不同颜色的label，应当是一个list，对应每个sentence的label。如果没有提供label，将会自动调用聚类算法对sentence进行分类，默认的类别为3类
            n_clusters: [optional] int
                用于聚类算法的类别数量
            n_annotation_posisions: [optional] int
                展示的语句总数，默认为20句
    tsne_plot
        input:
            n_annotation_posisions: [optional] int
                展示的语句总数，默认为20句
            return_fig: [optional] boolean
                是否返回图像对象，如果为False，将直接绘制图像，如果为True，将返回图像对象，默认为False
        Output:
            若return_fig为True，将返回图像对象

    
    '''

    def __init__(self, sentence, embed, label = None, n_clusters :int = 3, n_annotation_positions:int = 20):
        assert n_clusters > 0, "N must be greater than 0" 
        self.N = n_clusters
        self.test_X = pd.DataFrame({"text": sentence, "embed": [np.array(i) for i in embed]})
        self.test_y = pd.DataFrame({'label':label}) if label is not None else pd.DataFrame({"label": self.cluster()})
        self.embed = self.calculate_tsne()
        self.init_df()

        self.n_annotation_positions = n_annotation_positions
        self.show_sentence = []
        self.random_sentence()

        
        self.annotation_positions = []
        self.get_annotation_positions()
        self.mapping = {}
    
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


    def get_annotation_positions(self):
        min_x, max_x = self.df['x'].min()-1, self.df['x'].max()+2
        n = self.n_annotation_positions

        y_min, y_max = self.df['y'].min() * 3, self.df['y'].max() * 3

        add = 0 if n % 2 == 0 else 1
        y_values = np.linspace(y_min, y_max, n//2+add)

        left_positions = [(min_x, y) for y in y_values]
        right_positions = [(max_x, y) for y in y_values]


        self.annotation_positions = left_positions + right_positions

    
    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def map_points(self):
        # Get points from the dataframe using the show_sentence indices
        points1 = [(self.embed[i][0], self.embed[i][1]) for i in self.show_sentence]

        # Create a distance matrix between the points
        distance_matrix = np.zeros((len(points1), len(self.annotation_positions)))

        for i, point1 in enumerate(points1):
            for j, point2 in enumerate(self.annotation_positions):
                distance_matrix[i, j] = self.euclidean_distance(point1, point2)

        # Apply linear_sum_assignment to find the optimal mapping
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        for i, j in zip(row_ind, col_ind):
            self.mapping[self.show_sentence[i]] = self.annotation_positions[j]


    def show_text(self, show_sentence, text):
        sentence = []
        for i in range(len(text)):
            if i in show_sentence:
                s = text[i][:10] + "..." + text[i][-10:]
                sentence.append(s)
            else:
                sentence.append("")
        return sentence

    def init_df(self):
        X, Y = np.split(self.embed, 2, axis=1)
        data = {
            "x": X.flatten(),
            "y": Y.flatten(),
        }

        self.df = pd.DataFrame(data)


    def format_data(self):
        sentence = self.show_text(self.show_sentence, self.test_X["text"])
        X, Y = np.split(self.embed, 2, axis=1)
        n = len(self.test_X)
        data = {
            "x": X.flatten(),
            "y": Y.flatten(),
            "label": self.test_y["label"],
            "sentence" : sentence,
            "size" : [20 if i in self.show_sentence else 10 for i in range(n)],
            "pos" : [{"x_offset": self.mapping.get(i, (0, 0))[0], "y_offset": self.mapping.get(i, (0, 0))[1]} for i in range(n)],
            "annotate" : [True if i in self.show_sentence else False for i in range(n)],
        }
        self.df = pd.DataFrame(data)

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
            verbose=False,
        )
        embedding_train = tsne.fit(compact_embedding)
        embedding_train = embedding_train.optimize(n_iter=1000, momentum=0.8)
        return embedding_train
    
    def idx_list_dict(self, df):
        def get_idx(group):
            return group.index.tolist()
        group = df.groupby("label")
        return_dict = group.apply(get_idx).to_dict()
        return return_dict

    def random_sentence(self):
        #汇总所有标签的句子索引
        idx_dict = self.idx_list_dict(self.test_y)

        m = len(self.test_y)
        n = min(self.n_annotation_positions, m)

        #从每个标签中随机抽取一个句子
        show_sentence = []
        while n > 0:
            for l, lst in idx_dict.items():
                show_sentence.append(lst.pop(np.random.choice(len(lst), 1, replace=False)[0]))
                n -= 1
                if n == 0:
                    break

        self.show_sentence = show_sentence


    def plot(self, return_fig=False):
        min_x, max_x = self.df['x'].min()-1, self.df['x'].max()+2
        fig = go.Figure()
        fig = go.Figure(layout=go.Layout(
            autosize=False,  # 禁止图像自动调整大小
            height=800,  # 您可以根据需要调整这个值
            width=1500,  # 您可以根据需要调整这个值
            # plot_bgcolor="#262626",
        ))
        
        label_colors = self.generate_colormap(self.df['label'].unique())

        line_legend_group = "lines"

        # 为每个类别的点创建散点图
        for label, color in label_colors.items():
            mask = self.df["label"] == label
            fig.add_trace(go.Scatter(x=self.df[mask]['x'], y=self.df[mask]['y'], mode='markers', 
                                    marker=dict(color=color, size=self.df[mask]['size']),  # 添加 size 参数
                                    showlegend=True, legendgroup=line_legend_group,
                                    name = str(label)) 
                                    )


        # 为每个句子创建注释
        for x, y, label, sentence, pos, annotate in zip(self.df.x, self.df.y, self.df.label, self.df.sentence, self.df.pos, self.df.annotate):
            if not sentence:
                continue
            if not annotate:
                continue
            # pos在左边
            criteria = (pos["x_offset"] - min_x) < 1e-2

            sentence_annotation = dict(
                x=pos["x_offset"],
                y=pos["y_offset"],
                xref="x",
                yref="y",
                text=sentence,
                showarrow=False,
                xanchor="right" if criteria else 'left',
                yanchor='middle',
                font=dict(color="black"),
                bordercolor=label_colors.get(label, "black"),
                borderpad=2,
                bgcolor=self.divide_hex_color_by_half(label_colors.get(label, "black"))
            )
            fig.add_annotation(sentence_annotation)

            x_start = x - 1 if criteria else x + 1
            x_turn = x - 0.5 if criteria else x + 0.5
            y_turn = y

            fig.add_trace(go.Scatter(x=[pos["x_offset"], x_start, x_turn, x], y=[pos["y_offset"], pos["y_offset"], y_turn, y], mode='lines', 
                                    line=dict(color=label_colors.get(label, "black")), showlegend=False, legendgroup=line_legend_group))

        # 取消坐标轴的数字
        fig.update_xaxes(tickvals=[])
        fig.update_yaxes(tickvals=[])

        if not return_fig:
            fig.show()
        else:
            return fig
    
    def tsne_plot(self, n_sentence = 20, return_fig=False):
        # 计算t-SNE，返回降维后的数据，每个元素为一个二维向量
        embedding_train = self.calculate_tsne()

        # 随机抽取显示文本, n为抽取的数量，show_sentence为一个列表，每个元素为显示文本的索引
        if self.n_annotation_positions != min(n_sentence, len(self.test_y)):
            self.n_annotation_positions = min(n_sentence, len(self.test_y))
            self.random_sentence()
            self.get_annotation_positions()

        # find the optimal sentence positions
        self.map_points()

        # 格式化数据，输出为一个pandas的DataFrame，包含x, y, label, sentence, sentence_pos, size
        # x, y为降维后的坐标，label为类别，sentence为显示的文本，sentence_pos为文本的位置("left", "right")，size为被选中文本的大小
        self.format_data()
        # self.df = self.df.sort_values('y').reset_index(drop=True)

        if not return_fig:
            # 绘制图像
            self.plot()
        else:
            return self.plot(return_fig=return_fig)