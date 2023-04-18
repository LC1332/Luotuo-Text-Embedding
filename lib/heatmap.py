import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import ImageFont, ImageColor
from sklearn.metrics.pairwise import cosine_similarity
from plotly.colors import find_intermediate_color

class Heatmap():
    '''Class Heatmap
    Method:
    __init__:
        input:
            df:pandas dataframe
                包含四列：first, second为两列字符串，分别对应一个pair中的第一句话和第二句话；first_embed, second_embed为两列向量，分别对应first和second语句的embedding
            position: [optional] List[tuple]
                包含需要展示语句的位置，默认为对角线位置格式为[(x1, y1), (x2, y2)]

    creat_heatmap:
        input:
            position: [optional] List[tuple]
                包含需要展示语句的位置，默认为对角线位置格式为[(x1, y1), (x2, y2)]
            font_path: [optional] string
                显示字体的路径，默认为'./arial.ttf'

    '''

    def __init__(self, df, positions = None):
        self.df = df
        self.positions = positions if positions != None else [(i, i) for i in range(len(df))]
    
    def get_bg_color(self, hex_color):
        red, green, blue = hex_color[0], hex_color[1], hex_color[2]
        red_half = int(float(red)) // 10 + (255 - 25)
        green_half = int(float(green)) // 10 + (255 - 25)
        blue_half = int(float(blue)) // 10 + (255 - 25)

        half_hex_color = "#{:02x}{:02x}{:02x}".format(red_half, green_half, blue_half)
        return half_hex_color

    def show_text(self, text):
        return text[:10] + '...' + text[-10:] if len(text) > 20 else text

    def get_text_width(self, font, text):
        return font.getsize(text)[0]

    def get_color_value(self, index, colorscale, min_value, max_value):

        hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))
        
        # Normalize the data point to get a value between 0 and 1
        normalized_value = (index - min_value) / (max_value - min_value)

        for cutoff, color in colorscale:
            if normalized_value > cutoff:
                low_cutoff, low_color = cutoff, color
            else:
                high_cutoff, high_color = cutoff, color
                break

        if (low_color[0] == "#") or (high_color[0] == "#"):
            low_color = hex_to_rgb(low_color)
            high_color = hex_to_rgb(high_color)
        intermediate_color = find_intermediate_color(
            lowcolor=low_color,
            highcolor=high_color,
            intermed=((normalized_value - low_cutoff) / (high_cutoff - low_cutoff)),
            colortype="rgb",
        )
        return intermediate_color
    def create_heatmap(self, positions = None, font_path = './arial.ttf'):
        if positions is None:
            positions = self.positions
        # check if there's no repeated rows in positions, which is the second column of the positions list
        rows = [pos[1] for pos in positions]
        assert len(rows) == len(set(rows)), "There are repeated rows in positions"

        df = self.df
        # compute similarity matrix
        sim_matrix = cosine_similarity(df['first_embed'].tolist(), df['second_embed'].tolist())

        # Load the specific font and font size.
        font_default = ImageFont.truetype(font_path, 14)

        fig = go.Figure(data=go.Heatmap(z=sim_matrix, colorscale='Jet'))
        jet_colorscale = fig.data[0].colorscale
        
        # Add annotations for captions
        annotations = []
        shapes = []
        
        # Retrieve colors for the given positions
        position_colors = {}
        min_value, max_value = np.amin(sim_matrix), np.amax(sim_matrix)
        for col, row in positions:
            color_index = sim_matrix[row][col]
            color_value = self.get_color_value(color_index, jet_colorscale, min_value, max_value)
            position_colors[(col, row)] = color_value

        title_row = go.layout.Annotation(
            text="Row",
            font=dict(size=16, color="black"),
            x = -24,
            y = -1,
            showarrow=False,
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="bottom",
        )

        title_col = go.layout.Annotation(
            text="Col",
            font=dict(size=16, color="black"),
            x = -9,
            y = -1,
            showarrow=False,
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="bottom",
        )
        annotations.append(title_col)
        annotations.append(title_row)


        for pos in positions:
            col, row = pos
            for j, position in enumerate([(-15, 1), (-30, 0)]):
                text = self.show_text(df['first'][col] if j == 0 else df['second'][row])
                color = position_colors[(col, row)]
                hex_color = color.lstrip('rgb(').rstrip(')').split(",")
                bgcolor = self.get_bg_color(hex_color)

                # 文字
                annotations.append(
                    go.layout.Annotation(
                        x=position[0],
                        y=row,
                        text=text,
                        showarrow=False,
                        font=dict(size=14, color="black"),
                        bordercolor=color,
                        xref="x",
                        yref="y",
                        yshift=0,
                        xshift=-10,
                        xanchor="left",
                        yanchor="middle",
                        bgcolor=bgcolor,
                    )
                )
                if j == 0:
                    #线
                    text_width = self.get_text_width(font_default, text) / 90
                    shapes.append(
                        go.layout.Shape(
                            type='line',
                            x0= - text_width-2,
                            x1=col,
                            y0=row,
                            y1=row,
                            yref='y',
                            xref='x',
                            line=dict(
                                color=color,
                                width=2,
                            )
                        )
                    )
        
        custom_width = 1.8 * len(sim_matrix) * len(sim_matrix)  # Increase the multiplier (40) for larger cells
        custom_height = 1 * len(sim_matrix) * len(sim_matrix)

        fig.update_layout(
            width=custom_width,
            height=custom_height,
            margin=dict(l=120),
            xaxis=dict(tickmode="array", tickvals=list(range(len(sim_matrix))), ticktext=list(range(len(sim_matrix))), title='X Axis'),
            yaxis=dict(tickmode="array", tickvals=list(range(len(sim_matrix))), ticktext=list(range(len(sim_matrix))), autorange="reversed", title='Y Axis'),
            annotations=annotations,
            shapes=shapes
        )

        # display plot
        fig.show()
