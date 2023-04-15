import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import ImageFont
from sklearn.metrics.pairwise import cosine_similarity

class Heatmap():
    def __init__(self, df, positions = None, font_path = './arial.ttf'):
        self.df = df
        self.positions = positions if positions != None else [(i, i) for i in range(len(df))]
        self.font_path = font_path
    
    def get_bg_color(self, hex_color):
        red, green, blue = hex_color[0], hex_color[1], hex_color[2]
        red_half = int(red) // 10 + (255 - 25)
        green_half = int(green) // 10 + (255 - 25)
        blue_half = int(blue) // 10 + (255 - 25)

        half_hex_color = "#{:02x}{:02x}{:02x}".format(red_half, green_half, blue_half)
        return half_hex_color

    def show_text(self, text):
        return text[:10] + '...' + text[-10:] if len(text) > 20 else text

    def get_text_width(self, font, text):
        return font.getsize(text)[0]

    def get_color_value(self, index, colorscale, min_value, max_value):
        value_range = max_value - min_value
        scaled_value = (index - min_value) / value_range
        color_index = int(scaled_value * (len(colorscale) - 1))
        return colorscale[color_index][1]

    def create_heatmap(self, positions = None):
        if positions is None:
            positions = self.positions
        # check if there's no repeated rows in positions, which is the second column of the positions list
        rows = [pos[1] for pos in positions]
        assert len(rows) == len(set(rows)), "There are repeated rows in positions"

        df = self.df
        # compute similarity matrix
        sim_matrix = cosine_similarity(df['first_embed'].tolist(), df['second_embed'].tolist())

        # Load the specific font and font size.
        font_default = ImageFont.truetype(self.font_path, 14)
        
        jet_colorscale = [
            [0.0, "rgb(0, 0, 255)"],
            [0.1, "rgb(0, 100, 255)"],
            [0.2, "rgb(0, 200, 255)"],
            [0.3, "rgb(50, 255, 255)"],
            [0.4, "rgb(150, 255, 255)"],
            [0.5, "rgb(255, 255, 0)"],
            [0.6, "rgb(255, 150, 0)"],
            [0.7, "rgb(255, 100, 0)"],
            [0.8, "rgb(255, 50, 0)"],
            [0.9, "rgb(255, 0, 0)"],
            [1.0, "rgb(150, 0, 0)"],
        ]
        
        fig = go.Figure(data=go.Heatmap(z=sim_matrix, colorscale=jet_colorscale))
        
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
                hex_color = color.lstrip('rgb(').rstrip(')').split(", ")
                bgcolor = self.get_bg_color(hex_color)
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
                                width=1.5,
                            )
                        )
                    )
        
        custom_width = 70 * len(sim_matrix)  # Increase the multiplier (40) for larger cells
        custom_height = 35 * len(sim_matrix)

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
