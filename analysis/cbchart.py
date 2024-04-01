from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
import numpy as np

# Sample data
categories = ['Category A', 'Category B', 'Category C']
values1 = [20, 35, 30]
values2 = [25, 32, 34]
line_data = [10, 20, 15]  # Example line data

# Create figures for each subplot
subplots = []
for i in range(4):
    p = figure(title=f'Subplot {i+1}', x_range=categories, height=250, width=300)

    # Clustered bar chart
    p.vbar(x=np.arange(len(categories)) + 0.1, top=values1, width=0.2, color="blue", legend_label='Group 1')
    p.vbar(x=np.arange(len(categories)) - 0.1, top=values2, width=0.2, color="orange", legend_label='Group 2')

    # Line graph
    p.line(categories, line_data, line_color="green", line_width=2, y_range_name="line")

    p.y_range.start = 0
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    subplots.append(p)

# Arrange subplots in a 2x2 grid
grid = gridplot([[subplots[0], subplots[1]], [subplots[2], subplots[3]]])

# Show the grid
show(grid)