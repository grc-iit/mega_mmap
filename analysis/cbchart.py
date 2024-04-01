import plotly.graph_objs as go

# Sample data
categories = ['Category A', 'Category B', 'Category C']
values1 = [20, 35, 30]
values2 = [25, 32, 34]

# Creating traces for bar charts
trace1 = go.Bar(
    x=categories,
    y=values1,
    name='Group 1'
)
trace2 = go.Bar(
    x=categories,
    y=values2,
    name='Group 2'
)

# Creating trace for line graph
line_trace = go.Scatter(
    x=categories,
    y=[10, 20, 15],  # Example line data
    mode='lines+markers',
    name='Line Data',
    yaxis='y2'  # Secondary y-axis
)

# Define layout
layout = go.Layout(
    title='Combination Bar Chart and Line Graph',
    xaxis=dict(title='Categories'),
    yaxis=dict(title='Values'),
    yaxis2=dict(
        title='Line Data',
        overlaying='y',
        side='right'
    ),
    barmode='group'
)

# Create figure
fig = go.Figure(data=[trace1, trace2, line_trace], layout=layout)

# Show plot
fig.show()
