import os
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"))

value_counts = df['entropy'].value_counts()
count_of_desired_value = value_counts[0.0]
print("Zero entropy samples: " + str(count_of_desired_value))

column_data = df['entropy']

fig = px.histogram(column_data, nbins=20, marginal='rug', histnorm='probability density')

# Calculate KDE
kde = gaussian_kde(column_data)
x_vals = np.linspace(column_data.min(), column_data.max(), 1000)
y_vals = kde(x_vals)

kde_trace = go.Scatter(x=x_vals, y=y_vals, mode='lines', name='KDE')

fig.add_trace(kde_trace)

fig.show()

