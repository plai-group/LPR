import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd


def setup_matplotlib():
    nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
    plt.rcParams.update(nice_fonts)


setup_matplotlib()


# Extracting strategy names and their corresponding color for the plot.

curr_df = pd.read_csv('~/Downloads/c100_grad_ratio_curr.csv')
replay_df = pd.read_csv('~/Downloads/c100_grad_ratio_replay.csv')
data = pd.concat((curr_df, replay_df))

print(data.head())


loss_columns = ['Step', 'strategy.lpr: true - Grad Current/Grad Norm Projection Ratio',
                'strategy.lpr: true - Grad Replay/Grad Norm Projection Ratio']
data = data[loss_columns].rename(columns={
    'strategy.lpr: true - Grad Current/Grad Norm Projection Ratio': 'Current Loss',
    'strategy.lpr: true - Grad Replay/Grad Norm Projection Ratio': 'Replay Loss'})
loss_names = ['Replay Loss', 'Current Loss']

palette = sns.color_palette(None, len(loss_names))

# Now we will plot each strategy with its corresponding confidence interval
plt.figure(figsize=(14, 8))

lengends_list = []
for i, loss_col in enumerate(loss_names):
    # Extract strategy-specific columns for accuracy and confidence intervals
    legends = loss_col
    lengends_list.append(legends)
    sns.lineplot(data=data, x='Step', y=loss_col,
                 label=legends, color=palette[i])  # , linewidth=3.5)

# Enhance the plot with titles and labels
plt.title('Proximal Gradient Norm / Standard Gradient Norm')  # , fontsize=30)
# plt.title('Average Training Accuracy per Task TinyImageNet', fontsize=30)
plt.xlabel('Training Step')  # , fontsize=25)
plt.ylabel('Ratio')  # , fontsize=25)

# plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20)

# Show plot with a tight layout
plt.xticks()  # fontsize=18)
plt.yticks()  # fontsize=18)

plt.tight_layout()
plt.savefig('c100_loss_decomp.pdf')
