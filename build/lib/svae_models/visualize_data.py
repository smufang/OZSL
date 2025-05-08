import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

def tsne_plot(data, target, file_name, use_marker=False):
    num_classes = np.unique(target).shape[0]

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(data)

    df = pd.DataFrame()
    df["y"] = target
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    scatterplot = plt.figure(figsize=(16,10))
    args = {'x': "comp-1", 
            'y': "comp-2", 
            'hue': df.y.tolist(),
            'palette': sns.color_palette("hls", num_classes),
            'data': df}
    if use_marker:
        args['style'] = df.y.tolist()
    
    sns.scatterplot(**args)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig = scatterplot.get_figure()
    fig.savefig("{}.png".format(file_name)) 
    plt.close()