import matplotlib.pyplot as plt
import pandas as pd

def plot_line(df: pd.DataFrame, col:str, ascending = True, title: str = ''):
    k = df[col].sort_values(ascending = ascending)
    fig = plt.figure()
    plt.xticks( fontsize=10, rotation=40)
    plt.title(title)
    ax = plt.axes()
    ax.plot(range(len(k)), k)
    return fig,ax
