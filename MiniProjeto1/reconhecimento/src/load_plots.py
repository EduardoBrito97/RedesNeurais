import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    files = ["results_30hn.csv", "results_60hn.csv", "results_60ep_30hn.csv", "results_60ep_60hn.csv"]
    batchSizeIndexes = [0, 1, 2]

    for fileName in files:
        for batchSizeIndex in batchSizeIndexes:
            plt.figure()
            df = pd.read_csv(fileName)
            b_size = pow(2,int(batchSizeIndex)+2)
            data = df[df['Batch Size']==b_size]
            sns.scatterplot(x='Learning Rate', y='Accuracy', hue='Hidden Layers', s=30, palette=['red','green','blue', 'black'], data=data, marker="+", legend="full")
            plt.savefig('Graphs/' + fileName + ' - ' + 'Batch'+str(b_size)+'.png')