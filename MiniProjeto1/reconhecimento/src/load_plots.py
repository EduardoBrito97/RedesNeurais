import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: load_plots.py [batch_index=0,1 or 2] [path_to_csv]')
        exit(1)
    df = pd.read_csv(sys.argv[2])
    b_size = pow(2,int(sys.argv[1])+2)
    data= df[df['Batch Size']==b_size]
    sns.lineplot(x='Learning Rate', y='Accuracy', hue='Hidden Layers', palette=['red','yellow','blue', 'black'],data=data)
    plt.savefig('Batch'+str(b_size)+'.png')