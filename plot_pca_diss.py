import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE


def plot_features(file):
    features = []
    labels = []

    with open(file) as f:
        for line in f:
            values = line.split(' ')
            labels.append(values[0])
            features.append([float(i.strip().split(':')[1]) for i in values[1:] if i != ''])

    features = np.array(features)

    pca = IncrementalPCA(n_components=2, batch_size=3)
    pca.fit(features)
    data = pca.transform(features)

    # tsne = TSNE(n_components=2)
    # data = tsne.fit_transform(features)

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    print(le.classes_)

    colors = ['navy', 'darkorange']

    plt.figure(figsize=(8, 8))

    for color, i, target_name in zip(colors, [0, 1], le.classes_):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], color=color, lw=2, label=target_name)

    plt.show()
