from feature_selection import *
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import os
import pickle as pkl
from sklearn_som.som import SOM
from sklearn.cluster import SpectralClustering


def clustering_expermient(main_folder):
    print('Clustering ... ')

    run_info_path = os.path.join(main_folder, 'run_info.pickle')

    with open(run_info_path, 'rb') as handle:
        run_info = pkl.load(handle)

    main_folder = run_info['main_folder']

    tsne, y_train, tsne_test, y_test = clustering_prepare(main_folder)

    features_path = main_folder + 'features/'

    vae_features_train = np.load(features_path + 'vae_train.npy')
    vae_features_test = np.load(features_path + 'vae_test.npy')
    y_train = np.load(main_folder + 'train_label.npy')  # [:, 1]
    y_test = np.load(main_folder + 'test_label.npy')  # [:, 1]

    lda_features_train = np.load(features_path + 'lda_tfidf_train.npy')
    lda_features_test = np.load(features_path + 'lda_tfidf_test.npy')
    data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
    data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)

    for m in [2, 4, 6, 8, 10, 12]:
        for n in [2, 4, 6, 8, 10, 12]:
            som = SOM(m=4, n=4, dim=64)
            som.fit(data3_train)
            predictions = som.predict(data3_train)
            plot_clustering(tsne, predictions, y_train, main_folder, 'dbscan_' + 'train_' + str(m) + '_' + str(n))

            som = SOM(m=4, n=4, dim=64)
            som.fit(data3_test)
            predictions_te = som.predict(data3_test)
            plot_clustering(tsne_test, predictions_te, y_test, main_folder, 'dbscan_' + 'test_' + str(m) + '_' + str(n))



    dbscan_clustering(tsne, y_train, main_folder, 'train')
    dbscan_clustering(tsne_test, y_test, main_folder, 'test')

    hierarchical_clustering(tsne, main_folder, 'train')
    hierarchical_clustering(tsne_test, main_folder, 'test')

    agglomerative_clustering(tsne, y_train, main_folder, 'train')
    agglomerative_clustering(tsne_test, y_test, main_folder, 'test')

    som_clustering(tsne, y_train, main_folder, 'train')
    som_clustering(tsne_test, y_test, main_folder, 'test')


def plot_clustering(tsne, labels, y, folder, name):

    tsne1 = tsne[:, 0]
    tsne2 = tsne[:, 1]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharey='all')

    axes[0].scatter(tsne1[list(np.where(y == 0)[0])], tsne2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0].scatter(tsne1[list(np.where(y == 1)[0])], tsne2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[1].scatter(tsne[:,0], tsne[:,1], c=labels, s=5, cmap='rainbow')

    axes[0].legend(loc='upper right', prop={'size': 14}, markerscale=6)

    axes[0].set_ylabel('TSNE 2', fontsize=14)
    axes[0].set_xlabel('TSNE 1', fontsize=14)
    axes[1].set_xlabel('TSNE 1', fontsize=14)

    fig.savefig(folder + name + '.png')


def clustering_prepare(main_folder):

    features_path = main_folder + 'features/'

    vae_features_train = np.load(features_path + 'vae_train.npy')
    vae_features_test = np.load(features_path + 'vae_test.npy')
    lda_features_train = np.load(features_path + 'lda_tfidf_train.npy')
    lda_features_test = np.load(features_path + 'lda_tfidf_test.npy')
    y_train = np.load(main_folder + 'train_label.npy')  # [:, 1]
    y_test = np.load(main_folder + 'test_label.npy')  # [:, 1]

    data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
    data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)

    tsne1, tsne2 = make_tSNE(data3_train, tsne_perplexity=40, n_components=2)
    tsne = np.concatenate([np.expand_dims(tsne1, -1), np.expand_dims(tsne2, -1)], axis=1)


    tsne1_test, tsne2_test = make_tSNE(data3_test, tsne_perplexity=40, n_components=2)
    tsne_test = np.concatenate([np.expand_dims(tsne1_test, -1), np.expand_dims(tsne2_test, -1)], axis=1)

    # fig = plt.figure()
    # plt.scatter(tsne1[list(np.where(y_train == 0)[0])], tsne2[list(np.where(y_train == 0)[0])], s=5, alpha=0.5, label='real')
    # plt.scatter(tsne1[list(np.where(y_train == 1)[0])], tsne2[list(np.where(y_train == 1)[0])], s=5, alpha=0.5, label='fake')
    # plt.tight_layout()
    # fig.savefig(main_folder + 'clustering_tr_full.png')


    return tsne, y_train, tsne_test, y_test


def dbscan_clustering(tsne, y_train, main_folder,  data_name):
    print('DBScan')
    clustering = DBSCAN(eps=2, min_samples=2).fit(tsne)
    plot_clustering(tsne, clustering.labels_, y_train, main_folder, 'dbscan_' + data_name)


def hierarchical_clustering(tsne, main_folder, data_name):
    print('Hierarchical')

    linked = linkage(tsne, 'single')

    f3 = plt.figure(figsize=(10, 7))
    dendrogram(linked,
                orientation='top',
                distance_sort='descending',
                show_leaf_counts=True)
    f3.savefig(main_folder + 'hierarchical' + data_name + '.png')


def agglomerative_clustering(tsne, y_train, main_folder, data_name):
    print('Agglomerative')

    cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
    cluster.fit_predict(tsne)
    plot_clustering(tsne, cluster.labels_, y_train, main_folder, 'agglomerative_' + data_name)


def som_clustering(tsne, y_train, main_folder, data_name):
    print('Self Organizing Map')

    som = SOM(m=10, n=10, dim=2)
    som.fit(tsne)
    predictions = som.predict(tsne)
    plot_clustering(tsne, predictions, y_train, main_folder, 'som_' + data_name)


def spectral_clustering(tsne, y_train, main_folder, data_name):
    print('Spectral')

    clustering = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0).fit(tsne)
    plot_clustering(tsne, clustering.labels_, y_train, main_folder, 'spectral_' + data_name)
