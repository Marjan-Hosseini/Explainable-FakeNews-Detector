import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


def spearsman_FS(input_tr, output_tr, threshold, rank_num, fs_mode="rank"):
    """The function returns the columns names of the selected features and
    the spearsman matrix for the selected one according to the mode"""

    """Fmode can be "rank" or "threshold" """

    col_list = input_tr.columns.values.tolist()
    selected_features = []
    spearsman_matrix = pd.DataFrame(data=None, index=['rho', 'p_value'], columns=col_list)

    for col in col_list:
        rho, p_val = spearmanr(input_tr.loc[:, col], output_tr)
        spearsman_matrix.loc['rho', col] = rho
        spearsman_matrix.loc['p_value', col] = p_val
        if fs_mode == "threshold":
            if np.abs(rho) >= threshold:
                selected_features.append(col)

    if fs_mode == "rank":
        sortedF = np.argsort(np.abs(spearsman_matrix.loc['rho', :])).tolist()
        for f in range(rank_num):
            selected_features.append(col_list[sortedF[-f-1]])

    return selected_features, spearsman_matrix


def SelectKBestFS(input, output, n_f):
    input = normalize_data_min_max_sk(input)
    selected_features = SelectKBest(chi2, k=n_f).fit_transform(input, output)
    return selected_features


def make_tSNE(features, tsne_perplexity=40, n_components=2):
    # make tsne object
    tsne = TSNE(n_components=n_components, verbose=0, perplexity=tsne_perplexity, n_jobs=-1)

    # perform the dimensionality reduction using tsne object
    tsne_results = tsne.fit_transform(features)

    # results of tSNE
    tsne1 = tsne_results[:, 0]
    tsne2 = tsne_results[:, 1]

    return tsne1, tsne2


def plot_tsne_results(tsne1, tsne2, y, tsne, tsne_perplexity, main_folder, model_name, dataname):
    f2 = plt.figure(figsize=(10, 9))

    plt.scatter(tsne1[y == 0], tsne2[y == 0], s=5, alpha=0.5, label='real')
    plt.scatter(tsne1[y == 1], tsne2[y == 1], s=5, alpha=0.5, label='fake')

    plt.xlabel('tSNE axis 1')
    plt.ylabel('tSNE axis 2')
    # mpl.rcParams['legend.markerscale'] = 3
    plt.legend(loc='upper right', prop={"size": 10})
    plt.title('Perplexity = ' + str(tsne_perplexity) + ' - Kl-divergence = ' + str(tsne.kl_divergence_))
    f2.savefig(main_folder + 'plots/tsne_perp' + str(tsne_perplexity) + model_name + dataname + '.png')
    f2.savefig(main_folder + 'plots/tsne_perp' + str(tsne_perplexity) + model_name + dataname + '.pdf')


def normalize_data_min_max_sk(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)


def make_pca(data, n_components=2): #defautl is None to get all dim
    # print('wait ... running PCA feature selection')
    normalized_data = normalize_data_min_max_sk(data)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(normalized_data)
    pca1 = pca_result[:, 0]
    pca2 = pca_result[:, 1]
    return pca1, pca2


def plot_pca_results(pca1, pca2, y, main_folder, dataname):
    f2 = plt.figure(figsize=(10, 9))
    plt.scatter(pca1[list(np.where(y == 0)[0])], pca2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    plt.scatter(pca1[list(np.where(y == 1)[0])], pca2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')

    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(loc='upper right', prop={"size": 10})
    f2.savefig(main_folder + 'plots/PCA_' + dataname + '.png')
    f2.savefig(main_folder + 'plots/PCA_' + dataname + '.pdf')


def random_forest_feasture_selection(x_train, y_train, nf):
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    aa=clf.feature_importances_
    ranked = np.argsort(aa)
    selected_indices = ranked[::-1][:nf]
    return x_train[:, selected_indices]
