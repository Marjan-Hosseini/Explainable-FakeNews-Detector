import sys
import os
sys.path.append('../')
import matplotlib
matplotlib.use('Agg')
from feature_selection import *
import seaborn as sns
import pickle as pkl
import matplotlib.gridspec as gridspec
from wordcloud import WordCloud


def plot_history_independent(top_folder, plots_path, dataset_name):
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if dataset_name in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[1]

        with open(run + '/model_history.pickle', 'rb') as handle:
            model_history = pkl.load(handle)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

        axes[0].plot(model_history['decoded_txt_accuracy'], label='Reconstruction (Training)')
        axes[0].plot(model_history['fnd_output_accuracy'], label='Classifier (Training)')
        axes[0].plot(model_history['val_decoded_txt_accuracy'], label='Reconstruction (Validation)')
        axes[0].plot(model_history['val_fnd_output_accuracy'], label='Classifier (Validation)')
        # axes[0].set_title('Subplot 1', fontsize=14)
        axes[0].set_xlabel('Epoch', fontsize=14)
        axes[0].set_ylabel('Accuracy', fontsize=14)
        axes[0].legend(loc='lower right')

        axes[1].plot(model_history['decoded_txt_loss'], label='Reconstruction (Training)')
        axes[1].plot(model_history['fnd_output_loss'], label='Classifier (Training)')
        axes[1].plot(model_history['val_decoded_txt_loss'], label='Reconstruction (Validation)')
        axes[1].plot(model_history['val_fnd_output_loss'], label='Classifier (Validation)')
        # axes[0].set_title('Subplot 1', fontsize=14)
        axes[1].set_xlabel('Epoch', fontsize=14)
        axes[1].set_ylabel('Loss', fontsize=14)
        axes[1].legend(loc='lower right')

        # Save figure
        fig.savefig(plots_path + 'model_history' + model_name + '.png')
        fig.savefig(plots_path + 'model_history' + model_name + '.pdf')


def plot_history_with_2_outputs(main_folder, model_history):

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    axes[0].plot(model_history['decoded_txt_accuracy'], label='Reconstruction (Training)')
    axes[0].plot(model_history['fnd_output_accuracy'], label='Classifier (Training)')
    axes[0].plot(model_history['val_decoded_txt_accuracy'], label='Reconstruction (Validation)')
    axes[0].plot(model_history['val_fnd_output_accuracy'], label='Classifier (Validation)')
    # axes[0].set_title('Subplot 1', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Accuracy', fontsize=14)
    axes[0].legend(loc='lower right')

    axes[1].plot(model_history['decoded_txt_loss'], label='Reconstruction (Training)')
    axes[1].plot(model_history['fnd_output_loss'], label='Classifier (Training)')
    axes[1].plot(model_history['val_decoded_txt_loss'], label='Reconstruction (Validation)')
    axes[1].plot(model_history['val_fnd_output_loss'], label='Classifier (Validation)')
    # axes[0].set_title('Subplot 1', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('Loss', fontsize=14)
    axes[1].legend(loc='lower right')

    # Save figure
    fig.savefig(main_folder + 'model_history.png')
    fig.savefig(main_folder + 'model_history.pdf')


def plot_history_with_2_outputs2(main_folder, model_history, plot_name):

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    axes[0].plot(model_history['decoded_txt_accuracy'], label='Reconstruction (Training)')
    axes[0].plot(model_history['fnd_output_accuracy'], label='Classifier (Training)')
    axes[0].plot(model_history['val_decoded_txt_accuracy'], label='Reconstruction (Validation)')
    axes[0].plot(model_history['val_fnd_output_accuracy'], label='Classifier (Validation)')
    # axes[0].set_title('Subplot 1', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Accuracy', fontsize=14)
    axes[0].legend(loc='lower right')

    axes[1].plot(model_history['decoded_txt_loss'], label='Reconstruction (Training)')
    axes[1].plot(model_history['fnd_output_loss'], label='Classifier (Training)')
    axes[1].plot(model_history['val_decoded_txt_loss'], label='Reconstruction (Validation)')
    axes[1].plot(model_history['val_fnd_output_loss'], label='Classifier (Validation)')
    # axes[0].set_title('Subplot 1', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('Loss', fontsize=14)
    axes[1].legend(loc='lower right')

    # Save figure
    fig.savefig(main_folder + 'model_history_' + plot_name + '.png')
    fig.savefig(main_folder + 'model_history_' + plot_name + '.pdf')


def plot_history_with_1_outputs(top_folder, plots_path, dataset_name):
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if dataset_name in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[1]

        with open(run + '/model_history.pickle', 'rb') as handle:
            model_history = pkl.load(handle)

            fig, axes = plt.subplots(figsize=(10, 5))

            axes.plot(model_history['decoded_txt_accuracy'], label='Reconstruction (Training)')
            axes.plot(model_history['fnd_output_accuracy'], label='Classifier (Training)')
            axes.plot(model_history['val_decoded_txt_accuracy'], label='Reconstruction (Validation)')
            axes.plot(model_history['val_fnd_output_accuracy'], label='Classifier (Validation)')
            # axes[0].set_title('Subplot 1', fontsize=14)
            axes.set_xlabel('Epoch', fontsize=14)
            axes.set_ylabel('Accuracy', fontsize=14)
            axes.legend(loc='lower right')

            # Save figure
            fig.savefig(plots_path + 'model_history_acc_' + model_name + '.png')
            fig.savefig(plots_path + 'model_history_acc_' + model_name + '.pdf')


def plot_history_2_data(top_folder, plots_path, config='topics_32_latent_dim_32'):

    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if config in dname]
    if len(runs) > 2:
        print('More than 2 runs.')

    with open(runs[0] + '/model_history.pickle', 'rb') as handle:
        model_history_1 = pkl.load(handle)

    with open(runs[1] + '/model_history.pickle', 'rb') as handle:
        model_history_2 = pkl.load(handle)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), sharex=False)

        axes[0].plot(model_history_1['decoded_txt_accuracy'], label='Reconstruction (Training)')
        axes[0].plot(model_history_1['fnd_output_accuracy'], label='Classifier (Training)')
        axes[0].plot(model_history_1['val_decoded_txt_accuracy'], label='Reconstruction (Validation)')
        axes[0].plot(model_history_1['val_fnd_output_accuracy'], label='Classifier (Validation)')
        # axes[0].set_title('ISOT', fontsize=14)
        # axes[0].set_xlabel('Epoch', fontsize=14)
        axes[0].set_ylabel('Accuracy', fontsize=14)
        legend1 = axes[0].legend(loc='lower right', prop={'size': 14}, markerscale=6, title='ISOT')
        plt.setp(legend1.get_title(), fontsize='x-large')

        axes[1].plot(model_history_2['decoded_txt_accuracy'], label='Reconstruction (Training)')
        axes[1].plot(model_history_2['fnd_output_accuracy'], label='Classifier (Training)')
        axes[1].plot(model_history_2['val_decoded_txt_accuracy'], label='Reconstruction (Validation)')
        axes[1].plot(model_history_2['val_fnd_output_accuracy'], label='Classifier (Validation)')
        # axes[1].set_title('Twitter', fontsize=14)
        axes[1].set_xlabel('Epoch', fontsize=14)
        axes[1].set_ylabel('Accuracy', fontsize=14)
        legend2 = axes[1].legend(loc='lower right', prop={'size': 14}, markerscale=6, title='Twitter')
        plt.setp(legend2.get_title(), fontsize='x-large')

        # axes[0, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)
        # axes[1, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)
        # axes[2, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)
        #
        # axes[0, 0].text(0.02, 0.03, 'VAE, ' + r'$\mathcal{D}_{tr}$',
        #                 verticalalignment='bottom', horizontalalignment='left',
        #                 transform=axes[0, 0].transAxes,
        #                 color='black', fontsize=20)
        # axes[1, 0].text(0.02, 0.03, 'LDA, ' + r'$\mathcal{D}_{tr}$',
        #                 verticalalignment='bottom', horizontalalignment='left',
        #                 transform=axes[1, 0].transAxes,
        #                 color='black', fontsize=20)

        fig.savefig(plots_path + 'model_history_acc_' + config + '.png')
        fig.savefig(plots_path + 'model_history_acc_' + config + '.pdf')


def general_plot():
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    x = np.linspace(1, 100, 100)
    y1 = (1+(1/x))**x
    y2 = (1-(1/x))**x
    axes[0].plot(x, y1, label=r'$f_1(x) = (1 + \frac{1}{x})^x$')
    axes[0].set_xlabel('x', fontsize=14)
    axes[0].set_ylabel(r'$f_1(x) = (1 + \frac{1}{x})^x$', fontsize=10)
    axes[0].legend(loc='upper left')

    axes[1].plot(x, y2, label=r'$f_1(x) = (1 - \frac{1}{x})^x$')
    axes[1].set_xlabel('x', fontsize=14)
    axes[1].set_ylabel(r'$f_1(x) = (1 - \frac{1}{x})^x$', fontsize=10)
    axes[1].legend(loc='upper right')
    plt.show()

    plt.subplots(figsize=(10, 10))
    x = np.linspace(1, 100, 100)
    y1 = (1+(1/x))**x
    y2 = (1-(1/x))**x
    y3 = y2*y1
    plt.plot(x, y1, label=r'$f(x) = (1 + \frac{1}{x})^x$')
    plt.plot(x, y2, label=r'$f(x) = (1 - \frac{1}{x})^x$')
    plt.plot(x, y3, label=r'$f(x) = (1 + \frac{1}{x})^x (1 - \frac{1}{x})^x$')
    plt.xlabel('x', fontsize=14)
    plt.ylabel(r'$f(x)$', fontsize=10)
    plt.legend(loc='upper right')
    plt.show()


def plot_3_pca(data1, data2, data3, y_train, folder, plot_name='self'):
    # y = y_train[:, 1]
    y = y_train

    pca1_data1, pca2_data1 = make_pca(data1, n_components=2)
    pca1_data2, pca2_data2 = make_pca(data2, n_components=2)
    pca1_data3, pca2_data3 = make_pca(data3, n_components=2)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
    axes[0].scatter(pca1_data1[list(np.where(y == 0)[0])], pca2_data1[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0].scatter(pca1_data1[list(np.where(y == 1)[0])], pca2_data1[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[1].scatter(pca1_data2[list(np.where(y == 0)[0])], pca2_data2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1].scatter(pca1_data2[list(np.where(y == 1)[0])], pca2_data2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[2].scatter(pca1_data3[list(np.where(y == 0)[0])], pca2_data3[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2].scatter(pca1_data3[list(np.where(y == 1)[0])], pca2_data3[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    axes[2].legend(loc='upper right')
    axes[0].set_ylabel('PCA2', fontsize=8)
    axes[1].set_ylabel('PCA2', fontsize=8)
    axes[2].set_xlabel('PCA1', fontsize=8)
    axes[2].set_ylabel('PCA2', fontsize=8)
    axes[0].set_title('VAE features', fontsize=10)
    axes[1].set_title('LDA features', fontsize=10)
    axes[2].set_title('VAE + LDA features', fontsize=10)

    plt.tight_layout()
    fig.savefig(folder + 'PCA_all_' + plot_name + '.png')
    fig.savefig(folder + 'PCA_all_' + plot_name + '.pdf')


def plot_3_pca_new(data1, data2, data3, y_train, folder, plot_name='self'):
    y = y_train[:, 1]
    pca1_data1, pca2_data1 = make_pca(data1, n_components=2)
    pca1_data2, pca2_data2 = make_pca(data2, n_components=2)
    pca1_data3, pca2_data3 = make_pca(data3, n_components=2)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 10), sharex=True)
    axes[0].scatter(pca1_data1[list(np.where(y == 0)[0])], pca2_data1[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0].scatter(pca1_data1[list(np.where(y == 1)[0])], pca2_data1[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[1].scatter(pca1_data2[list(np.where(y == 0)[0])], pca2_data2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1].scatter(pca1_data2[list(np.where(y == 1)[0])], pca2_data2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[2].scatter(pca1_data3[list(np.where(y == 0)[0])], pca2_data3[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2].scatter(pca1_data3[list(np.where(y == 1)[0])], pca2_data3[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[0].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[1].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[2].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[0].set_ylabel('PCA2', fontsize=14)
    axes[1].set_ylabel('PCA2', fontsize=14)
    axes[2].set_xlabel('PCA1', fontsize=14)
    axes[2].set_ylabel('PCA2', fontsize=14)
    axes[0].text(0.02, 0.03, 'VAE',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[0].transAxes,
            color='black', fontsize=20)
    axes[1].text(0.02, 0.03, 'LDA',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[1].transAxes,
            color='black', fontsize=20)
    axes[2].text(0.02, 0.03, 'VAE + LDA',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[2].transAxes,
            color='black', fontsize=20)

    # axes[0].set_title('VAE features', fontsize=10)
    # axes[1].set_title('LDA features', fontsize=10)
    # axes[2].set_title('VAE + LDA features', fontsize=10)
    plt.tight_layout()
    fig.savefig(folder + 'PCA_all_' + plot_name + '.png')
    fig.savefig(folder + 'PCA_all_' + plot_name + '.pdf')


def plot_3_pca_new_trte(tr_data, te_data, y_train, y_test, folder, plot_name='self'):
    if len(y_train.shape) == 2:
        y = y_train[:, 1]
    else:
        y = y_train

    if len(y_test.shape) == 2:
        y_te = y_test[:, 1]
    else:
        y_te = y_test

    (tr_data1, tr_data2, tr_data3) = tr_data
    (te_data1, te_data2, te_data3) = te_data

    tr_pca1_data1, tr_pca2_data1 = make_pca(tr_data1, n_components=2)
    tr_pca1_data2, tr_pca2_data2 = make_pca(tr_data2, n_components=2)
    tr_pca1_data3, tr_pca2_data3 = make_pca(tr_data3, n_components=2)

    te_pca1_data1, te_pca2_data1 = make_pca(te_data1, n_components=2)
    te_pca1_data2, te_pca2_data2 = make_pca(te_data2, n_components=2)
    te_pca1_data3, te_pca2_data3 = make_pca(te_data3, n_components=2)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
    axes[0, 0].scatter(tr_pca1_data1[list(np.where(y == 0)[0])], tr_pca2_data1[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0, 0].scatter(tr_pca1_data1[list(np.where(y == 1)[0])], tr_pca2_data1[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[1, 0].scatter(tr_pca1_data2[list(np.where(y == 0)[0])], tr_pca2_data2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1, 0].scatter(tr_pca1_data2[list(np.where(y == 1)[0])], tr_pca2_data2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[2, 0].scatter(tr_pca1_data3[list(np.where(y == 0)[0])], tr_pca2_data3[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2, 0].scatter(tr_pca1_data3[list(np.where(y == 1)[0])], tr_pca2_data3[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[0, 1].scatter(te_pca1_data1[list(np.where(y_te == 0)[0])], te_pca2_data1[list(np.where(y_te == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0, 1].scatter(te_pca1_data1[list(np.where(y_te == 1)[0])], te_pca2_data1[list(np.where(y_te == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[1, 1].scatter(te_pca1_data2[list(np.where(y_te == 0)[0])], te_pca2_data2[list(np.where(y_te == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1, 1].scatter(te_pca1_data2[list(np.where(y_te == 1)[0])], te_pca2_data2[list(np.where(y_te == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[2, 1].scatter(te_pca1_data3[list(np.where(y_te == 0)[0])], te_pca2_data3[list(np.where(y_te == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2, 1].scatter(te_pca1_data3[list(np.where(y_te == 1)[0])], te_pca2_data3[list(np.where(y_te == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[0, 0].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[1, 0].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[2, 0].legend(loc='upper right', prop={'size': 14}, markerscale=6)

    axes[0, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[1, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[2, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)

    axes[0, 0].set_ylabel('PCA2', fontsize=14)
    axes[1, 0].set_ylabel('PCA2', fontsize=14)
    axes[2, 0].set_ylabel('PCA2', fontsize=14)
    axes[2, 0].set_xlabel('PCA1', fontsize=14)
    axes[2, 1].set_xlabel('PCA1', fontsize=14)

    axes[0,0].text(0.02, 0.03, 'VAE, '+r'$\mathcal{D}_{tr}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[0, 0].transAxes,
            color='black', fontsize=20)
    axes[1,0].text(0.02, 0.03, 'LDA, '+r'$\mathcal{D}_{tr}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[1, 0].transAxes,
            color='black', fontsize=20)
    axes[2,0].text(0.02, 0.03, 'VAE + LDA, '+r'$\mathcal{D}_{tr}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[2, 0].transAxes,
            color='black', fontsize=20)

    axes[0,1].text(0.02, 0.03, 'VAE, '+r'$\mathcal{D}_{te}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[0, 1].transAxes,
            color='black', fontsize=20)
    axes[1,1].text(0.02, 0.03, 'LDA, '+r'$\mathcal{D}_{te}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[1, 1].transAxes,
            color='black', fontsize=20)
    axes[2,1].text(0.02, 0.03, 'VAE + LDA, '+r'$\mathcal{D}_{te}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[2, 1].transAxes,
            color='black', fontsize=20)

    # axes[0].set_title('VAE features', fontsize=10)
    # axes[1].set_title('LDA features', fontsize=10)
    # axes[2].set_title('VAE + LDA features', fontsize=10)
    plt.tight_layout()
    fig.savefig(folder + 'PCA_all_' + plot_name + '.png')
    fig.savefig(folder + 'PCA_all_' + plot_name + '.pdf')


def plot_3_tsne_new_trte(tr_data, te_data, y_train, y_test, folder, plot_name='self'):

    if len(y_train.shape) == 2:
        y = y_train[:, 1]
    else:
        y = y_train

    if len(y_test.shape) == 2:
        y_te = y_test[:, 1]
    else:
        y_te = y_test

    (tr_data1, tr_data2, tr_data3) = tr_data
    (te_data1, te_data2, te_data3) = te_data

    tr_pca1_data1, tr_pca2_data1 = make_tSNE(tr_data1, tsne_perplexity=40, n_components=2)
    tr_pca1_data2, tr_pca2_data2 = make_tSNE(tr_data2, tsne_perplexity=40, n_components=2)
    tr_pca1_data3, tr_pca2_data3 = make_tSNE(tr_data3, tsne_perplexity=40, n_components=2)

    te_pca1_data1, te_pca2_data1 = make_tSNE(te_data1, tsne_perplexity=40, n_components=2)
    te_pca1_data2, te_pca2_data2 = make_tSNE(te_data2, tsne_perplexity=40, n_components=2)
    te_pca1_data3, te_pca2_data3 = make_tSNE(te_data3, tsne_perplexity=40, n_components=2)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
    axes[0, 0].scatter(tr_pca1_data1[list(np.where(y == 0)[0])], tr_pca2_data1[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0, 0].scatter(tr_pca1_data1[list(np.where(y == 1)[0])], tr_pca2_data1[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[1, 0].scatter(tr_pca1_data2[list(np.where(y == 0)[0])], tr_pca2_data2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1, 0].scatter(tr_pca1_data2[list(np.where(y == 1)[0])], tr_pca2_data2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[2, 0].scatter(tr_pca1_data3[list(np.where(y == 0)[0])], tr_pca2_data3[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2, 0].scatter(tr_pca1_data3[list(np.where(y == 1)[0])], tr_pca2_data3[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[0, 1].scatter(te_pca1_data1[list(np.where(y_te == 0)[0])], te_pca2_data1[list(np.where(y_te == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0, 1].scatter(te_pca1_data1[list(np.where(y_te == 1)[0])], te_pca2_data1[list(np.where(y_te == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[1, 1].scatter(te_pca1_data2[list(np.where(y_te == 0)[0])], te_pca2_data2[list(np.where(y_te == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1, 1].scatter(te_pca1_data2[list(np.where(y_te == 1)[0])], te_pca2_data2[list(np.where(y_te == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[2, 1].scatter(te_pca1_data3[list(np.where(y_te == 0)[0])], te_pca2_data3[list(np.where(y_te == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2, 1].scatter(te_pca1_data3[list(np.where(y_te == 1)[0])], te_pca2_data3[list(np.where(y_te == 1)[0])], s=5, alpha=0.5, label='fake')

    axes[0, 0].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[1, 0].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[2, 0].legend(loc='upper right', prop={'size': 14}, markerscale=6)

    axes[0, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[1, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)
    axes[2, 1].legend(loc='upper right', prop={'size': 14}, markerscale=6)

    axes[0, 0].set_ylabel('tSNE2', fontsize=14)
    axes[1, 0].set_ylabel('tSNE2', fontsize=14)
    axes[2, 0].set_ylabel('tSNE2', fontsize=14)
    axes[2, 0].set_xlabel('tSNE1', fontsize=14)
    axes[2, 1].set_xlabel('tSNE1', fontsize=14)

    axes[0,0].text(0.02, 0.03, 'VAE, '+r'$\mathcal{D}_{tr}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[0, 0].transAxes,
            color='black', fontsize=20)
    axes[1,0].text(0.02, 0.03, 'LDA, '+r'$\mathcal{D}_{tr}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[1, 0].transAxes,
            color='black', fontsize=20)
    axes[2,0].text(0.02, 0.03, 'VAE + LDA, '+r'$\mathcal{D}_{tr}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[2, 0].transAxes,
            color='black', fontsize=20)

    axes[0,1].text(0.02, 0.03, 'VAE, '+r'$\mathcal{D}_{te}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[0, 1].transAxes,
            color='black', fontsize=20)
    axes[1,1].text(0.02, 0.03, 'LDA, '+r'$\mathcal{D}_{te}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[1, 1].transAxes,
            color='black', fontsize=20)
    axes[2,1].text(0.02, 0.03, 'VAE + LDA, '+r'$\mathcal{D}_{te}$',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[2, 1].transAxes,
            color='black', fontsize=20)

    plt.tight_layout()
    fig.savefig(folder + 'tSNE_all_' + plot_name + '.png')
    fig.savefig(folder + 'tSNE_all_' + plot_name + '.pdf')


def plot_3_tsne(data1, data2, data3, y_train, folder, plot_name='self'):
    # y = y_train[:, 1]
    y = y_train
    pca1_data1, pca2_data1 = make_tSNE(data1, tsne_perplexity=40, n_components=2)
    pca1_data2, pca2_data2 = make_tSNE(data2, tsne_perplexity=40, n_components=2)
    pca1_data3, pca2_data3 = make_tSNE(data3, tsne_perplexity=40, n_components=2)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
    axes[0].scatter(pca1_data1[list(np.where(y == 0)[0])], pca2_data1[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0].scatter(pca1_data1[list(np.where(y == 1)[0])], pca2_data1[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[1].scatter(pca1_data2[list(np.where(y == 0)[0])], pca2_data2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1].scatter(pca1_data2[list(np.where(y == 1)[0])], pca2_data2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[2].scatter(pca1_data3[list(np.where(y == 0)[0])], pca2_data3[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2].scatter(pca1_data3[list(np.where(y == 1)[0])], pca2_data3[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    axes[2].legend(loc='upper right')
    axes[0].set_ylabel('tSNE2', fontsize=8)
    axes[1].set_ylabel('tSNE2', fontsize=8)
    axes[2].set_xlabel('tSNE1', fontsize=8)
    axes[2].set_ylabel('tSNE2', fontsize=8)
    axes[0].set_title('VAE features', fontsize=10)
    axes[1].set_title('LDA features', fontsize=10)
    axes[2].set_title('VAE + LDA features', fontsize=10)
    plt.tight_layout()
    fig.savefig(folder + 'tSNE_all_' + plot_name + '.png')
    fig.savefig(folder + 'tSNE_all_' + plot_name + '.pdf')


def plot_all_pca_tsne_datasets(top_folder, plots_path, dataset_name):
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if dataset_name in dname]
    for run in runs:
        model_name = run.split('/')[1]
        ntopics = int(model_name.split('_')[model_name.split('_').index('topics')+1])
        nlatent = int(model_name.split('_')[model_name.split('_').index('dim')+1])
        if 'w2v' in model_name.split('_'):
            w2v = int(model_name.split('_')[model_name.split('_').index('w2v')+1])
        else:
            w2v = nlatent
        features_path = run + '/features/'
        vae_features_train = np.load(features_path + 'vae_train.npy')
        vae_features_test = np.load(features_path + 'vae_test.npy')
        lda_features_train = np.load(features_path + 'lda_tf_train.npy')
        lda_features_test = np.load(features_path + 'lda_tf_test.npy')
        data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
        data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
        y_train = np.load(run + '/train_label.npy')
        y_test = np.load(run + '/test_label.npy')

        plot_name = dataset_name + '_n_topis_' + str(ntopics) +'_latent_dim_' + str(nlatent) + '_w2v_' + str(w2v) + '_train'
        plot_3_tsne(vae_features_train, lda_features_train, data3_train, y_train, plots_path, plot_name=plot_name)
        plot_3_pca(vae_features_train, lda_features_train, data3_train, y_train, plots_path, plot_name=plot_name)

        plot_name = dataset_name + '_n_topis_' + str(ntopics) +'_latent_dim_' + str(nlatent) + '_w2v_' + str(w2v) + '_test'
        plot_3_tsne(vae_features_test, lda_features_test, data3_test, y_test, plots_path, plot_name=plot_name)
        plot_3_pca(vae_features_test, lda_features_test, data3_test, y_test, plots_path, plot_name=plot_name)


def plot_all_pca_tsne_config(top_folder, plots_path, config='topics_10_latent_dim_32_w2v_32'):
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if config in dname and 'CLOSE' not in
            dname and '_ep_10' not in dname]

    for run in runs:
        model_name = run.split('/')[1]
        dataset_name = model_name.split('_')[0]
        ntopics = int(model_name.split('_')[model_name.split('_').index('topics')+1])
        nlatent = int(model_name.split('_')[model_name.split('_').index('dim')+1])
        if 'w2v' in model_name.split('_'):
            w2v = int(model_name.split('_')[model_name.split('_').index('w2v')+1])
        else:
            w2v = nlatent
        features_path = run + '/features/'
        vae_features_train = np.load(features_path + 'vae_train.npy')
        vae_features_test = np.load(features_path + 'vae_test.npy')
        lda_features_train = np.load(features_path + 'lda_tf_train.npy')
        lda_features_test = np.load(features_path + 'lda_tf_test.npy')
        data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
        data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
        y_train = np.load(run + '/train_label.npy')
        y_test = np.load(run + '/test_label.npy')

        plot_name = dataset_name + '_n_topis_' + str(ntopics) +'_latent_dim_' + str(nlatent) + '_w2v_' + str(w2v) + '_train_new'
        plot_3_pca_new(vae_features_train, lda_features_train, data3_train, y_train, plots_path, plot_name=plot_name)
        plot_3_tsne(vae_features_train, lda_features_train, data3_train, y_train, plots_path, plot_name=plot_name)

        plot_name = dataset_name + '_n_topis_' + str(ntopics) +'_latent_dim_' + str(nlatent) + '_w2v_' + str(w2v) + '_test_new'
        plot_3_tsne(vae_features_test, lda_features_test, data3_test, y_test, plots_path, plot_name=plot_name)
        plot_3_pca_new(vae_features_test, lda_features_test, data3_test, y_test, plots_path, plot_name=plot_name)

        tr_data = (vae_features_train, lda_features_train, data3_train)
        te_data = (vae_features_test, lda_features_test, data3_test)
        plot_name = dataset_name + '_n_topis_' + str(ntopics) +'_latent_dim_' + str(nlatent) + '_w2v_' + str(w2v) + '_all_new'
        plot_3_pca_new_trte(tr_data, te_data, y_train, y_test, plots_path, plot_name=plot_name)
        plot_3_tsne_new_trte(tr_data, te_data, y_train, y_test, plots_path, plot_name=plot_name)


def plot_pca_tsne(run_info, over_sampled=False):

    main_folder = run_info['main_folder']
    dataset_name = run_info['dataset_name']
    latent_dim = run_info['latent_dim']
    word2vec_dim = run_info['word2vec_dim']
    n_topics = run_info['n_topics']

    features_path = main_folder + 'features/'
    if over_sampled:
        vae_features_train = np.load(features_path + 'vae_ext_train.npy')
        vae_features_test = np.load(features_path + 'vae_ext_test.npy')
        y_train = np.load(main_folder + 'train_ext_label.npy')
        y_test = np.load(main_folder + 'test_ext_label.npy')
    else:
        vae_features_train = np.load(features_path + 'vae_train.npy')
        vae_features_test = np.load(features_path + 'vae_test.npy')
        y_train = np.load(main_folder + 'train_label.npy')#[:, 1]
        y_test = np.load(main_folder + 'test_label.npy')#[:, 1]

    lda_features_train = np.load(features_path + 'lda_tfidf_train.npy')
    lda_features_test = np.load(features_path + 'lda_tfidf_test.npy')
    data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
    data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)

    plot_name_tr = dataset_name + '_n_topis_' + str(n_topics) +'_latent_dim_' + str(latent_dim) + '_w2v_' + \
                   str(word2vec_dim) + '_train'
    if over_sampled:
        plot_name_tr = plot_name_tr + '_ext'

    plot_3_tsne(vae_features_train, lda_features_train, data3_train, y_train, main_folder, plot_name=plot_name_tr)

    plot_3_pca(vae_features_train, lda_features_train, data3_train, y_train, main_folder, plot_name=plot_name_tr)

    plot_name_te = dataset_name + '_n_topis_' + str(n_topics) +'_latent_dim_' + str(latent_dim) + '_w2v_' + \
                   str(word2vec_dim) + '_test'
    if over_sampled:
        plot_name_te = plot_name_te + '_ext'
    plot_3_tsne(vae_features_test, lda_features_test, data3_test, y_test, main_folder, plot_name=plot_name_te)
    plot_3_pca(vae_features_test, lda_features_test, data3_test, y_test, main_folder, plot_name=plot_name_te)

    tr_data = (vae_features_train, lda_features_train, data3_train)
    te_data = (vae_features_test, lda_features_test, data3_test)
    plot_3_pca_new_trte(tr_data, te_data, y_train, y_test, main_folder, plot_name=dataset_name + '_paper_PCA')
    plot_3_tsne_new_trte(tr_data, te_data, y_train, y_test, main_folder, plot_name=dataset_name + '_paper_tSNE')


def plot_accuracy_metrics(results_path, plots_path):
    runs = [os.path.join(results_path, dname) for dname in os.listdir(results_path) if '.csv' in dname and 'with_fs'
            not in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[2].split('.csv')[0]
        model_df = pd.read_csv(run)
        model_df_filtered = model_df[model_df['Metric'] != 'FNR']
        model_df_filtered = model_df_filtered[model_df_filtered['Metric'] != 'FPR']
        model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'FPR']
        model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'SVM_nonlinear']
        model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'linear disriminat Analysis']
        model_df_filtered.loc[model_df_filtered['Classifier'] == 'SVM_linear', 'Classifier'] = 'SVM (linear)'

        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()
        sns_plot = sns.relplot(data=model_df_filtered, x="Metric", y="Value", col="Feature", hue="Classifier", style="Dataset", aspect=.45)
        leg = sns_plot._legend
        leg.set_bbox_to_anchor([0.99, 0.35])
        plt.tight_layout()
        sns_plot.savefig(plots_path + 'Accuracy_' + model_name + '.png')
        sns_plot.savefig(plots_path + 'Accuracy_' + model_name + '.pdf')
        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()


def FNR_FPR_table_latex(results_path, plots_path):
    runs = [os.path.join(results_path, dname) for dname in os.listdir(results_path) if
            '.csv' in dname and 'with_fs'
            not in dname]
    for run in runs:
        print(run)
        model_df = pd.read_csv(run)
        model_df_filtered2 = model_df[model_df['Metric'] != 'Accuracy']
        model_df_filtered2 = model_df_filtered2[model_df_filtered2['Metric'] != 'Precision']
        model_df_filtered2 = model_df_filtered2[model_df_filtered2['Metric'] != 'Recall']
        model_df_filtered2 = model_df_filtered2[model_df_filtered2['Metric'] != 'F-score']
        model_df_filtered2 = model_df_filtered2[model_df_filtered2['Classifier'] != 'SVM_nonlinear']
        model_df_filtered2 = model_df_filtered2[model_df_filtered2['Classifier'] != 'linear disriminat Analysis']
        model_df_filtered2.loc[model_df_filtered2['Classifier'] == 'SVM_linear', 'Classifier'] = 'SVM (linear)'
        model_df_filtered2 = model_df_filtered2.reset_index(drop=True)
        new_df = pd.DataFrame(data=0, columns=['Classifier', 'Metric', 'VAE_Train', 'VAE_Test', 'LDA_Train', 'LDA_Test',
                                               'VAE + LDA_Train', 'VAE + LDA_Test'], index=range(12))
        # classifiers = sorted(list(set(model_df_filtered2['Classifier'])))
        classifiers = ['SVM (linear)', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP', 'KNN (K = 3)']
        metrics = list(set(model_df_filtered2['Metric']))
        c = 0
        for i in range(len(classifiers)):
            for j in range(len(metrics)):
                new_df.loc[c, 'Classifier'] = classifiers[i]
                new_df.loc[c, 'Metric'] = metrics[j]
                c += 1

        for i in range(len(model_df_filtered2)):
            classifier = model_df_filtered2.loc[i, 'Classifier']
            value = model_df_filtered2.loc[i, 'Value']
            feature = model_df_filtered2.loc[i, 'Feature']
            dataset = model_df_filtered2.loc[i, 'Dataset']
            metric = model_df_filtered2.loc[i, 'Metric']
            new_df.loc[(new_df['Classifier'] == classifier) & (new_df['Metric'] == metric), feature + '_' + dataset] = round(value, 4)

        print(new_df.to_latex(index=False))


def plot_classifiers_result(run_info, over_sampled=False):
    main_folder = run_info['main_folder']
    if over_sampled:
        model_df = pd.read_csv(main_folder + 'Classifiers_over_sampled.csv')
        plot_name = 'Metrics_classifiers_over_sampled'
    else:
        model_df = pd.read_csv(main_folder + 'Classifiers.csv')
        plot_name = 'Metrics_classifiers'

    model_df_filtered = model_df[model_df['Metric'] != 'FNR']
    model_df_filtered = model_df_filtered[model_df_filtered['Metric'] != 'FPR']
    model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'SVM_nonlinear']
    model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'linear disriminat Analysis']
    model_df_filtered.loc[model_df_filtered['Classifier'] == 'SVM_linear', 'Classifier'] = 'SVM (linear)'

    plt.close('all')
    plt.close()
    plt.clf()
    plt.cla()
    sns_plot = sns.relplot(data=model_df_filtered, x="Metric", y="Value", col="Feature", hue="Classifier",
                           style="Dataset", aspect=.45)
    leg = sns_plot._legend
    leg.set_bbox_to_anchor([1.25, 0.4])
    plt.tight_layout()
    sns_plot.savefig(main_folder + plot_name + '.png')
    sns_plot.savefig(main_folder + plot_name + '.pdf')
    plt.close('all')
    plt.close()
    plt.clf()
    plt.cla()


def plot_accuracy_metrics2(results_path, plots_path):
    runs = [os.path.join(results_path, dname) for dname in os.listdir(results_path) if '.csv' in dname and 'with_fs'
            not in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[2].split('.csv')[0]
        model_df = pd.read_csv(run)
        model_df_filtered = model_df[model_df['Metric'] != 'FNR']
        model_df_filtered = model_df_filtered[model_df['Metric'] != 'FPR']
        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()
        sns_plot = sns.relplot(data=model_df_filtered, x="Classifier", y="Value", col="Feature", hue="Metric", style="Dataset")
        plt.tight_layout()
        sns_plot.savefig(plots_path + 'Accuracy_2_' + model_name + '.png')
        sns_plot.savefig(plots_path + 'Accuracy_2_' + model_name + '.pdf')
        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()


def plot_accuracy_with_fs(results_path, plots_path):
    runs = [os.path.join(results_path, dname) for dname in os.listdir(results_path) if '.csv' in dname and 'with_fs' in
            dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[2].split('.csv')[0]
        model_df = pd.read_csv(run)
        model_df_filtered = model_df[model_df['Metric'] != 'FNR']
        model_df_filtered = model_df_filtered[model_df_filtered['Metric'] != 'FPR']
        model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'SVM (nonlinear)']
        model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'linear disriminat Analysis']
        model_df_filtered = model_df_filtered[model_df_filtered['Dataset'] != 'Train']
        model_df_filtered = model_df_filtered[model_df_filtered['Metric'] == 'Accuracy']
        # model_df_filtered.loc[model_df_filtered['Classifier'] == 'SVM_linear', 'Classifier'] = 'SVM (linear)'

        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()

        # sns_plot = sns.relplot(data=model_df_filtered, x="# Selected Features", y="Value", row="Feature",
        #                        col="Metric", hue="Classifier", style="Dataset", aspect=1.0, kind='line')
        sns_plot = sns.relplot(data=model_df_filtered, x="# Selected Features", y="Value", col="Feature",
                               hue="Classifier", kind="line")
        # sns_plot = sns.relplot(data=model_df_filtered, x="# Selected Features", y="Value", col="Classifier",
        #                        hue="Feature", style="Dataset", aspect=1.0, kind='line', col_wrap=3)

        leg = sns_plot._legend
        leg.set_bbox_to_anchor([1, 0.45])
        plt.tight_layout()
        sns_plot.savefig(plots_path + 'Accuracy_' + model_name + '.png')
        sns_plot.savefig(plots_path + 'Accuracy_' + model_name + '.pdf')
        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()


def plot_classifiers_with_fs_result(run_info, fs_name, over_sampled=False):
    main_folder = run_info['main_folder']
    if over_sampled:
        model_df = pd.read_csv(main_folder + 'Classifiers_with_' + fs_name + '_fs_over_sampled.csv')
        plot_name = 'Metrics_classifiers_with_' + fs_name + '_fs_over_sampled'

    else:
        model_df = pd.read_csv(main_folder + 'Classifiers_with_' + fs_name + '_fs.csv')
        plot_name = 'Metrics_classifiers_with_' + fs_name + '_fs'

    model_df_filtered = model_df[model_df['Metric'] != 'FNR']
    model_df_filtered = model_df_filtered[model_df_filtered['Metric'] != 'FPR']
    model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'SVM (nonlinear)']
    model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'linear disriminat Analysis']
    model_df_filtered = model_df_filtered[model_df_filtered['Dataset'] != 'Train']
    model_df_filtered = model_df_filtered[model_df_filtered['Metric'] == 'Accuracy']

    plt.close('all')
    plt.close()
    plt.clf()
    plt.cla()
    # sns_plot = sns.relplot(data=model_df_filtered, x="# Selected Features", y="Value", row="Feature",
    #                        col="Metric", hue="Classifier", style="Dataset", aspect=1.0, kind='line')
    # sns.set_style("whitegrid")
    # sns.set_style("white")
    # sns.set_style(style=None)
    # sns.set_style("whitegrid", {'axes.grid': False})
    # sns.set_style("ticks")
    sns_plot = sns.relplot(data=model_df_filtered, x="# Selected Features", y="Value", col="Feature",
                           hue="Classifier", kind="line", aspect=.50)

    # sns_plot.set_style("whitegrid")
    # sns_plot = sns.relplot(data=model_df_filtered, x="# Selected Features", y="Value", col="Classifier",
    #                        hue="Feature", style="Dataset", aspect=1.0, kind='line', col_wrap=3)
    leg = sns_plot._legend
    leg.set_bbox_to_anchor([1, 0.45])
    plt.tight_layout()
    sns_plot.savefig(main_folder + plot_name + '.png')
    sns_plot.savefig(main_folder + plot_name + '.pdf')
    plt.close('all')
    plt.close()
    plt.clf()
    plt.cla()

def plot_WordClouds(data_tr, data_te, data_2_true, data_2_false, main_folder):
    wordcloud0 = WordCloud(background_color='white').generate(' '.join(data_tr['text']))
    wordcloud1 = WordCloud(background_color='white').generate(' '.join(data_te['text']))
    wordcloud2 = WordCloud(background_color='white').generate(' '.join(data_2_true['text']))
    wordcloud3 = WordCloud(background_color='white').generate(' '.join(data_2_false['text']))

    plt.figure(figsize=(8, 8))
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.025, hspace=0.005)  # set the spacing between axes.

    i = 0
    ax0 = plt.subplot(gs1[i])

    plt.imshow(wordcloud0, interpolation="bilinear")
    plt.axis('off')

    i = 1
    ax1 = plt.subplot(gs1[i])

    plt.imshow(wordcloud1, interpolation="bilinear")
    plt.axis('off')
    i = 2

    ax2 = plt.subplot(gs1[i])
    plt.imshow(wordcloud2, interpolation="bilinear")
    plt.axis('off')

    i = 3
    ax3 = plt.subplot(gs1[i])
    plt.imshow(wordcloud3, interpolation="bilinear")
    plt.axis('off')
    plt.savefig(main_folder + '/WorldCloud.png')


def plot_WordClouds_twitter(data_df_all, main_folder):
    # data_df_all = filtered_data_df_all
    real_no = data_df_all[data_df_all['label'] == 'real']
    fake_no = data_df_all[data_df_all['label'] == 'fake']

    wordcloud0 = WordCloud(background_color='white').generate(' '.join(real_no['text']))
    wordcloud1 = WordCloud(background_color='white').generate(' '.join(fake_no['text']))

    plt.figure(figsize=(16, 8))
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.025, hspace=0.005)  # set the spacing between axes.

    i = 0
    ax0 = plt.subplot(gs1[i])

    plt.imshow(wordcloud0, interpolation="bilinear")
    plt.axis('off')

    i = 1
    ax1 = plt.subplot(gs1[i])

    plt.imshow(wordcloud1, interpolation="bilinear")
    plt.axis('off')

    plt.savefig(main_folder + '/WorldCloud_twitter.png')
    plt.savefig(main_folder + '/WorldCloud_twitter.pdf')


def plot_WordClouds_covid(data_df_all, main_folder):

    real_no = data_df_all[data_df_all['label'] == 'real']
    fake_no = data_df_all[data_df_all['label'] == 'fake']

    wordcloud0 = WordCloud(background_color='white').generate(' '.join(real_no['headlines']))
    wordcloud1 = WordCloud(background_color='white').generate(' '.join(fake_no['headlines']))

    plt.figure(figsize=(16, 8))
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.025, hspace=0.005)  # set the spacing between axes.

    i = 0
    ax0 = plt.subplot(gs1[i])

    plt.imshow(wordcloud0, interpolation="bilinear")
    plt.axis('off')

    i = 1
    ax1 = plt.subplot(gs1[i])

    plt.imshow(wordcloud1, interpolation="bilinear")
    plt.axis('off')

    plt.savefig(main_folder + '/WorldCloud_covid.png')
    plt.savefig(main_folder + '/WorldCloud_covid.pdf')

