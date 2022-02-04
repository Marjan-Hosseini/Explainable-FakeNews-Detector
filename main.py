from LVAE.lvae import *
from LVAE.lda import *
from LVAE.preprocessing import *
from plots import *
from classifiers import *
from clustering import *
import logging
sys.path.append('../')


def vae_experiment(run_info, top_folder):

    main_folder = run_info['main_folder']

    target_column = run_info['target_column']
    dataset_name = run_info['dataset_name']

    word2vec_dim = run_info['word2vec_dim']
    this_data_folder = top_folder + 'data/' + dataset_name + '/'

    # x_train = pd.read_csv(this_data_folder + 'x_train_' + str(word2vec_dim) + '.csv')
    # x_test = pd.read_csv(this_data_folder + 'x_test_' + str(word2vec_dim) + '.csv')

    if not os.path.exists(main_folder + 'model_weights.hdf5') or \
            not os.path.exists(main_folder + 'model_history.pickle'):
        train_vae(run_info, top_folder)

    # lvae feature extraction run_info, top_folder, filtered_data_df, data_name, col_label
    if not os.path.exists(main_folder+'features/'):
        os.makedirs(main_folder+'features/')

    if not os.path.exists(main_folder + 'features/vae_train.npy') or \
            not os.path.exists(main_folder + 'features/train_label.npy'):

        # vae_features_tr, output_tr = extract_features(run_info, top_folder, x_train, data_name='train',
        #                                               col_label=target_column, output_label='label')

        vae_features_tr, output_tr = new_extract_features(run_info, top_folder, data_name='train')

        np.save(main_folder + 'features/vae_train', vae_features_tr)
        np.save(main_folder + 'train_label', output_tr)

        # ext_x_train, ext_y_train = extract_features_over_sample(run_info, top_folder, x_train, data_name='train',
        #                                                   col_label=target_column, output_label='label')
        # np.save(main_folder + 'features/vae_ext_train', ext_x_train)
        # np.save(main_folder + 'train_ext_label', ext_y_train)

    if not os.path.exists(main_folder + 'features/vae_test.npy') or \
            not os.path.exists(main_folder + 'test_label.npy'):

        # vae_features_te, output_te = extract_features(run_info, top_folder, x_test, data_name='test',
        #                                               col_label=target_column, output_label='label')

        vae_features_te, output_te = new_extract_features(run_info, top_folder, data_name='test')

        np.save(main_folder + 'features/vae_test', vae_features_te)
        np.save(main_folder + 'test_label', output_te)

        # ext_x_test, ext_y_test = extract_features_over_sample(run_info, top_folder, x_test, data_name='test',
        #                                                   col_label=target_column, output_label='label')
        # np.save(main_folder + 'features/vae_ext_test', ext_x_test)
        # np.save(main_folder + 'test_ext_label', ext_y_test)

    # prediction
    y_tr, y_pred_tr = test_vae(run_info, top_folder, phase='train')
    y_te, y_pred_te = test_vae(run_info, top_folder, phase='test')

    tr_metrics = compute_accuracy_metrics(y_tr, y_pred_tr)
    te_metrics = compute_accuracy_metrics(y_te, y_pred_te)

    accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr_metrics
    accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te_metrics

    mvae_results = pd.DataFrame(data=0, columns=['Value', 'Metric', 'Data'], index=range(12))
    mvae_results.loc[0:5, 'Data'] = 'train'
    mvae_results.loc[6:12, 'Data'] = 'test'
    mvae_results.loc[0:5, 'Value'] = accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr
    mvae_results.loc[6:12, 'Value'] = accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te
    mvae_results.loc[[0, 6], 'Metric'] = 'Accuracy'
    mvae_results.loc[[1, 7], 'Metric'] = 'Precision'
    mvae_results.loc[[2, 8], 'Metric'] = 'Recall'
    mvae_results.loc[[3, 9], 'Metric'] = 'F-Score'
    mvae_results.loc[[4, 10], 'Metric'] = 'FPR'
    mvae_results.loc[[5, 11], 'Metric'] = 'FNR'
    mvae_results.to_csv(main_folder + 'model_classifier.csv')


def tvae_experiment(run_info, top_folder):

    main_folder = run_info['main_folder']

    target_column = run_info['target_column']
    dataset_name = run_info['dataset_name']
    # sequence_length = run_info['sequence_length']
    # reg_lambda = run_info['reg_lambda']
    # fnd_lambda = run_info['fnd_lambda']
    word2vec_dim = run_info['word2vec_dim']
    this_data_folder = top_folder + 'data/' + dataset_name + '/'

    x_train = pd.read_csv(this_data_folder + 'x_train_' + str(word2vec_dim) + '.csv')
    x_test = pd.read_csv(this_data_folder + 'x_test_' + str(word2vec_dim) + '.csv')

    if not os.path.exists(main_folder + 'tvae_model_weights.hdf5') or \
            not os.path.exists(main_folder + 'tvae_model_history.pickle'):
        train_vae(run_info, top_folder)

    # lvae feature extraction run_info, top_folder, filtered_data_df, data_name, col_label
    if not os.path.exists(main_folder+'features/'):
        os.makedirs(main_folder+'features/')

    if not os.path.exists(main_folder + 'features/vae_train.npy') or \
            not os.path.exists(main_folder + 'features/train_label.npy'):

        vae_features_tr, output_tr = extract_features(run_info, top_folder, x_train, data_name='train',
                                                      col_label=target_column, output_label='label')

        np.save(main_folder + 'features/vae_train', vae_features_tr)
        np.save(main_folder + 'train_label', output_tr)

    if not os.path.exists(main_folder + 'features/vae_test.npy') or \
            not os.path.exists(main_folder + 'features/test_label.npy'):

        vae_features_te, output_te = extract_features(run_info, top_folder, x_test, data_name='test',
                                                      col_label=target_column, output_label='label')
        np.save(main_folder + 'features/vae_test', vae_features_te)
        np.save(main_folder + 'test_label', output_te)

#
# def lda_experiment(run_info, top_folder):
#     main_folder = run_info['main_folder']
#     if not os.path.exists(main_folder):
#         os.makedirs(main_folder)
#     n_topics = run_info['n_topics']
#     n_top_words = run_info['n_top_words']
#     n_features = run_info['n_features']
#     n_iter = run_info['n_iter']
#     target_column = run_info['target_column']
#     dataset_name = run_info['dataset_name']
#     word2vec_dim = run_info['word2vec_dim']
#     print('LDA experiment with', n_topics, 'topics:')
#     with open(main_folder + 'run_info.pickle', 'wb') as handle:
#         pkl.dump(run_info, handle)
#
#     this_data_folder = top_folder + 'data/' + dataset_name + '/'
#
#     x_train = pd.read_csv(this_data_folder + 'x_train_' + str(word2vec_dim) + '.csv')
#     x_test = pd.read_csv(this_data_folder + 'x_test_' + str(word2vec_dim) + '.csv')
#
#     # train lda
#     data_df_all = x_train.append(x_test, ignore_index=True)
#
#     lda_tf, tf_vectorizer = LDA_train_tf(main_folder, n_topics, data_df_all[target_column], n_top_words,
#                                          n_features, n_iter=n_iter)
#
#     # lda_tfidf, tfidf_vectorizer = LDA_train_tf_idf(main_folder, n_topics, data_df_all[target_column],
#     #                                                n_top_words, n_features, n_iter=n_iter)
#
#     # LDA
#     # tf features
#     if not os.path.exists(main_folder+'features/'):
#         os.makedirs(main_folder+'features/')
#     lda_features_tf_tr = extract_lda_tf_features(main_folder, x_train[target_column], data_name='train')
#     lda_features_tf_te = extract_lda_tf_features(main_folder, x_test[target_column], data_name='test')
#     np.save(main_folder + 'features/lda_tf_train', lda_features_tf_tr)
#     np.save(main_folder + 'features/lda_tf_test', lda_features_tf_te)
#
#     lda_features_tfidf_tr, lda_features_tfidf_te = LDA_train_tf_idf_extract_features(main_folder, n_topics,
#                                                                                      data_df_all[target_column],
#                                                                                      x_train[target_column],
#                                                                                      x_test[target_column],
#                                                                                      n_top_words, n_features,
#                                                                                      n_iter=n_iter)
#
#     # # tfidf features
#     # lda_features_tfidf_tr = extract_lda_tfidf_features(main_folder, x_train[target_column], data_name='train')
#     # lda_features_tfidf_te = extract_lda_tfidf_features(main_folder, x_test[target_column], data_name='test')
#     np.save(main_folder + 'features/lda_tfidf_train', lda_features_tfidf_tr)
#     np.save(main_folder + 'features/lda_tfidf_test', lda_features_tfidf_te)
#
#
# def new_lda_experiment(run_info, top_folder):
#     main_folder = run_info['main_folder']
#     if not os.path.exists(main_folder):
#         os.makedirs(main_folder)
#     n_topics = run_info['n_topics']
#     n_top_words = run_info['n_top_words']
#     n_features = run_info['n_features']
#     n_iter = run_info['n_iter']
#     target_column = run_info['target_column']
#     dataset_name = run_info['dataset_name']
#     word2vec_dim = run_info['word2vec_dim']
#     print('LDA experiment with', n_topics, 'topics:')
#     with open(main_folder + 'run_info.pickle', 'wb') as handle:
#         pkl.dump(run_info, handle)
#
#     this_data_folder = top_folder + 'data/' + dataset_name + '/'
#
#     x_train = np.load(this_data_folder + 'train_text_' + str(word2vec_dim) + '.npy')
#     x_test = np.load(this_data_folder + 'test_text_' + str(word2vec_dim) + '.npy')
#
#     x_df_all = np.concatenate([x_train, x_test], axis=0)
#     new_df = []
#     for i in range(x_df_all.shape[0]):
#         temp = []
#         for j in range(x_df_all.shape[1]):
#             temp.append(str(x_df_all[i, j]))
#         xx = ' '.join(temp)
#         new_df.append(xx)
#
#     # new_df = list(data_df_all['headlines'])
#     # new_df = list(data_df_all['title'])
#
#     tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, lowercase=False, max_features=n_features,
#                                        stop_words={'english'}, analyzer='word')
#     tfidf = tfidf_vectorizer.fit_transform(new_df)
#
#     with open(main_folder + 'lda_vectorizer_tfidf.pkl', 'wb') as handle:
#         pkl.dump(tfidf_vectorizer.vocabulary_, handle)
#
#     lda_2 = LatentDirichletAllocation(n_components=n_topics, max_iter=n_iter, learning_method='batch',
#                                       learning_offset=50., random_state=0, verbose=1)
#     lda_2.fit(tfidf)
#
#     # with open(main_folder + 'lda_model_tfidf.pkl', 'rb') as handle:
#     #     lda_2 = pkl.load(handle)
#
#     # with open(main_folder + 'lda_vectorizer_tfidf.pkl', 'rb') as handle:
#     #     tfidf_vectorizer = pkl.load(handle)
#
#     with open(main_folder + 'lda_model_tfidf.pkl', 'wb') as handle:
#         pkl.dump(lda_2, handle)
#
#     # tf_feature_names = tfidf_vectorizer.get_feature_names()
#     # if n_topics <= 10:
#     #     plot_name = main_folder + dataset_name + '_lda_topic_tf_' + str(n_topics)
#     #     plot_top_words(lda_2, tf_feature_names, n_top_words, plot_name)
#
#     print('Extracting LDA with tf_idf features ...')
#
#     tr_df = []
#     for i in range(x_train.shape[0]):
#         temp = []
#         for j in range(x_train.shape[1]):
#             temp.append(str(x_train[i, j]))
#         xx = ' '.join(temp)
#         tr_df.append(xx)
#
#     X_train_vec = tfidf_vectorizer.transform(tr_df)
#     X_train_topics = lda_2.transform(X_train_vec)
#
#     te_df = []
#     for i in range(x_test.shape[0]):
#         temp = []
#         for j in range(x_test.shape[1]):
#             temp.append(str(x_test[i, j]))
#         xx = ' '.join(temp)
#         te_df.append(xx)
#     X_test_vec = tfidf_vectorizer.transform(te_df)
#     X_test_topics = lda_2.transform(X_test_vec)
#
#     np.save(main_folder + 'features/lda_tfidf_train', X_train_topics)
#     np.save(main_folder + 'features/lda_tfidf_test', X_test_topics)


def n_topic_cross_validation(corpus, tr_corpus, te_corpus, d, main_folder):
    coherence_u_mass_all = []
    coherence_u_mass_tr = []
    coherence_u_mass_test = []
    for ntop in [2, 5, 10, 32, 50, 64, 100]:
        print(ntop)
        Lda = gensim.models.ldamodel.LdaModel
        ldamodel = Lda(corpus, num_topics=ntop, id2word=d, passes=200, iterations=1000, chunksize=10000, eval_every=10)
        temp_file = os.path.join(main_folder, 'ldamodels/', "lda_model_" + str(ntop))

        with open(os.path.join(main_folder, 'ldamodels/', "lda_model_" + str(ntop) + '_topic_words.txt'), 'w') as f:
            for top in range(ntop):
                topw = ldamodel.get_topic_terms(topicid=top, topn=10)
                topwords = [(d[i[0]], str(i[1])) for i in topw]
                f.write("%s\n" % topwords)

        ldamodel.save(temp_file)
        # ldamodel = gensim.models.ldamodel.LdaModel.load(main_folder + "lda_model_" + str(k))
        cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, corpus=corpus, dictionary=d,
                                                         coherence='u_mass')
        coherence_u_mass_all.append((ntop, cm.get_coherence()))

        cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, corpus=tr_corpus, dictionary=d,
                                                         coherence='u_mass')
        coherence_u_mass_tr.append((ntop, cm.get_coherence()))

        cm2 = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, corpus=te_corpus, dictionary=d,
                                                          coherence='u_mass')
        coherence_u_mass_test.append((ntop, cm2.get_coherence()))

    with open(os.path.join(main_folder, 'ldamodels/', 'coherence_score_all.txt'), 'w') as f:
        for elem in coherence_u_mass_all:
            f.write("%s\n" % str(elem))

    with open(os.path.join(main_folder, 'ldamodels/', 'coherence_score_tr.txt'), 'w') as f:
        for elem in coherence_u_mass_tr:
            f.write("%s\n" % str(elem))

    with open(os.path.join(main_folder, 'ldamodels/', 'coherence_score_te.txt'), 'w') as f:
        for elem in coherence_u_mass_test:
            f.write("%s\n" % str(elem))

    plt.clf()
    plt.plot([ee[0] for ee in coherence_u_mass_tr], [ee[1] for ee in coherence_u_mass_tr], 'x-', label='Train')
    plt.plot([ee[0] for ee in coherence_u_mass_test], [ee[1] for ee in coherence_u_mass_test], 'o-', label='Test')
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence')
    plt.xticks([ee[0] for ee in coherence_u_mass_tr])
    plt.grid('on')
    plt.legend()
    plt.savefig(main_folder + 'lda_cross_validation.png')


def new_lda_experiment_gensim(run_info, top_folder):
    main_folder = run_info['main_folder']
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    n_topics = run_info['n_topics']
    target_column = run_info['target_column']
    dataset_name = run_info['dataset_name']
    word2vec_dim = run_info['word2vec_dim']
    print('LDA experiment with', n_topics, 'topics:')

    this_data_folder = top_folder + 'data/' + dataset_name + '/'

    x_train = pd.read_csv(this_data_folder + 'x_train_' + str(word2vec_dim) + '.csv')
    x_test = pd.read_csv(this_data_folder + 'x_test_' + str(word2vec_dim) + '.csv')

    my_stopwords = list(feature_extraction.text.ENGLISH_STOP_WORDS)

    x_train[target_column] = x_train[target_column].apply(lambda x: ' '.join([w for w in x.split() if w not in my_stopwords]))
    x_test[target_column] = x_test[target_column].apply(lambda x: ' '.join([w for w in x.split() if w not in my_stopwords]))

    x_train[target_column] = x_train[target_column].apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(w) for w in x.split()]))
    x_test[target_column] = x_test[target_column].apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(w) for w in x.split()]))

    # train lda
    data_df_all = x_train.append(x_test, ignore_index=True)

    with open(this_data_folder + 'word_index_' + str(word2vec_dim) + '.pkl', 'rb') as handle:
        word_index_orig = pkl.load(handle)

    wc = 0
    word_index = {}
    for key in word_index_orig:
        if key not in my_stopwords and len(key) > 2 and key != "n't":
            word_index[key] = wc
            wc += 1

    dictionary = {}
    for k, v in word_index.items():
        dictionary[v] = k

    d = gensim.corpora.Dictionary()
    d.id2token = dictionary
    d.token2id = word_index

    documents = list(data_df_all[target_column])
    texts = [[word for word in document.split()] for document in documents]
    corpus = [[(word_index[word], txt.count(word)) for word in txt if word in word_index.keys()] for txt in texts]

    tr_documents = list(x_train[target_column])
    tr_texts = [[word for word in document.split()] for document in tr_documents]
    tr_corpus = [[(word_index[word], txt.count(word)) for word in txt if word in word_index.keys()] for txt in tr_texts]

    te_documents = list(x_test[target_column])
    te_texts = [[word for word in document.split()] for document in te_documents]
    te_corpus = [[(word_index[word], txt.count(word)) for word in txt if word in word_index.keys()] for txt in te_texts]

    model_tfidf = gensim.models.TfidfModel(corpus, normalize=True)
    corpus_tfidf = model_tfidf[corpus]
    tr_corpus_tfidf = model_tfidf[tr_corpus]
    te_corpus_tfidf = model_tfidf[te_corpus]

    logging.basicConfig(filename='gensim_qual.log',
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.INFO)

    n_topic_cross_validation(corpus_tfidf, tr_corpus_tfidf, te_corpus_tfidf, d, main_folder)
    sel = 10

    ldamodel = gensim.models.ldamodel.LdaModel.load(main_folder + "lda_model_" + str(sel))

    with open(os.path.join(main_folder, 'ldamodels/', "lda_model_" + str(sel) + '_topic_words_freq.txt'), 'w') as f:
        for top in range(sel):
            topw = ldamodel.get_topic_terms(topicid=top, topn=10)
            topwords = [(d[i[0]], str(i[1])) for i in topw]
            f.write("%s\n" % topwords)

    if not os.path.exists(os.path.join(main_folder, 'features')):
        os.mkdir(os.path.join(main_folder, 'features'))

    tr_gensim = np.array(ldamodel.get_document_topics(tr_corpus_tfidf))
    lda_features_tfidf_tr = np.zeros([len(tr_gensim), n_topics])
    for i in range(len(tr_gensim)):
        for elem in tr_gensim[i]:
            lda_features_tfidf_tr[i, elem[0]] = elem[1]
    np.save(main_folder + 'features/lda_tfidf_train', lda_features_tfidf_tr)

    te_gensim = np.array(ldamodel.get_document_topics(te_corpus_tfidf))
    lda_features_tfidf_te = np.zeros([len(te_gensim), n_topics])
    for i in range(len(te_gensim)):
        for elem in te_gensim[i]:
            lda_features_tfidf_te[i, elem[0]] = elem[1]
    np.save(main_folder + 'features/lda_tfidf_test', lda_features_tfidf_te)

    tr_gensim = np.array(ldamodel.get_document_topics(tr_corpus))
    lda_features_tfidf_tr = np.zeros([len(tr_gensim), n_topics])
    for i in range(len(tr_gensim)):
        for elem in tr_gensim[i]:
            lda_features_tfidf_tr[i, elem[0]] = elem[1]
    np.save(main_folder + 'features/lda_train', lda_features_tfidf_tr)

    te_gensim = np.array(ldamodel.get_document_topics(te_corpus))
    lda_features_tfidf_te = np.zeros([len(te_gensim), n_topics])
    for i in range(len(te_gensim)):
        for elem in te_gensim[i]:
            lda_features_tfidf_te[i, elem[0]] = elem[1]
    np.save(main_folder + 'features/lda_test', lda_features_tfidf_te)


def concat_features(run_info):

    main_folder = run_info['main_folder']
    included_features = run_info['included_features']
    print('Concatenating feature sets in: ', included_features)

    # included_features = ['vae', 'lda_tf', 'lda_tfidf']
    address_tr = main_folder + 'features/' + included_features[0] + '_train.npy'
    features_tr = np.load(address_tr)

    address_te = main_folder + 'features/' + included_features[0] + '_test.npy'
    features_te = np.load(address_te)
    feat_name = included_features[0]
    if len(included_features) > 1:

        for feat in included_features[1:]:
            address_tr = main_folder + 'features/' + feat + '_train.npy'
            feat_tr = np.load(address_tr)
            features_tr = np.concatenate((features_tr, feat_tr), axis=1)

            address_te = main_folder + 'features/' + feat + '_test.npy'
            feat_te = np.load(address_te)
            features_te = np.concatenate((features_te, feat_te), axis=1)
            feat_name = feat_name + '_' + feat
    # vae_features_tr = np.load(main_folder + 'features/vae_train.npy')
    # vae_features_te = np.load(main_folder + 'features/vae_test.npy')
    # lda_features_tf_tr = np.load(main_folder + 'features/lda_tf_train.npy')
    # lda_features_tf_te = np.load(main_folder + 'features/lda_tf_test.npy')
    # lda_features_tfidf_tr = np.load(main_folder + 'features/lda_tfidf_train.npy')
    # lda_features_tfidf_te = np.load(main_folder + 'features/lda_tfidf_test.npy')
    #
    # lvae_features_tr = np.concatenate((vae_features_tr, lda_features_tf_tr), axis=1)
    # lvae_features_te = np.concatenate((vae_features_te, lda_features_tf_te), axis=1)

    # save the features
    np.save(main_folder + 'features/lvae_train_' + feat_name, features_tr)
    np.save(main_folder + 'features/lvae_test_' + feat_name, features_te)


def make_results_df(top_folder, results_path, dataset_name):
    # top_folder = 'runs/'
    # dataset_name = 'ISOT' # 'test'
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if dataset_name in dname]
            # and 'topics_32' in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[1]
        features_path = run + '/features/'
        vae_features_train = np.load(features_path + 'vae_train.npy')
        vae_features_test = np.load(features_path + 'vae_test.npy')
        lda_features_train = np.load(features_path + 'lda_tf_train.npy')
        lda_features_test = np.load(features_path + 'lda_tf_test.npy')
        data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
        data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
        y_train = np.load(run + '/train_label.npy')[:, 1]
        y_test = np.load(run + '/test_label.npy')[:, 1]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
        classifiers = ['SVM_linear', 'SVM_nonlinear', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                       'KNN (K = 3)', 'linear disriminat Analysis']
        classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                             mlp_classifier, knn3, linear_discriminat_analysis]
        features = ['VAE', 'LDA', 'VAE + LDA']
        datasets = ['Train', 'Test']

        data_tr = [vae_features_train, lda_features_train, data3_train]
        data_te = [vae_features_test, lda_features_test, data3_test]
        len_df = len(classifiers) * len(metrics) * len(datasets) * len(data_tr)
        results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric'], index=range(len_df))
        ct = 0
        for c in range(len(classifiers)):
            classifier_name = classifiers[c]
            classifier = classifiers_funcs[c]
            print(classifier_name)
            for d in range(len(data_tr)):
                this_tr = data_tr[d]
                this_te = data_te[d]
                this_feature = features[d]
                print(this_feature)
                tr, te = classifier(this_tr, this_te, y_train, y_test)
                accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
                accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
                results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy'
                results_pd.loc[ct+1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision'
                results_pd.loc[ct+2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall'
                results_pd.loc[ct+3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score'
                results_pd.loc[ct+4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR'
                results_pd.loc[ct+5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR'
                results_pd.loc[ct+6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy'
                results_pd.loc[ct+7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision'
                results_pd.loc[ct+8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall'
                results_pd.loc[ct+9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score'
                results_pd.loc[ct+10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR'
                results_pd.loc[ct+11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR'
                ct += 12
            results_pd.to_csv(results_path + model_name + '.csv', index=False)


def make_classifiers_df(run_info, k, over_sampled=False):
    print('Evaluating the features by classifiers ...')
    main_folder = run_info['main_folder']

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

    lda_features_train = np.load(features_path + 'lda_tfidf_train_' + str(k) + '.npy')
    lda_features_test = np.load(features_path + 'lda_tfidf_test_' + str(k) + '.npy')
    data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
    data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
    classifiers = ['SVM_linear', 'SVM_nonlinear', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                   'KNN (K = 3)']
    classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                         mlp_classifier, knn3]
    features = ['VAE', 'LDA', 'VAE + LDA']
    datasets = ['Train', 'Test']

    data_tr = [vae_features_train, lda_features_train, data3_train]
    data_te = [vae_features_test, lda_features_test, data3_test]
    len_df = len(classifiers) * len(metrics) * len(datasets) * len(data_tr)
    results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric'], index=range(len_df))
    ct = 0
    for c in range(len(classifiers)):
        classifier_name = classifiers[c]
        print(classifier_name)

        classifier = classifiers_funcs[c]
        # print(classifier_name)
        for d in range(len(data_tr)):
            print(d)
            this_tr = data_tr[d]
            this_te = data_te[d]
            this_feature = features[d]
            # print(this_feature)
            tr, te = classifier(this_tr, this_te, y_train, y_test)
            accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
            accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
            results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy'
            results_pd.loc[ct+1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision'
            results_pd.loc[ct+2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall'
            results_pd.loc[ct+3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score'
            results_pd.loc[ct+4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR'
            results_pd.loc[ct+5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR'
            results_pd.loc[ct+6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy'
            results_pd.loc[ct+7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision'
            results_pd.loc[ct+8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall'
            results_pd.loc[ct+9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score'
            results_pd.loc[ct+10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR'
            results_pd.loc[ct+11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR'
            ct += 12
        # results_pd.to_csv(results_path + model_name + '.csv', index=False)
        if over_sampled:
            results_pd.to_csv(main_folder + 'Classifiers_over_sampled.csv', index=False)
        else:
            results_pd.to_csv(main_folder + 'Classifiers_lda_' + str(k) + '.csv', index=False)


def make_classifiers_with_fs_chi2_df(run_info, over_sampled=False):
    print('Evaluating the features by classifiers and chi2 feature selection...')
    main_folder = run_info['main_folder']
    latent_dim = run_info['latent_dim']
    n_topics = run_info['n_topics']
    if min(latent_dim, n_topics) <= 32:
        check_point_nf = [1]
        [check_point_nf.append(cc) for cc in list(np.arange(0, min(latent_dim, n_topics), 5))[1:]]
        check_point_nf.append(min(latent_dim, n_topics))
    else:
        check_point_nf = [1]
        [check_point_nf.append(cc) for cc in list(np.arange(0, min(latent_dim, n_topics), 10))[1:]]
        check_point_nf.append(min(latent_dim, n_topics))

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

    lda_features_train = np.load(features_path + 'lda_tfidf_train_50.npy')
    lda_features_test = np.load(features_path + 'lda_tfidf_test_50.npy')
    data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
    data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
    classifiers = ['SVM (linear)', 'SVM (nonlinear)', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                   'KNN (K = 3)', 'linear disriminat Analysis']
    classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                         mlp_classifier, knn3, linear_discriminat_analysis]
    features = ['VAE', 'LDA', 'VAE + LDA']
    datasets = ['Train', 'Test']
    len_df = len(classifiers) * len(metrics) * len(datasets) * len(features) * len(check_point_nf)
    results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric',
                                               '# Selected Features'], index=range(len_df))
    ct = 0
    for nf in check_point_nf:
        # print('Number of features = ', nf)
        new_vae_features_train = SelectKBestFS(vae_features_train, y_train, nf)
        new_vae_features_test = SelectKBestFS(vae_features_test, y_test, nf)
        new_lda_features_train = SelectKBestFS(lda_features_train, y_train, nf)
        new_lda_features_test = SelectKBestFS(lda_features_test, y_test, nf)
        new_data3_train = SelectKBestFS(data3_train, y_train, nf)
        new_data3_test = SelectKBestFS(data3_test, y_test, nf)

        data_tr = [new_vae_features_train, new_lda_features_train, new_data3_train]
        data_te = [new_vae_features_test, new_lda_features_test, new_data3_test]

        for c in range(len(classifiers)):
            classifier_name = classifiers[c]
            classifier = classifiers_funcs[c]
            # print(classifier_name)
            for d in range(len(data_tr)):
                this_tr = data_tr[d]
                this_te = data_te[d]
                this_feature = features[d]
                # print(this_feature)
                tr, te = classifier(this_tr, this_te, y_train, y_test)
                accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
                accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
                results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy', nf
                results_pd.loc[ct+1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision', nf
                results_pd.loc[ct+2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall', nf
                results_pd.loc[ct+3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score', nf
                results_pd.loc[ct+4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR', nf
                results_pd.loc[ct+5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR', nf
                results_pd.loc[ct+6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy', nf
                results_pd.loc[ct+7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision', nf
                results_pd.loc[ct+8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall', nf
                results_pd.loc[ct+9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score', nf
                results_pd.loc[ct+10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR', nf
                results_pd.loc[ct+11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR', nf
                ct += 12
            if over_sampled:
                results_pd.to_csv(main_folder + 'Classifiers_with_chi2_fs_over_sampled.csv', index=False)
            else:
                results_pd.to_csv(main_folder + 'Classifiers_with_chi2_fs.csv', index=False)


def make_classifiers_with_fs_gini_df(run_info, over_sampled=False):
    print('Evaluating the features by classifiers and gini feature selection ...')
    main_folder = run_info['main_folder']
    latent_dim = run_info['latent_dim']
    n_topics = run_info['n_topics']
    if min(latent_dim, n_topics) <= 32:
        check_point_nf = [1]
        [check_point_nf.append(cc) for cc in list(np.arange(0, min(latent_dim, n_topics), 5))[1:]]
        check_point_nf.append(min(latent_dim, n_topics))
    else:
        check_point_nf = [1]
        [check_point_nf.append(cc) for cc in list(np.arange(0, min(latent_dim, n_topics), 10))[1:]]
        check_point_nf.append(min(latent_dim, n_topics))

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

    metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
    classifiers = ['SVM (linear)', 'SVM (nonlinear)', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                   'KNN (K = 3)', 'linear disriminat Analysis']
    classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                         mlp_classifier, knn3, linear_discriminat_analysis]
    features = ['VAE', 'LDA', 'VAE + LDA']
    datasets = ['Train', 'Test']
    len_df = len(classifiers) * len(metrics) * len(datasets) * len(features) * len(check_point_nf)
    results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric',
                                               '# Selected Features'], index=range(len_df))
    ct = 0
    for nf in check_point_nf:
        # print('Number of features = ', nf)
        new_vae_features_train = random_forest_feasture_selection(vae_features_train, y_train, nf)
        new_vae_features_test = random_forest_feasture_selection(vae_features_test, y_test, nf)
        new_lda_features_train = random_forest_feasture_selection(lda_features_train, y_train, nf)
        new_lda_features_test = random_forest_feasture_selection(lda_features_test, y_test, nf)
        new_data3_train = random_forest_feasture_selection(data3_train, y_train, nf)
        new_data3_test = random_forest_feasture_selection(data3_test, y_test, nf)

        data_tr = [new_vae_features_train, new_lda_features_train, new_data3_train]
        data_te = [new_vae_features_test, new_lda_features_test, new_data3_test]

        for c in range(len(classifiers)):
            classifier_name = classifiers[c]
            classifier = classifiers_funcs[c]
            # print(classifier_name)
            for d in range(len(data_tr)):
                this_tr = data_tr[d]
                this_te = data_te[d]
                this_feature = features[d]
                # print(this_feature)
                tr, te = classifier(this_tr, this_te, y_train, y_test)
                accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
                accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
                results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy', nf
                results_pd.loc[ct + 1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision', nf
                results_pd.loc[ct + 2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall', nf
                results_pd.loc[ct + 3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score', nf
                results_pd.loc[ct + 4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR', nf
                results_pd.loc[ct + 5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR', nf
                results_pd.loc[ct + 6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy', nf
                results_pd.loc[ct + 7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision', nf
                results_pd.loc[ct + 8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall', nf
                results_pd.loc[ct + 9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score', nf
                results_pd.loc[ct + 10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR', nf
                results_pd.loc[ct + 11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR', nf
                ct += 12

            if over_sampled:
                results_pd.to_csv(main_folder + 'Classifiers_with_gini_fs_over_sampled.csv', index=False)
            else:
                results_pd.to_csv(main_folder + 'Classifiers_with_gini_fs.csv', index=False)


def make_results_with_fs_df(top_folder, results_path, dataset_name):
    # top_folder = 'runs/'
    # dataset_name = 'ISOT' # 'test'
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if dataset_name in dname
            and 'topics_32' in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[1]
        features_path = run + '/features/'
        vae_features_train = np.load(features_path + 'vae_train.npy')
        vae_features_test = np.load(features_path + 'vae_test.npy')
        lda_features_train = np.load(features_path + 'lda_tf_train.npy')
        lda_features_test = np.load(features_path + 'lda_tf_test.npy')
        data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
        data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
        y_train = np.load(run + '/train_label.npy')[:, 1]
        y_test = np.load(run + '/test_label.npy')[:, 1]
        nfall = lda_features_test.shape[1]
        # nfvae = vae_features_test.shape[1]
        check_point_nf = [1, 5, 10, 15, 20, 25, 30, nfall]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
        classifiers = ['SVM (linear)', 'SVM (nonlinear)', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                       'KNN (K = 3)', 'linear disriminat Analysis']
        classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                             mlp_classifier, knn3, linear_discriminat_analysis]
        features = ['VAE', 'LDA', 'VAE + LDA']
        datasets = ['Train', 'Test']
        len_df = len(classifiers) * len(metrics) * len(datasets) * len(features) * len(check_point_nf)
        results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric',
                                                   '# Selected Features'], index=range(len_df))
        ct = 0
        for nf in check_point_nf:
            print('Number of features = ', nf)
            new_vae_features_train = SelectKBestFS(vae_features_train, y_train, nf)
            new_vae_features_test = SelectKBestFS(vae_features_test, y_test, nf)
            new_lda_features_train = SelectKBestFS(lda_features_train, y_train, nf)
            new_lda_features_test = SelectKBestFS(lda_features_test, y_test, nf)
            new_data3_train = SelectKBestFS(data3_train, y_train, nf)
            new_data3_test = SelectKBestFS(data3_test, y_test, nf)

            data_tr = [new_vae_features_train, new_lda_features_train, new_data3_train]
            data_te = [new_vae_features_test, new_lda_features_test, new_data3_test]

            for c in range(len(classifiers)):
                classifier_name = classifiers[c]
                classifier = classifiers_funcs[c]
                print(classifier_name)
                for d in range(len(data_tr)):
                    this_tr = data_tr[d]
                    this_te = data_te[d]
                    this_feature = features[d]
                    print(this_feature)
                    tr, te = classifier(this_tr, this_te, y_train, y_test)
                    accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
                    accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
                    results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy', nf
                    results_pd.loc[ct+1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision', nf
                    results_pd.loc[ct+2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall', nf
                    results_pd.loc[ct+3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score', nf
                    results_pd.loc[ct+4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR', nf
                    results_pd.loc[ct+5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR', nf
                    results_pd.loc[ct+6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy', nf
                    results_pd.loc[ct+7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision', nf
                    results_pd.loc[ct+8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall', nf
                    results_pd.loc[ct+9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score', nf
                    results_pd.loc[ct+10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR', nf
                    results_pd.loc[ct+11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR', nf
                    ct += 12
                results_pd.to_csv(results_path + model_name + '_with_fs.csv', index=False)


def make_results_with_randome_forest_fs_df(top_folder, results_path, dataset_name):
    # top_folder = 'runs/'
    # dataset_name = 'ISOT' # 'test'
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if dataset_name in dname
             and 'topics_32' in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[1]
        features_path = run + '/features/'
        vae_features_train = np.load(features_path + 'vae_train.npy')
        vae_features_test = np.load(features_path + 'vae_test.npy')
        lda_features_train = np.load(features_path + 'lda_tf_train.npy')
        lda_features_test = np.load(features_path + 'lda_tf_test.npy')
        data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
        data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
        y_train = np.load(run + '/train_label.npy')[:, 1]
        y_test = np.load(run + '/test_label.npy')[:, 1]
        nfall = lda_features_test.shape[1]
        # nfvae = vae_features_test.shape[1]
        check_point_nf = [1, 5, 10, 15, 20, 25, 30, nfall]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
        classifiers = ['SVM (linear)', 'SVM (nonlinear)', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                       'KNN (K = 3)', 'linear disriminat Analysis']
        classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                             mlp_classifier, knn3, linear_discriminat_analysis]
        features = ['VAE', 'LDA', 'VAE + LDA']
        datasets = ['Train', 'Test']
        len_df = len(classifiers) * len(metrics) * len(datasets) * len(features) * len(check_point_nf)
        results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric',
                                                   '# Selected Features'], index=range(len_df))
        ct = 0
        for nf in check_point_nf:
            print('Number of features = ', nf)
            new_vae_features_train = random_forest_feasture_selection(vae_features_train, y_train, nf)
            new_vae_features_test = random_forest_feasture_selection(vae_features_test, y_test, nf)
            new_lda_features_train = random_forest_feasture_selection(lda_features_train, y_train, nf)
            new_lda_features_test = random_forest_feasture_selection(lda_features_test, y_test, nf)
            new_data3_train = random_forest_feasture_selection(data3_train, y_train, nf)
            new_data3_test = random_forest_feasture_selection(data3_test, y_test, nf)

            data_tr = [new_vae_features_train, new_lda_features_train, new_data3_train]
            data_te = [new_vae_features_test, new_lda_features_test, new_data3_test]

            for c in range(len(classifiers)):
                classifier_name = classifiers[c]
                classifier = classifiers_funcs[c]
                print(classifier_name)
                for d in range(len(data_tr)):
                    this_tr = data_tr[d]
                    this_te = data_te[d]
                    this_feature = features[d]
                    print(this_feature)
                    tr, te = classifier(this_tr, this_te, y_train, y_test)
                    accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
                    accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
                    results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy', nf
                    results_pd.loc[ct+1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision', nf
                    results_pd.loc[ct+2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall', nf
                    results_pd.loc[ct+3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score', nf
                    results_pd.loc[ct+4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR', nf
                    results_pd.loc[ct+5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR', nf
                    results_pd.loc[ct+6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy', nf
                    results_pd.loc[ct+7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision', nf
                    results_pd.loc[ct+8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall', nf
                    results_pd.loc[ct+9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score', nf
                    results_pd.loc[ct+10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR', nf
                    results_pd.loc[ct+11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR', nf
                    ct += 12
                results_pd.to_csv(results_path + model_name + '_with_randoem_forest_fs.csv', index=False)


def evaluation_expermient(info):
    print('Evaluating the results ...')

    for k in [32, 50, 10, 64]:
        make_classifiers_df(info, k)
    # plot_classifiers_result(info)

    make_classifiers_with_fs_chi2_df(info)
    plot_classifiers_with_fs_result(info, 'chi2')

    make_classifiers_with_fs_gini_df(info)
    plot_classifiers_with_fs_result(info, 'gini')

    plot_pca_tsne(info)


def compute_classifier_results_all_runs(dataset_name):
    plots_path = top_folder + 'Plots/'
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    results_path = top_folder + 'Results/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    plot_all_pca_tsne_datasets(top_folder, plots_path, dataset_name)

    make_results_df(top_folder, results_path, dataset_name)

    make_results_with_fs_df(top_folder, results_path, dataset_name)

    plot_history_independent(top_folder, plots_path, dataset_name)

    plot_history_with_1_outputs(top_folder, plots_path, dataset_name)

    plot_accuracy_metrics(results_path, plots_path)
    plot_accuracy_with_fs(results_path, plots_path)


def main(run_info):
    main_folder = run_info['main_folder']

    # data preparation
    if not os.path.exists(top_folder + 'data/' + dataset_name + '/x_train_' + str(word2vec_dim) + '.csv') or \
            not os.path.exists(top_folder + 'data/' + dataset_name + '/x_test_' + str(word2vec_dim) + '.csv'):
        new_prepare_data(run_info, top_folder, dataset_address)

    # # VAE
    # if not os.path.exists(main_folder + 'features/vae_train.npy') or \
    #         not os.path.exists(main_folder + 'features/vae_test.npy'):
    #     vae_experiment(run_info, top_folder)

    # LDA
    if not os.path.exists(main_folder + 'features/lda_tfidf_train.npy') or \
            not os.path.exists(main_folder + 'features/lda_tfidf_test.npy'):
        # lda_experiment(run_info, top_folder)
        # new_lda_experiment(run_info, top_folder)
        new_lda_experiment_gensim(run_info, top_folder)
    print('done')
    exit(0)
    # LVAE (VAE + LDA): Our method
    if not os.path.exists(main_folder + 'features/lvae_train_vae_lda_tfidf.npy') or \
            not os.path.exists(main_folder + 'features/lvae_test_vae_lda_tfidf.npy'):
        concat_features(run_info)

    # classification and evaluation (feature selection and dimensionality reduction)
    evaluation_expermient(run_info)
    clustering_expermient(main_folder)

    # run this function only if results for runs are not already computed

    lvae_classifier(run_info)


if __name__ == '__main__':

    if '-f' in sys.argv:
        top_folder = sys.argv[sys.argv.index('-f') + 1]
    else:
        top_folder = 'runs/'

    if '-d' in sys.argv:
        dataset_name = sys.argv[sys.argv.index('-d') + 1]
    else:
        dataset_name = exit('Error: You need to specify the dataset name with -d command. \nDatasets choices could be '
                            'Twitter or ISOT or Covid.')
        # dataset_name = 'ISOT'

    if '-a' in sys.argv:
        dataset_address = sys.argv[sys.argv.index('-a') + 1]
    else:
        dataset_address = exit('Error: You need to specify the address of top folder contatining both dataset folders '
                               'with -a command, eg. -a "data/".')
        # dataset_address = 'data/'

    if '-e' in sys.argv:
        epoch_no = int(sys.argv[sys.argv.index('-e') + 1])
    else:
        epoch_no = 10

    if '-t' in sys.argv:
        n_topics = int(sys.argv[sys.argv.index('-t') + 1])
    else:
        n_topics = 32

    if '-i' in sys.argv:
        n_iter = int(sys.argv[sys.argv.index('-i') + 1])

    else:
        n_iter = 500

    if '-l' in sys.argv:
        latent_dim = int(sys.argv[sys.argv.index('-l') + 1])
    else:
        latent_dim = 32

    if '-w' in sys.argv:
        word2vec_dim = int(sys.argv[sys.argv.index('-w') + 1])
    else:
        word2vec_dim = 32

    run_info = make_run_info(top_folder, dataset_name, latent_dim, epoch_no, n_topics, n_iter, word2vec_dim)

    main(run_info)
