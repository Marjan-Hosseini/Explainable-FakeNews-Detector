from LVAE.preprocessing import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from nltk.stem import WordNetLemmatizer
sys.path.append('../')
np.random.seed(2021)


def get_top_n_words(n, n_topics, keys, document_term_matrix, count_vectorizer):
    '''
    returns a list of n_topic strings, where each string contains the n most common
    words in a predicted category, in order
    '''
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
        top_word_indices.append(top_n_word_indices)
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
            temp_word_vector[:,index] = 1
            the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii',errors="ignore").decode('utf-8',errors="ignore"))
        top_words.append(" ".join(topic_words))
    return top_words


def get_keys(topic_matrix):
    '''
    returns an integer list of predicted topic
    categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys


def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their
    accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)


def LDA_train(n_topics, news_df, tf_idf=True, learning_method='batch', max_iter=100):
    # n_topics = 10

    small_count_vectorizer = CountVectorizer(stop_words='english', max_features=40000, max_df=0.95, min_df=2)
    text_vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_features=40000, max_df=0.95, min_df=2)
    # processed_docs = news_df['text'].map(preprocess)

    if tf_idf:
        small_document_term_matrix = text_vectorizer_tfidf.fit_transform(news_df['text'])
    else:
        small_document_term_matrix = small_count_vectorizer.fit_transform(news_df['text'])

    lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method=learning_method,
                                          verbose=1, max_iter=max_iter)

    lda_topic_matrix = lda_model.fit_transform(small_document_term_matrix)

    lda_keys = get_keys(lda_topic_matrix)

    # lda_categories, lda_counts = keys_to_counts(lda_keys)
    if tf_idf:
        top_n_words_lda = get_top_n_words(20, n_topics, lda_keys, small_document_term_matrix, text_vectorizer_tfidf)
    else:
        top_n_words_lda = get_top_n_words(20, n_topics, lda_keys, small_document_term_matrix, small_count_vectorizer)

    for i in range(len(top_n_words_lda)):
        print("Topic {}: ".format(i+1), top_n_words_lda[i])
    return lda_model


def LDA_train_tf(folder, n_components, data, n_top_words, n_features, n_iter=1000):

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    # max_features=n_features,
                                    stop_words='english', decode_error="replace")
    tf = tf_vectorizer.fit_transform(data)

    # path = folder + 'lda/'
    # if not os.path.exists(path):
    #     os.makedirs(path)

    with open(folder + 'lda_vectorizer_tf.pkl', 'wb') as handle:
        pkl.dump(tf_vectorizer.vocabulary_, handle)

    lda = LatentDirichletAllocation(n_components=n_components, max_iter=n_iter, learning_method='batch',
                                    learning_offset=50., random_state=0)

    lda.fit(tf)
    with open(folder + 'lda_model_tf.pkl', 'wb') as handle:
        pkl.dump(lda, handle)

    tf_feature_names = tf_vectorizer.get_feature_names()
    if n_components <= 10:
        plot_name = folder + 'lda_topic_tf'
        plot_top_words(lda, tf_feature_names, n_top_words, plot_name)
    return lda, tf_vectorizer


def LDA_train_tf_idf(folder, n_components, data, n_top_words, n_features, n_iter=1000):
    new_data = pd.DataFrame(data=0, columns=['text'], index=range(len(data)))
    for id in range(len(data)):
        post = data.iloc[id]
        new_data.iloc[id] = new_process_text(post)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       # max_features=n_features,
                                       stop_words='english', decode_error="replace")
    tfidf = tfidf_vectorizer.fit_transform(new_data['text'])

    with open(folder + 'lda_vectorizer_tfidf.pkl', 'wb') as handle:
        pkl.dump(tfidf_vectorizer.vocabulary_, handle)

    lda_2 = LatentDirichletAllocation(n_components=n_components, max_iter=n_iter, learning_method='batch',
                                      learning_offset=50., random_state=0)
    lda_2.fit(tfidf)

    with open(folder + 'lda_model_tfidf.pkl', 'wb') as handle:
        pkl.dump(lda_2, handle)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    plot_name = folder + 'lda_topic_tfidf'
    plot_top_words(lda_2, tfidf_feature_names, n_top_words, plot_name)

    return lda_2, tfidf_vectorizer


def LDA_train_tf_idf_extract_features(folder, n_components, data, train_data, test_data, n_top_words, n_features, n_iter=1000):

    tfidf_vectorizer = TfidfVectorizer(max_df=0.7, min_df=2,
                                       # max_features=n_features,
                                       stop_words='english', decode_error="replace")
    tfidf = tfidf_vectorizer.fit_transform(data)

    with open(folder + 'lda_vectorizer_tfidf.pkl', 'wb') as handle:
        pkl.dump(tfidf_vectorizer.vocabulary_, handle)

    lda_2 = LatentDirichletAllocation(n_components=n_components, max_iter=n_iter, learning_method='batch',
                                      learning_offset=50., random_state=0)
    lda_2.fit(tfidf)

    with open(folder + 'lda_model_tfidf.pkl', 'wb') as handle:
        pkl.dump(lda_2, handle)

    print('Extracting LDA with tf_idf features ...')

    X_train_vec = tfidf_vectorizer.transform(train_data)
    X_train_topics = lda_2.transform(X_train_vec)

    X_test_vec = tfidf_vectorizer.transform(test_data)
    X_test_topics = lda_2.transform(X_test_vec)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    if n_components <= 10:
        plot_name = folder + 'lda_topic_tfidf'
        plot_top_words(lda_2, tfidf_feature_names, n_top_words, plot_name)

    return X_train_topics, X_test_topics


def plot_top_words(model, feature_names, n_top_words, title):

    fig, axes = plt.subplots(1, 5, figsize=(30, 6), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        # ax.set_title(f'Topic  {topic_idx +1}', fontdict={'fontsize': 15})
        id_str = str(topic_idx + 1)
        eq = r'$(\beta_{k=' + id_str + '})$'
        txt = 'Topic ' + id_str + ' '
        # aa = r'$\text\{Topic {0}\} (\beta_\{k={1}\})$'.format(sss, sss)

        ax.set_title(txt + eq, fontdict={'fontsize': 15})

        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=13)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        # fig.suptitle(title, fontsize=25)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.30, hspace=0.1)
    plt.savefig(title+'.png')
    plt.savefig(title+'.pdf')


def extract_lda_tf_features(main_folder, data, data_name):
    print('Extracting LDA features for', data_name)
    with open(main_folder + 'lda_model_tf.pkl', 'rb') as handle:
        lda_tf = pkl.load(handle)

    with open(main_folder + 'lda_vectorizer_tf.pkl', 'rb') as handle:
        tf_vectorizer = pkl.load(handle)

    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=tf_vectorizer)

    X_train_vec = loaded_vec.transform(data)
    X_train_topics = lda_tf.transform(X_train_vec)
    np.save(main_folder + 'lda_tf_features_' + data_name, X_train_topics)
    return X_train_topics


def extract_lda_tfidf_features(main_folder, data, data_name):
    with open(main_folder + 'lda_model_tfidf.pkl', 'rb') as handle:
        lda_tfidf = pkl.load(handle)

    with open(main_folder + 'lda_vectorizer_tfidf.pkl', 'rb') as handle:
        tfidf_vectorizer = pkl.load(handle)

    loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=tfidf_vectorizer)

    X_train_vec = loaded_vec.transform(data)
    X_train_topics = lda_tfidf.transform(X_train_vec)
    np.save(main_folder + 'lda_tfidf_features_' + data_name, X_train_topics)
    return X_train_topics

