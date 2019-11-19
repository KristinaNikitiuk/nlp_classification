import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from gensim import corpora, similarities
from gensim.models import TfidfModel, LsiModel
from tqdm import tqdm
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


class TextProcessing(object):

    def __init__(self):
        pass

    def _get_most_frequent_words(self, review_phrases):
        top_words = []
        fd = FreqDist()
        for text in tqdm(review_phrases):
            fd.update(text)
        for i in fd.most_common(1000):
            top_words.append(i[0])
        print('top_words - ', top_words[:10])
        return top_words

    def topic_modeling(self, review_phrases):
        dictionary = corpora.Dictionary(review_phrases)

        corpus = [dictionary.doc2bow(text) for text in review_phrases]
        print(corpus[0:2])

        tfidf = TfidfModel(corpus)
        corpus_idf = tfidf[corpus]
        print("corpus_idf - ", corpus_idf[0])

        lsi = LsiModel(corpus=corpus_idf, id2word=dictionary, num_topics=5)
        corpus_lsi = lsi[corpus]
        index = similarities.MatrixSimilarity(lsi[corpus])
        sims = index[corpus_lsi]
        print(sims[0:3])

    def clean_phrases(self, dataframe):
        review_phrases = []
        lemmatizer = WordNetLemmatizer()

        print('count: \n', dataframe.groupby('sentiment').count())

        stop_words_list = stopwords.words('english') + ['br', 'wa', 'ha']
        for i in dataframe['review']:
            review_text = re.sub(r'\W', ' ', i)
            review_text = re.sub(r'\s+[a-zA-Z]\s+', " ", review_text)
            words = word_tokenize(review_text.lower())
            lemma_words = [lemmatizer.lemmatize(i) for i in words]
            filtered_words = [word for word in lemma_words if word not in stop_words_list]
            filtered_words = ' '.join(filtered_words)
            review_phrases.append(filtered_words)

        return review_phrases


