import pandas as pd

from model import Model
from text_processing import TextProcessing


def main():
    dataframe = pd.read_csv("review_dataset_2.csv", sep=',')
    y = dataframe.sentiment
    clean_review = TextProcessing().clean_phrases(dataframe)
    print('text processing finished!')
    # TextProcessing().topic_modeling(clean_review)
    Model(clean_review, y).run_model()


if __name__ == "__main__":
    main()

