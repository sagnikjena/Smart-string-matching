import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint


def pre_process_text(name, n=3):
    """

    :param name:
    :param n: determines the n-grams (bi-gram, tri-gram, etc). By default, it's tri-gram
    :return: string with the n-grams
    """

    name = re.sub(r'[,-./]|\sBD', r'', name)
    name = re.sub(" +", "", name)
    tokens = zip(*[name[i:] for i in range(n)])
    return [''.join(token) for token in tokens]


def get_top_n_match(row, n_top=5):
    """

    :param row:
    :param n_top: number of results to be determined
    :return: list of tuples with index of the match and the cosine similarity score
    """

    row_count = row.getnnz()
    if row_count == 0:
        return None
    elif row_count <= n_top:
        result = zip(row.indices, row.data)
    else:
        arg_idx = np.argpartition(row.data, -n_top)[-n_top:]
        result = zip(row.indices[arg_idx], row.data[arg_idx])
    return sorted(result, key=(lambda x: -x[1]))


def match_company_name(input_name, vectorizer, comp_name_vectors, comp_name_df):
    """

    :param input_name: input company name whose matches need to be found
    :param vectorizer: TFIDF vectorizer which was initialized earlier
    :param comp_name_vectors: the company names' vectors of the whole data set
    :param comp_name_df: the company names dataframe
    :return: a dataframe with top N matching names with match score
    """

    input_name_vector = vectorizer.transform([input_name])
    result_vector = input_name_vector.dot(comp_name_vectors.T)
    matched_data = [get_top_n_match(row) for row in result_vector]
    flat_matched_data = [tup for data_row in matched_data for tup in data_row]
    lkp_idx, lkp_sim = zip(*flat_matched_data)
    nr_matches = len(lkp_idx)
    matched_names = np.empty([nr_matches], dtype=object)
    sim = np.zeros(nr_matches)
    for i in range(nr_matches):
        matched_names[i] = comp_name_df['Company Name'][lkp_idx[i]]
        sim[i] = lkp_sim[i]
    return pd.DataFrame({"Matching company name": matched_names,
                         "Match Score (%)": sim*100})


if __name__ == '__main__':

    company_names_df = pd.read_csv("./data/company_names.csv")
    tfidf = TfidfVectorizer(min_df=5, analyzer=pre_process_text)
    company_name_vectors = tfidf.fit_transform(company_names_df['Company Name'])

    # Example
    result_df = match_company_name("ADVISORY U S EQUITY MARKET", tfidf, company_name_vectors, company_names_df)
    print(result_df)