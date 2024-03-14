import pandas as pd
import re


def split_paragraph_to_dataframe(paragraph):
    # Split the paragraph into lines at sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)

    # Calculate the total number of sentences in the paragraph
    total_sentences = len(sentences)
    total_lines = [total_sentences]*total_sentences

    # Create a DataFrame with sentence numbers and sentences
    df = pd.DataFrame({'Sentence Content': sentences,'Line number': range(0, total_sentences ),'Total lines':total_lines })

    return df, df['Sentence Content'].tolist()

# df , total_sentences = split_paragraph_to_dataframe()

def split_chars(text):
    return ' '.join(list(text))

def one_hot(df,tf):
    test_line_number_one_hot = tf.one_hot(df['Line number'].to_numpy(), depth=15)
    test_total_lines_one_hot = tf.one_hot(df['Total lines'].to_numpy(),depth=20)
    return test_line_number_one_hot, test_total_lines_one_hot
