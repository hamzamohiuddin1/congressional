from congress import Congress
import math
import nltk
import random
# nltk.download('punkt')
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

PRO_KEY = 'uHFYJYtV4pnqdvOivQceJCaSKwQ2DKsshXDcAvoN'


def main():
    """Calculate top TF-IDF for a corpus of documents."""

    if len(sys.argv) != 4:
        sys.exit("Usage: python congressional.py corpus firstname lastname")
    print("Loading data...")
    corpus, names = load_data(sys.argv[1])
    query = f"{sys.argv[2]} {sys.argv[3]}".lower()

    # Get all words in corpus
    print("Extracting words from corpus...")
    words = set()
    for filename in corpus:
        words.update(corpus[filename])

    # Calculate IDFs
    print("Calculating inverse document frequencies...")
    idfs = dict()
    for word in words:
        f = sum(word in corpus[filename] for filename in corpus)
        idf = math.log(len(corpus) / f)
        idfs[word] = idf

    # Calculate TF-IDFs
    print("Calculating term frequencies...")
    tfidfs = dict()
    for filename in corpus:
        tfidfs[filename] = []
        for word in corpus[filename]:
            tf = corpus[filename][word]
            tfidfs[filename].append((word, tf * idfs[word]))

    # Sort and get top 5 TF-IDFs for each file
    print("Computing top terms...")
    for filename in corpus:
        tfidfs[filename].sort(key=lambda tfidf: tfidf[1], reverse=True)
        tfidfs[filename] = tfidfs[filename][:10]

    # Create dictionary (filename, list of keywords)
    keywords = dict()
    for filename in corpus:
        terms = []
        for term in tfidfs[filename]:
            terms.append(term[0])
        keywords[filename] = terms

    # Create dictionary (name, list of passable bills) and (name, list of failed bills)
    names_to_passable_bills = dict()
    names_to_failable_bills = dict()
    for filename in names:
        for n in names[filename][0]:
            if n not in names_to_passable_bills.keys():
                names_to_passable_bills[n] = []
            if filename not in names_to_passable_bills[n]:
                names_to_passable_bills[n].append(filename)
        for n in names[filename][1]:
            if n not in names_to_failable_bills.keys():
                names_to_failable_bills[n] = []
            if filename not in names_to_failable_bills[n]:
                names_to_failable_bills[n].append(filename)

    # create list of keyword sets for passable bills based on a query
    passes = []
    for bill in names_to_passable_bills[query]:
        print(f"Pass: {bill}")
        passes.append(keywords[bill])

    # create list of keyword sets for failed bills
    fails = []
    for bill in names_to_failable_bills[query]:
        print(f"Fail: {bill}")
        fails.append(keywords[bill])

    # create a set of all bill text
    bill_text = set()
    for document in passes:
        bill_text.update(document)
    for document in fails:
        bill_text.update(document)

    # extract features from text
    training = []
    training.extend(generate_features(passes, bill_text, "Pass"))
    training.extend(generate_features(fails, bill_text, "Fail"))

    # classify a new sample
    classifier = nltk.NaiveBayesClassifier.train(training)
    while True:
        s = input("Enter a bill: ").lower()
        result = (classify(classifier, s, bill_text))
        pass_probability = result.prob("Pass")
        fail_probability = result.prob("Fail")
        print(f"Probability of pass: {pass_probability}, "
              f"Probability of fail: {fail_probability}")


def load_data(directory):
    """
    :param directory:
    :return: files: a dictionary mapping bill titles to the text content of the bill
             names: a dictionary mapping bill titles to the supporters and opposers of the bill
    """
    files = dict()
    names = dict()
    senate = ["s", "sconres", "sjres", "sres", ".DS_Store"]
    for folder in os.listdir(directory):
        if folder in senate:
            continue
        print(folder)
        for filename in os.listdir(os.path.join(directory, folder)):
            if filename == '.DS_Store':
                continue
            with open(os.path.join(directory, folder, filename, "fdsys_billstatus.xml")) as f:
                # Extract words
                contents = [
                    word.lower() for word in
                    nltk.word_tokenize(f.read())
                    if word.isalnum()
                ]
                if "rollnumber" not in contents:
                    continue
                supporters, opposers = get_supporters(get_roll_call(contents))
                # Count frequencies
                frequencies = dict()
                for i, word in enumerate(contents):
                    if word not in frequencies:
                        frequencies[word] = 1
                    else:
                        frequencies[word] += 1

                files[filename] = frequencies
                names[filename] = supporters, opposers

    return files, names


def generate_features(documents, words, label):
    features = []
    for document in documents:
        features.append(({
                             word: (word in document)
                             for word in words
                         }, label))
    return features


def classify(classifier, document, words):
    document_words = extract_words(document)
    features = {
        word: (word in document_words)
        for word in words
    }
    return classifier.prob_classify(features)


def extract_words(document):
    return set(
        word.lower() for word in nltk.word_tokenize(document)
        if any(c.isalpha() for c in word)
    )


def get_supporters(roll_call_no):
    supporters = []
    opposers = []
    congress = Congress(PRO_KEY)
    rollcall = congress.votes.get('house', roll_call_no, 1, 118)
    for position in rollcall['votes']['vote']['positions']:
        if position['vote_position'] == "Yes":
            supporters.append(position['name'].lower())
        elif position['vote_position'] == "No":
            opposers.append(position['name'].lower())
        else:
            continue
    return supporters, opposers


def get_roll_call(contents):
    for i, word in enumerate(contents):
        if word == "roll":
            return int(contents[i + 2])
        else:
            continue
    return 0


if __name__ == "__main__":
    # get_supporters(161)
    main()
