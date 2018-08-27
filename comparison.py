import pandas as pd
import numpy as np
from collections import Counter
import math
from random import shuffle
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

document_number = 0
corpus = ""
documents = []
transformed_documents = []
document_labels = []
category_list = set()


def bag_of_words():
    average_confusion_matrix = np.zeros(shape=(len(category_list), len(category_list)))
    average_accuracy = 0
    words = list(zip(*Counter(corpus.split()).most_common(1000)))[0]

    for i in range(0, document_number):
        # print(i)
        transform = np.zeros(len(words))
        current = Counter(documents[i].split())
        keys = list(current)
        for j in range(0, len(current)):
            try:
                index = words.index(keys[j])
                transform[index] += current[keys[j]]
            except:
                error = 1
        contains = math.fsum(transform)
        if contains > 0:
            transformed_documents.append([x / contains for x in transform])
        else:
            transformed_documents.append(transform)

    new_documents = list(zip(transformed_documents, document_labels))
    shuffle(new_documents)
    svm_input = np.asarray(list(zip(*new_documents))[0])
    svm_output = np.asarray(list(zip(*new_documents))[1])

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(svm_input):
        x_train, x_test = svm_input[train_index], svm_input[test_index]
        y_train, y_test = svm_output[train_index], svm_output[test_index]

        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(x_train, y_train)
        predicted = clf.predict(x_test)
        confusion = confusion_matrix(predicted, y_test, labels=list(category_list))
        average_confusion_matrix = [[sum(x) for x in zip(average_confusion_matrix[i], confusion[i])] for i in range(len(average_confusion_matrix))]
        average_accuracy += accuracy_score(predicted, y_test)
        # print(confusion)
        # print(accuracy_score(predicted, y_test))

    average_confusion_matrix = [[x / 5 for x in average_confusion_matrix[i]] for i in range(len(average_confusion_matrix))]
    print(average_confusion_matrix)
    print(average_accuracy / 5)


def information_gain_words():
    average_confusion_matrix = np.zeros(shape=(len(category_list), len(category_list)))
    average_accuracy = 0
    column_names = ['W', 'Score Of Algorithm']
    data = pd.read_csv('/Users/negin/Desktop/Result_Information_Gain.csv', names=column_names)
    words = list(data['W'])[1:]

    for i in range(0, document_number):
        # print(i)
        transform = np.zeros(len(words))
        current = Counter(documents[i].split())
        keys = list(current)
        for j in range(0, len(current)):
            try:
                index = words.index(keys[j])
                transform[index] += current[keys[j]]
            except:
                error = 1
        contains = math.fsum(transform)
        if contains > 0:
            transformed_documents.append([x / contains for x in transform])
        else:
            transformed_documents.append(transform)

    new_documents = list(zip(transformed_documents, document_labels))
    shuffle(new_documents)
    svm_input = np.asarray(list(zip(*new_documents))[0])
    svm_output = np.asarray(list(zip(*new_documents))[1])

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(svm_input):
        x_train, x_test = svm_input[train_index], svm_input[test_index]
        y_train, y_test = svm_output[train_index], svm_output[test_index]

        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(x_train, y_train)
        predicted = clf.predict(x_test)
        confusion = confusion_matrix(predicted, y_test, labels=list(category_list))
        average_confusion_matrix = [[sum(x) for x in zip(average_confusion_matrix[i], confusion[i])] for i in range(len(average_confusion_matrix))]
        average_accuracy += accuracy_score(predicted, y_test)
        # print(confusion)
        # print(accuracy_score(predicted, y_test))

    average_confusion_matrix = [[x / 5 for x in average_confusion_matrix[i]] for i in range(len(average_confusion_matrix))]
    print(average_confusion_matrix)
    print(average_accuracy / 5)


def mutual_information_words():
    average_confusion_matrix = np.zeros(shape=(len(category_list), len(category_list)))
    average_accuracy = 0
    column_names = ['W', 'Score Of Algorithm', 'Main Domain', 'Score Of The Main Domain']
    data = pd.read_csv('/Users/negin/Desktop/Result_Mutual_Information.csv', names=column_names)
    words = list(data['W'])[1:]

    for i in range(0, document_number):
        # print(i)
        transform = np.zeros(len(words))
        current = Counter(documents[i].split())
        keys = list(current)
        for j in range(0, len(current)):
            try:
                index = words.index(keys[j])
                transform[index] += current[keys[j]]
            except:
                error = 1
        contains = math.fsum(transform)
        if contains > 0:
            transformed_documents.append([x / contains for x in transform])
        else:
            transformed_documents.append(transform)

    new_documents = list(zip(transformed_documents, document_labels))
    shuffle(new_documents)
    svm_input = np.asarray(list(zip(*new_documents))[0])
    svm_output = np.asarray(list(zip(*new_documents))[1])

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(svm_input):
        x_train, x_test = svm_input[train_index], svm_input[test_index]
        y_train, y_test = svm_output[train_index], svm_output[test_index]

        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(x_train, y_train)
        predicted = clf.predict(x_test)
        confusion = confusion_matrix(predicted, y_test, labels=list(category_list))
        average_confusion_matrix = [[sum(x) for x in zip(average_confusion_matrix[i], confusion[i])] for i in range(len(average_confusion_matrix))]
        average_accuracy += accuracy_score(predicted, y_test)
        # print(confusion)
        # print(accuracy_score(predicted, y_test))

    average_confusion_matrix = [[x / 5 for x in average_confusion_matrix[i]] for i in range(len(average_confusion_matrix))]
    print(average_confusion_matrix)
    print(average_accuracy / 5)


def chi_square_words():
    average_confusion_matrix = np.zeros(shape=(len(category_list), len(category_list)))
    average_accuracy = 0
    column_names = ['W', 'Score Of Algorithm', 'Main Domain', 'Score Of The Main Domain']
    data = pd.read_csv('/Users/negin/Desktop/Result_Chi_Square.csv', names=column_names)
    words = list(data['W'])[1:]

    for i in range(0, document_number):
        # print(i)
        transform = np.zeros(len(words))
        current = Counter(documents[i].split())
        keys = list(current)
        for j in range(0, len(current)):
            try:
                index = words.index(keys[j])
                transform[index] += current[keys[j]]
            except:
                error = 1
        contains = math.fsum(transform)
        if contains > 0:
            transformed_documents.append([x / contains for x in transform])
        else:
            transformed_documents.append(transform)

    new_documents = list(zip(transformed_documents, document_labels))
    shuffle(new_documents)
    svm_input = np.asarray(list(zip(*new_documents))[0])
    svm_output = np.asarray(list(zip(*new_documents))[1])

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(svm_input):
        x_train, x_test = svm_input[train_index], svm_input[test_index]
        y_train, y_test = svm_output[train_index], svm_output[test_index]

        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(x_train, y_train)
        predicted = clf.predict(x_test)
        confusion = confusion_matrix(predicted, y_test, labels=list(category_list))
        average_confusion_matrix = [[sum(x) for x in zip(average_confusion_matrix[i], confusion[i])] for i in range(len(average_confusion_matrix))]
        average_accuracy += accuracy_score(predicted, y_test)
        # print(confusion)
        # print(accuracy_score(predicted, y_test))

    average_confusion_matrix = [[x / 5 for x in average_confusion_matrix[i]] for i in range(len(average_confusion_matrix))]
    print(average_confusion_matrix)
    print(average_accuracy / 5)


if __name__ == '__main__':

    with open('./dataset/corpus.txt', 'r') as fileCollection:
        collection = fileCollection.read().split('\n')
        fileCollection.close()
    document_number = len(collection) - 1
    for d in range(0, document_number):
        current_document = collection[d].split('@@@@@@@@@@')
        if current_document[0] not in category_list:
            category_list.add(current_document[0])
        document_labels.append(current_document[0])
        documents.append(current_document[1])
        corpus += current_document[1]
        corpus += " "

    print(list(category_list))
    # bag_of_words()
    information_gain_words()
    # mutual_information_words()
    # chi_square_words()
