from collections import Counter
import math
import codecs
import numpy as np

document_number = 0
category_document_number = list()
category_list = set()
dictionary = dict()
corpus = ""
words = []
collection_contain = []
category_contain = []
category_entropy = 0
score_information_gain = []
score_mutual_information = []
score_chi_square = []
information_gain_contain_probability = 0
information_gain_not_contain_probability = 0
word_score_mutual_information = 0
max_score_mutual_information = 0
max_category_mutual_information = 0
word_score_chi_square = 0
max_score_chi_square = 0
max_category_chi_square = 0


def category_document_counter():

    for i in range(0, len(category_list)):
        category_document_number.append(len(dictionary.get(category_list[i])))


def collection_counter():
    global collection_contain
    collection_contain = np.zeros(len(words))
    for i in range(0, document_number):
        current = Counter(collection[i].split('@@@@@@@@@@')[1].split())
        keys = list(current)
        for j in range(0, len(current)):
            index = words.index(keys[j])
            collection_contain[index] += 1


def category_counter():
    global category_contain
    category_contain = np.zeros(shape=(len(words), len(category_list)))
    for k in range(0, len(category_list)):
        current_category_documents = list(dictionary.get(category_list[k]))
        for i in range(0, len(current_category_documents)):
            current = Counter(current_category_documents[i].split())
            keys = list(current)
            for j in range(0, len(current)):
                index = words.index(keys[j])
                category_contain[index][k] += 1


def entropy():
    global category_entropy
    for i in range(0, len(category_list)):
        probability = category_document_number[i] / document_number
        category_entropy += probability * math.log10(probability)
    category_entropy *= -1


def information_gain(word, category):
    global information_gain_contain_probability, information_gain_not_contain_probability
    probability = category_contain[word][category] / collection_contain[word]
    try:
        information_gain_contain_probability += probability * math.log10(probability)
    except:
        error = 1
    probability = (category_document_number[category] - category_contain[word][category]) / (document_number - collection_contain[word])
    try:
        information_gain_not_contain_probability += probability * math.log10(probability)
    except:
        error = 1


def mutual_information(word, category):
    global word_score_mutual_information, max_score_mutual_information, max_category_mutual_information
    p_w_c = category_contain[word][category] / collection_contain[word]
    p_w = collection_contain[word] / document_number
    p_c = category_document_number[category] / document_number
    try:
        probability = (p_w_c * p_w) * math.log10(p_w_c / p_c)
        word_score_mutual_information += p_c * probability
        if probability > max_score_mutual_information:
            max_score_mutual_information = probability
            max_category_mutual_information = category
    except:
        error = 1


def chi_square(word, category):
    global word_score_chi_square, max_score_chi_square, max_category_chi_square
    n_i_w = category_contain[word][category]
    n_i_w_n = category_document_number[category] - category_contain[word][category]
    n_i_n_w = math.fsum(category_contain[word]) - category_contain[word][category]
    n_i_n_w_n = math.fsum(category_document_number) - category_document_number[category] - n_i_n_w
    p_c = category_document_number[category] / document_number
    probability = (document_number * math.pow((n_i_w * n_i_n_w_n - n_i_w_n * n_i_n_w), 2)) / ((n_i_w + n_i_w_n) * (n_i_n_w + n_i_n_w_n) * (n_i_w + n_i_n_w) * (n_i_w_n + n_i_n_w_n))
    try:
        word_score_chi_square += p_c * probability
        if probability > max_score_chi_square:
            max_score_chi_square = probability
            max_category_chi_square = category
    except:
        error = 1


if __name__ == '__main__':

    with open('/Users/negin/Downloads/SNLP/HW1/HAM2-corpus-short-with-tag-selected.txt', 'r') as fileCollection:
        collection = fileCollection.read().split('\n')
        fileCollection.close()
    document_number = len(collection)-1
    for d in range(0, document_number):
        current_category = collection[d].split('@@@@@@@@@@')[0]
        current_document = collection[d].split('@@@@@@@@@@')[1]
        corpus += current_document
        corpus += " "
        if current_category in category_list:
            documents = list(dictionary.get(current_category))
            documents.append(current_document)
            new_entry = {current_category: documents}
            dictionary.update(new_entry)
        else:
            category_list.add(current_category)
            new_entry = {current_category: [current_document]}
            dictionary.update(new_entry)

    words = list(Counter(corpus.split()).keys())
    category_list = list(category_list)
    print("Number Of Documents: ", document_number)
    category_document_counter()
    print("Counting Documents Containing w ...")
    collection_counter()
    print("Counting Documents In Category c Containing w ...")
    category_counter()
    entropy()
    print("Calculating IG, MI, Chi-Square ...")

    for w in range(0, len(words)):
        information_gain_contain_probability = 0
        information_gain_not_contain_probability = 0
        word_score_mutual_information = 0
        max_score_mutual_information = 0
        max_category_mutual_information = 0
        word_score_chi_square = 0
        max_score_chi_square = 0
        max_category_chi_square = 0
        for c in range(0, len(category_list)):
            information_gain(w, c)
            mutual_information(w, c)
            chi_square(w, c)

        temp = collection_contain[w] / document_number
        word_score = category_entropy + temp * information_gain_contain_probability + (1-temp) * information_gain_not_contain_probability
        score_information_gain.append(word_score)

        score_mutual_information.append([word_score_mutual_information, max_category_mutual_information, max_score_mutual_information])
        score_chi_square.append([word_score_chi_square, max_category_chi_square, max_score_chi_square])

    print("Writing Result_Information_Gain.csv")
    score_information_gain = list(zip(words, score_information_gain))
    score_information_gain = sorted(score_information_gain, key=lambda l1: l1[1], reverse=True)
    column_title = "W, Score Of Algorithm\n"
    Result_Information_Gain = codecs.open("/Users/negin/Desktop/Result_Information_Gain.csv", "wb", "utf-8")
    Result_Information_Gain.write(column_title)
    for w in range(0, 100):
        line = str(score_information_gain[w][0])
        line += ', '
        line += str(score_information_gain[w][1])
        line += '\n'
        Result_Information_Gain.write(line)
    Result_Information_Gain.close()

    print("Writing Result_Mutual_Information.csv")
    score_mutual_information = list(zip(words, score_mutual_information))
    score_mutual_information = sorted(score_mutual_information, key=lambda l2: l2[1], reverse=True)
    column_title = "W, Score Of Algorithm, Main Domain, Score Of The Main Domain\n"
    Result_Mutual_Information = codecs.open("/Users/negin/Desktop/Result_Mutual_Information.csv", "wb", "utf-8")
    Result_Mutual_Information.write(column_title)
    for w in range(0, 100):
        line = str(score_mutual_information[w][0])
        line += ', '
        line += str(score_mutual_information[w][1][0])
        line += ', '
        line += str(category_list[score_mutual_information[w][1][1]])
        line += ', '
        line += str(score_mutual_information[w][1][2])
        line += '\n'
        Result_Mutual_Information.write(line)
    Result_Mutual_Information.close()

    print("Writing Result_Chi_Square.csv")
    score_chi_square = list(zip(words, score_chi_square))
    score_chi_square = sorted(score_chi_square, key=lambda l3: l3[1], reverse=True)
    Result_Chi_Square = codecs.open("/Users/negin/Desktop/Result_Chi_Square.csv", "wb", "utf-8")
    Result_Chi_Square.write(column_title)
    for w in range(0, 100):
        line = str(score_chi_square[w][0])
        line += ', '
        line += str(score_chi_square[w][1][0])
        line += ', '
        line += str(category_list[score_chi_square[w][1][1]])
        line += ', '
        line += str(score_chi_square[w][1][2])
        line += '\n'
        Result_Chi_Square.write(line)
    Result_Chi_Square.close()
