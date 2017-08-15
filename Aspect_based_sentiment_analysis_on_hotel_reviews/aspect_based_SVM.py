
def get_stopwords():
    stopword_file_contents = [line.strip() for line in open("stopword_list.txt", 'r')]
    stopwords = [[word.lower() for word in text.split()] for text in stopword_file_contents]
    s = []

    for i in range(len(stopwords)):
        row = stopwords[i]
        for j in range(len(row)):
            s.append(row[j])
    return s


def get_polarity_accuracy(y_predicted, y_test):
    count_positive = 0
    count_negative = 0
    count_neutral = 0
    for i in range(len(y_test)):
        if y_predicted[i] > '3.0' and y_test[i] > '3.0':
            count_positive += 1

        if y_predicted[i] == '3.0' and y_test[i] == '3.0':
            count_neutral += 1
        if y_predicted[i] < '3.0' and y_test[i] < '3.0':
            count_negative += 1

    # print 'count_neutral',count_neutral,'count_negative',count_positive,'count_positive',count_negative
    count = count_neutral + count_negative + count_positive
    polarity_accuracy = (count * 100) / len(y_predicted)
    print 'polarity accuracy =', polarity_accuracy


def findWholeWord(w):
    import re
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def prepare_data(aspect, aspect_file, path):
    import numpy as np

    import os
    import json
    import datetime

    from sklearn.feature_extraction.text import CountVectorizer

    files = []  # names of all json data files
    review_contents = []  # stores all review paragraphs from all json files
    overall_ratings = []  # stores the 'overall rating's from all json files
    bigrams = []  # bigrams of all review contents
    stopwords = get_stopwords()

    # prepare text data for classification

    bigram_vectorizer = CountVectorizer(ngram_range=(2, 3), token_pattern=r'\b\w+\b', min_df=2, max_df=0.20,
                                        stop_words=stopwords)
    analyze = bigram_vectorizer.build_analyzer()


    files = os.listdir(path)  # load all json files, Snames are stored in this files variable
    print 'number of json files =', len(files)  # print how many files are in the directory
    t_start_file_read = datetime.datetime.now()
    # print 'json file reading start'
    for i in range(len(files)):
        # print files[i]
        with open(path + '\\' + files[i]) as json_file:  # read each json file
            data = json.load(json_file)
            for reviews in data['Reviews']:
                # use the lowecase version fbecause meaning stays same
                if reviews.get("Ratings", {}).get(aspect):
                    content = reviews['Content']  # store the review paragraph for data
                    review_contents.append(content.lower())
                    overall_ratings.append(reviews.get("Ratings", {}).get(aspect))  # store overall rating for label

    # print len(overall_ratings)
    print 'number of reviews =', len(review_contents)

    t_finish_file_read = datetime.datetime.now()
    c = t_finish_file_read - t_start_file_read
    #print 'time to read json ', divmod(c.days * 86400 + c.seconds, 60)

    print 'starting bigram'
    X = bigram_vectorizer.fit_transform(review_contents)  # prepare feature vector by using bigram
    print 'bigram finished'
    print 'number of features =', bigram_vectorizer.vocabulary_.__len__()

    t_finish_bigram = datetime.datetime.now()
    c = t_finish_bigram - t_finish_file_read
    #print 'time to create 2-3 gram= ', divmod(c.days * 86400 + c.seconds, 60)

    aspect_data = []
    with open(aspect_file, 'r') as f:
        for line in f:
            for word in line.split():
                aspect_data.append(word)

    X = bigram_vectorizer.fit_transform(review_contents)
    #print bigram_vectorizer.get_feature_names()
    # print X
    list_to_remove = []
    print bigram_vectorizer.vocabulary_.__len__()
    for i in range(bigram_vectorizer.vocabulary_.__len__()):
        bigram_key = bigram_vectorizer.vocabulary_.keys().__getitem__(i)
        tag = 0
        for j in range(len(aspect_data)):
            #print 'bigram_key=', bigram_key, ' and room_data[', j, ']', aspect_data[j]

            if findWholeWord(aspect_data[j])(bigram_key):
                #print bigram_key, 'has found', aspect_data[j]
                tag = 1
                break
            else:
                tag = 0
                #print 'not found'
        if tag == 0:
            list_to_remove.append(bigram_key)
    for k in range(len(list_to_remove)):
        bigram_vectorizer.vocabulary_.__delitem__(list_to_remove[k])
    X = bigram_vectorizer.fit_transform(review_contents)
    y = np.array(overall_ratings)
    return X, y

def run_aspect_SVM(aspect, aspect_file, path):
    from sklearn.model_selection import train_test_split
    from sklearn import svm

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_fscore_support
    import datetime

    # covert label array to numpy array
    X, y = prepare_data(aspect, aspect_file, path)
    # apply SVM

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42)  # split dataset into training and testing

    # print 'length of X_train', np.shape(X_train)
    print 'length of y_train =', len(y_train)
    # print 'X_test', X_test
    print 'length of y_test =', len(y_test)  # this contains true results

    clf = svm.SVC(gamma=0.01, C=100)  # define classifier
    print 'training started'
    t_start_training = datetime.datetime.now()
    clf.fit(X_train, y_train)  # train data
    t_finish_training = datetime.datetime.now()
    c = t_finish_training - t_start_training
    print 'training finished, time for training =', divmod(c.days * 86400 + c.seconds, 60)

    print 'prediction starting'
    y_predicted = clf.predict(X_test)  # predict result
    t_end_prediction = datetime.datetime.now()
    c = t_end_prediction - t_finish_training
    print 'prediction finished, time needed =', divmod(c.days * 86400 + c.seconds, 60)

    # print X_test
    # print 'predictions', y_predicted

    # performance measurements
    # print confusion_matrix(y_test, y_predicted)
    print 'accuracy=', accuracy_score(y_test,
                                      y_predicted)  # traditional accuracy, where mismatch = belongs to different class
    get_polarity_accuracy(y_predicted, y_test)


    y_predicted = clf.predict(X_train)
    print 'accuracy of training set=', accuracy_score(y_train,
                                                      y_predicted), ' polarity accuracy=', get_polarity_accuracy(
        y_predicted, y_train)
    # print precision_recall_fscore_support(y_test, y_predicted, average='macro')
    # print precision_recall_fscore_support(y_test, y_predicted, average='micro')
    # print precision_recall_fscore_support(y_test, y_predicted, average='weighted')


def get_aspect_file_name( aspect):

    aspect_file = "overall.txt"  # set default value
    if aspect == 'Rooms':
        aspect_file = "room.txt"
    elif aspect == 'Cleanliness':
        aspect_file = "cleanliness.txt"
    elif aspect == 'Service':
        aspect_file = "service.txt"
    elif aspect == 'Value':
        aspect_file = "value.txt"
    elif aspect == 'Overall':
        aspect_file = "overall.txt"
    elif aspect == 'Location':
        aspect_file = "location.txt"
    return aspect_file

def get_path_of_dataset(data_num):
    path = "H:\study\ML_Lab contents\\ten_data"  # set default value
    # set path of dataset
    if (data_num == '10'):
        path = "H:\study\ML_Lab contents\\ten_data"
    elif (data_num == '50'):
        path = "H:\study\ML_Lab contents\\fifty_data"
    elif (data_num == '200'):
        path = "H:\study\ML_Lab contents\\two_hundred_data"
    return path

def main():

    print 'how many json file?'
    data_num=raw_input()

    print 'which aspect? Rooms, Service, Value, Cleanliness, Overall, Location'
    aspect=raw_input()

    path = get_path_of_dataset(data_num)
    aspect_file = get_aspect_file_name(aspect)
    print 'caling svm with ', aspect, aspect_file, 'and', path

    run_aspect_SVM(aspect, aspect_file, path)


if __name__=='__main__':
    main()