import csv
import getopt
import math
import sys

import numpy as np
import pandas as pd


def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[0] not in separated):
            separated[vector[0]] = []
        separated[vector[0]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    strddev = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(strddev)


def summarize(dataset):
    summary = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summary[:1]
    return summary


def prior(dataset):
    test1 = data[:, 0:1]
    test2 = test1.tolist()
    number_a = test2.count([1.0])
    number_b = len(data) - number_a
    prior_a = number_a / len(data)
    prior_b = number_b / len(data)
    return prior_a, prior_b


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summary = {}
    for classValue, instances in separated.items():
        summary[classValue] = summarize(instances)
    return summary


def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = input_vector[i + 1]
            probabilities[classValue] *= calculate_probability(x, mean, stdev)
    return probabilities


def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for classValue, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = classValue
    return best_label


def get_predictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def get_accuracy(test_set, predictions):
    miss_class = 0
    for i in range(len(test_set)):
        if test_set[i][0] != predictions[i]:
            miss_class += 1

    return miss_class


def write_data(row1, row2, row3, filename):
    with open(filename, 'w', newline='') as outfile:
        output_file = csv.writer(outfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        line = []
        for value in row1:
            line.append(value)
        output_file.writerow(line)
        line = []
        for value in row2:
            line.append(value)
        output_file.writerow(line)
        output_file.writerow(row3)
        print("Output file written in " + filename)


if __name__ == '__main__':
    argv = sys.argv[1:]
    outputfile = ""
    inputfile = ""
    if argv.__len__() != 0:
        try:
            opts, args = getopt.getopt(argv, "h:i:o:", ["data=", "output="])
        except getopt.GetoptError:
            print('usage: Naive_bayes.py -i||--data <inputfile> -o||--output <outputfile>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print('test.py -i <inputfile> -o <outputfile>')
                sys.exit()
            elif opt in ("-i", "--data"):
                inputfile = arg
                if '.tsv' not in inputfile:
                    inputfile = inputfile + '.tsv'
            elif opt in ("-o", "--output"):
                outputfile = arg
                if '.tsv' not in outputfile:
                    outputfile = outputfile + '.tsv'
    else:
        inputfile = input("Enter data file name: ")
        if '.tsv' not in inputfile:
            inputfile = inputfile + '.tsv'
        outputfile = input("Enter Output file name: ")
        if '.tsv' not in outputfile:
            outputfile = outputfile + '.tsv'
    if inputfile == "":
        inputfile = input("Enter data file name: ")
        if '.tsv' not in inputfile:
            inputfile = inputfile + '.tsv'
    if outputfile == "":
        outputfile = input("Enter Output file name: ")
        if '.tsv' not in outputfile:
            outputfile = outputfile + '.tsv'
    data = pd.read_csv(inputfile, sep='\t', header=None)
    data.columns = ['Class', 'x1', 'x2']
    data["Class"] = np.where(data["Class"] == 'A', 1, 0)
    data = data.values
    prior_Prob = prior(data)
    summaries = summarize_by_class(data)
    predictions = get_predictions(summaries, data)
    missclass = get_accuracy(data, predictions)
    row1 = []
    row2 = []
    row1.append(summaries[1.0][0][0])
    row1.append(math.pow(summaries[1.0][0][1], 2))
    row1.append(summaries[1.0][1][0])
    row1.append(math.pow(summaries[1.0][1][1], 2))
    row1.append(prior_Prob[0])
    print(row1)
    row2.append(summaries[0.0][0][0])
    row2.append(math.pow(summaries[0.0][0][1], 2))
    row2.append(summaries[0.0][1][0])
    row2.append(math.pow(summaries[0.0][1][1], 2))
    row2.append(prior_Prob[1])
    print(row2)
    print(missclass)
    write_data(row1, row2, [missclass], outputfile)
