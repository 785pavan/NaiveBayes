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
    return strddev


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[:1]
    return summaries


def prior(dataset):
    test1 = data[:, 0:1]
    test2 = test1.tolist()
    number_A = test2.count([1.0])
    number_B = len(data) - number_A
    prior_A = number_A / len(data)
    prior_B = number_B / len(data)
    return (prior_A, prior_B)


def summarizeByClass(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i + 1]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    miss_class = 0
    for i in range(len(testSet)):
        if testSet[i][0] != predictions[i]:
            miss_class += 1

    return (miss_class)


if __name__ == '__main__':
    argv = sys.argv[1:]
    iter_req = 0
    outputfile = ""
    inputfile = ""
    if argv.__len__() != 0:
        try:
            opts, args = getopt.getopt(argv, "hit:o:", ["data=", "output=", "iter="])
        except getopt.GetoptError:
            print('usage: perceptron.py -i||--data <inputfile> -o||--output <outputfile>')
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
            elif opt in ('-t', "--iter"):
                iter_req = int(arg)
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
    summaries = summarizeByClass(data)
    predictions = getPredictions(summaries, data)
    missclass = getAccuracy(data, predictions)
    print(prior_Prob, summaries, missclass)
