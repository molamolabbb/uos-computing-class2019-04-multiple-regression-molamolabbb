#!/usr/bin/env python3

import csv
from math import erf, sqrt
from randomness import randnr

rand = randnr(3)
def in_random_order(x):
    "Returns an iterator that presents the list x in a random order"
    indices = [i for i, _ in enumerate(x)]
    # "inside-out" Fisher-Yates shuffle. Step through the list, and at
    # each point, exchange the current element with a random element
    # in the list (including itself)
    for i in range(len(indices)):
        j = (rand.randint() // 65536) % (i+1)  # The lower bits of our random generator are correlated!
        indices[i], indices[j] = indices[j], indices[i]
    for i in indices:
        yield x[i]

# Functions go here!

if __name__ == "__main__":
    # Here, we load the boston dataset
    boston = csv.reader(open('boston.csv'))  # The boston housing dataset in csv format
    # First line contains the header, short info for each variable
    header = boston.__next__()  # In python2, you might need boston.next() instead
    # Data will hold the 13 data variables, target is what we are trying to predict
    data, target = [], []
    for row in boston:
        # All but the last are the data points
        data.append([float(r) for r in row[:-1]])
        # The last is the median house value we are trying to predict
        target.append(float(row[-1]))
    # Now, use the dataset with your regression functions to answer the exercise questions
    print("Names of the columns")
    print(header)
    print("First row of data ->variable to predict")
    print(data[0], " -> ", target[0])

    # The alpha parameter must be tuned low so that we don't jump too far
    start = # take the starting parameters as 0 for beta0 then the intercepts from the individual fits
    output = stochastic_minimize(loss, dloss, data, target, start, 1e-6)

    # Also need to calculate the full R^2!

    # Example of writing out the results.txt file
    fout = open('results.txt', 'w')
    for param in output:
        fout.write('%f\n' % (param))  # One line per variable
    fout.close()

