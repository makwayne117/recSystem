#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

def dotProduct(d1: FeatureVector, d2: WeightVector) -> float:
        sol = []
        for k in d1.keys():
            sol.append(d1.get(k,0)*d2.get(k,0))
        return sum(sol)
    
def increment(d1: WeightVector, scale: float, d2: FeatureVector):
    for f, v in d2.items():
        d1[f] = d1.get(f,0) - scale * v
        
def calculateGradient(y, x, weights):
    margin = y * dotProduct(x, weights)
    gradient = {}
    if 1-margin > 0:
        for feature, value in x.items():
            gradient[feature] =  -y * value
    else:
        for feature, value in x.items():
            gradient[feature] =  0
    return gradient

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE 
    final_vector = {}
    words = x.split()
    for w in words:
        if w not in list(final_vector.keys()):  
            final_vector[w] = 1
        else:
            final_vector[w] += 1
    return final_vector
    raise Exception("Not implemented yet")
    # END_YOUR_CODE


############################################################
# Problem 2b: stochastic gradient descent

T = TypeVar('T')


    
def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    weights = {}  
    
    
    for epoch in range(numEpochs):
        for x, y in trainExamples:
            features = featureExtractor(x)
            gradient = calculateGradient(y, features, weights)
            increment(weights, eta, gradient)
        trainErrorCount = 0
        validErrorCount = 0

        for x, y in trainExamples:
            if y * dotProduct(featureExtractor(x), weights) < 0:
                trainErrorCount += 1
        for x, y in validationExamples:
            if y * dotProduct(featureExtractor(x), weights) < 0:
                validErrorCount += 1
        trainingError = trainErrorCount/len(trainExamples)
        validError = validErrorCount/len(validationExamples)

        print("Epoch", epoch + 1, ", training error:", trainingError, ", validation error:", validError)
        
    return weights

    



############################################################
# Problem 2c: generate test case



def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    random.seed(42)  # Ensure reproducibility


    def generateExample() -> Example:
        phi = {}
        for key in weights.keys():
            phi[key] = random.uniform(-1,1)
        score = dotProduct(phi, weights)  
        if score >=0:
            y=1
        else:
            y=-1 
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 2d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE
        ngrams = {}
        x_new = x.replace(" ", "")
        idx = 0
        
        while idx+n <= len(x_new):
            if x_new[idx:idx+n] in ngrams:   
                ngrams[x_new[idx:idx+n]]+=1
            else:
                ngrams[x_new[idx:idx+n]]=1
            idx+=1
        return ngrams
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    return extract


############################################################
# Problem 2e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in cs256homework2.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))
