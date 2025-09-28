

import random
random.seed(1)
import numpy as np 

from candidatePaper import CandidatePaper

VOCABULARY = [...]
PAPERTYPES = ["Conference Proceeding","Journal"]

def numValue(min=0, max=5):
    """
    Creates a random  intnumValue between min and max
    """
    return str(np.random.randint(min, max))

def numFloatValue(min=0, max=5):
    """
    Creates a random float numValue between min and max
    """
    return str(np.random.uniform(min, max))

def candidateStudyValue(terminal):
    """
    Returns the label CandidateStudy
    """
    return bool(np.random.randint(0,2)) == 1

def JCRValue(terminal):
    return float(numFloatValue(terminal.minValue, terminal.maxValue))

def nCitesValue(terminal):
    return int(numValue(terminal.minValue, terminal.maxValue))

def nAuthorsValue(terminal):
    return int(numValue(terminal.minValue, terminal.maxValue))

def yearValue(terminal):
    return int(numValue(terminal.minValue, terminal.maxValue))

def paperTypeValue(terminal):
    return np.random.choice(PAPERTYPES)

def vocabularyValue(terminal):
    nWords = np.random.randint(1, 5)
    return np.random.choice(list(VOCABULARY),nWords if nWords <= len(list(VOCABULARY)) else len(list(VOCABULARY)))
    #return random.sample(list(VOCABULARY),nWords)

def relOpNot(values: list):
    return not values[0]

def relOpAnd(values: list):
    condition = True
    for v in values:
        if v == False:
            return False
        condition &= v
    return condition

def relOpOr(values: list):
    condition = False
    for v in values:
        if v == True:
            return True
        condition |= v
    return condition

def cmpEquals(values: list):
    return values[0] == values[1]

def cmpNotEquals(values: list):
    return values[0] != values[1]

def cmpLess(values: list):
    return values[0] < values[1]

def cmpGreater(values: list):
    return values[0] > values[1]

def cmpLessOrEquals(values: list):
    return values[0] <= values[1]

def cmpGreaterOrEquals(values: list):
    return values[0] >= values[1]

def cmpContainsAll(values: list):
    try:
        return all(word.lower() in values[0].lower() for word in values[1])
    except:
        return False

def cmpContainsAny(values: list):
    try:
        return any(word.lower() in values[0].lower() for word in values[1])
    except:
        return False

def opJCR(paper: CandidatePaper):
    return paper.JCR

def opnCites(paper: CandidatePaper):
    try:
        return int(paper.nCites)
    except:
        return 0

def opnAuthors(paper: CandidatePaper):
    try:
        return int(paper.nAuthors)
    except:
        return 0
    
def opYear(paper: CandidatePaper):
    try:
        return int(paper.year)
    except:
        return 0

def opIsCandidateStudy(paper: CandidatePaper = None):
    return paper.isCandidate == 'yes' or paper.isCandidate == '1'

def opTitle(paper: CandidatePaper = None):
    return paper.documentTitle

def opAbstract(paper: CandidatePaper = None):
    return paper.abstract

def opTitleAbstract(paper: CandidatePaper = None):
    return paper.documentTitle + " " + paper.abstract

def opPaperType(paper: CandidatePaper = None):
    return paper.aggregationType
