from datasetLoader import DatasetLoader
from classifier import Classifier
from g3pEngine import ReplacementStrategy, g3pEngineConfiguration, ClassificationStrategy
from logger import log, logResult, logStatistics, setExecutionTimer
import terminalLogic
import time
import random
import numpy as np 
import pandas as pd


def launchExperiment(datasetFilePath,grammarFilePath,nFolds,maxGenerations,populationSize,
    crossProb, mutationProb, fitnessThreshold, replacementStrategy, classificationStrategy,
     positiveWeight, seed):
    random.seed(seed)
    np.random.seed(seed)

    setExecutionTimer()
    log("New Experiment")
    log("Config. details:")
    log("Dataset: " + datasetFilePath)
    log("Grammar: " + grammarFilePath)
    log("Num. Folds: " + str(nFolds))
    log("Max. Generations: " + str(maxGenerations))
    log("Population size: " + str(populationSize))  
    log(f"Fitness Threshold: {fitnessThreshold}")  
    log(f"Replacement Strategy: {replacementStrategy}")  
    log(f"Classification Strategy: {classificationStrategy}")  
    log(f"Seed: {seed}")  

     #Load dataset and extract vocabulary via text mining.
    log("Loading data...")
    dl = DatasetLoader()
    dataset = dl.loadfile(datasetFilePath)
    log("Data loaded. "+str(time.perf_counter())+"s.")

    log("Preparing text mining...")
    terminalLogic.VOCABULARY = dl.extractVocabulary()
    log("Text mining done. "+str(time.perf_counter())+"s.")
    log("Initializing classifier...")
    config = g3pEngineConfiguration(None, grammarFilePath, maxGenerations, populationSize, crossProb, 
                                    mutationProb, fitnessThreshold, replacementStrategy, classificationStrategy, positiveWeight)
    config.datasetFilePath = datasetFilePath
    config.seed = seed
    classifier = Classifier(dataset, nFolds, config)    
    log("Classifier initialized. "+str(time.perf_counter())+"s.")
    log("Training...")
    classifier.train()
    log("Classifier initialized. "+str(time.perf_counter())+"s.")
    log("Testing...")
    classifier.test()
    log("Experiment End. "+str(time.perf_counter())+"s.")
    log("-------------")
    return classifier.avgMeasures

def statistics():
    nSeeds = 1
    nExperiments = 1
    df = pd.read_csv('results.txt', sep=';')
    logStatistics(f"{nExperiments} experiments | {nSeeds} nSeeds", True)
    logStatistics(f"Avg.", True)
    for i in range(int(nExperiments/nSeeds)):
        ba = df["balancedAcc"].iloc[(nSeeds*i):nSeeds*(i+1)].mean()
        a = df["accuracy"].iloc[(nSeeds*i):nSeeds*(i+1)].mean()
        p = df["precision"].iloc[(nSeeds*i):nSeeds*(i+1)].mean()
        r = df["recall"].iloc[(nSeeds*i):nSeeds*(i+1)].mean()
        s = df["specificity"].iloc[(nSeeds*i):nSeeds*(i+1)].mean()
        t = df["time"].iloc[(nSeeds*i):nSeeds*(i+1)].mean()
        logStatistics(f"{ba};{a};{p};{r};{s};{t}",True)

    logStatistics(f"Std.", True)
    for i in range(int(nExperiments/nSeeds)):
        ba = df["balancedAcc"].iloc[(nSeeds*i):nSeeds*(i+1)].std()
        a = df["accuracy"].iloc[(nSeeds*i):nSeeds*(i+1)].std()
        p = df["precision"].iloc[(nSeeds*i):nSeeds*(i+1)].std()
        r = df["recall"].iloc[(nSeeds*i):nSeeds*(i+1)].std()
        s = df["specificity"].iloc[(nSeeds*i):nSeeds*(i+1)].std()
        t = df["time"].iloc[(nSeeds*i):nSeeds*(i+1)].std()
        logStatistics(f"{ba};{a};{p};{r};{s};{t}",True)

    logStatistics(f"Max.", True)
    for i in range(int(nExperiments/nSeeds)):
        ba = df["balancedAcc"].iloc[(nSeeds*i):nSeeds*(i+1)].max()
        a = df["accuracy"].iloc[(nSeeds*i):nSeeds*(i+1)].max()
        p = df["precision"].iloc[(nSeeds*i):nSeeds*(i+1)].max()
        r = df["recall"].iloc[(nSeeds*i):nSeeds*(i+1)].max()
        s = df["specificity"].iloc[(nSeeds*i):nSeeds*(i+1)].max()
        t = df["time"].iloc[(nSeeds*i):nSeeds*(i+1)].max()
        logStatistics(f"{ba};{a};{p};{r};{s};{t}",True)

    logStatistics(f"Min.", True)
    for i in range(int(nExperiments/nSeeds)):
        ba = df["balancedAcc"].iloc[(nSeeds*i):nSeeds*(i+1)].min()
        a = df["accuracy"].iloc[(nSeeds*i):nSeeds*(i+1)].min()
        p = df["precision"].iloc[(nSeeds*i):nSeeds*(i+1)].min()
        r = df["recall"].iloc[(nSeeds*i):nSeeds*(i+1)].min()
        s = df["specificity"].iloc[(nSeeds*i):nSeeds*(i+1)].min()
        t = df["time"].iloc[(nSeeds*i):nSeeds*(i+1)].min()
        logStatistics(f"{ba};{a};{p};{r};{s};{t}",True)


if __name__ == "__main__":
    nSeeds = 10
    launchExperiment(datasetFilePath = 'HallSinNA.csv',grammarFilePath = 'assets/ml.xml',nFolds = 2,maxGenerations = 10,populationSize = 30,crossProb=0.9, mutationProb=0.1, fitnessThreshold = 0.4, replacementStrategy = ReplacementStrategy.NEWPOPULATION, classificationStrategy=ClassificationStrategy.CBA, positiveWeight=1.5, seed=1)
