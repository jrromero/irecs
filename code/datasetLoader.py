from candidatePaper import CandidatePaper
import pandas as pd
import csv
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

"""This class is prepared to load, save files and to perform text mining
following Zu&Menzies approach.
Reference: https://github.com/fastread/src/blob/master/src/util/mar.py"""
class DatasetLoader():
    def __init__(self):
        self.fea_num = 10        
        self.filename=""
        self.name=self.filename.split(".")[0]
        self.body={}
        self.papers=[]
        self.voc=[]

    def loadCSV(filepath):
        papers = []
        frame = pd.read_csv(filepath)
        #print(frame)
        for index, row in frame.iterrows():
            paper = CandidatePaper(row[0],row[1],row[2],row[3],row[4])
            papers.append(paper)
        return papers

    def loadfile(self, filename):
        self.filename = filename
        with open("assets/" + str(self.filename), "r", encoding="UTF-8-sig") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]
        fields = ["Document Title", "Abstract", "Year", "PDF Link","doi","nCites","authorCount","aggregationType"]
        header = content[0]
        for field in fields:
            try:
                ind = header.index(field)
                self.body[field] = [c[ind] for c in content[1:]]
            except:
                pass
        try:
            ind = header.index("label")
            self.body["label"] = [c[ind] for c in content[1:]]
        except:
            self.hasLabel=False
            self.body["label"] = ["unknown"] * (len(content) - 1)
        try:
            ind = header.index("doi")
            self.body["doi"] = [c[ind] for c in content[1:]]
        except:
            self.body["doi"]=['undetermined']*(len(content) - 1)
        try:
            ind = header.index("aggregationType")
            self.body["aggregationType"] = [c[ind] for c in content[1:]]
        except:
            self.body["aggregationType"]=['undetermined']*(len(content) - 1)
        try:
            ind = header.index("PDF Link")
            self.body["PDF Link"] = [c[ind] for c in content[1:]]
        except:
            self.body["PDF Link"]=['undetermined']*(len(content) - 1)
        try:
            ind = header.index("nCites")
            self.body["nCites"] = [c[ind] for c in content[1:]]
        except:
            self.body["nCites"]=[0]*(len(content) - 1)
        try:
            ind = header.index("authorCount")
            self.body["authorCount"] = [c[ind] for c in content[1:]]
        except:
            self.body["authorCount"]=[0]*(len(content) - 1)
        try:
            ind = header.index("time")
            self.body["time"] = [c[ind] for c in content[1:]]
        except:
            self.body["time"]=[0]*(len(content) - 1)

        self.papers = []
        
        for index in range(len(self.body["Document Title"])) :
            self.papers.append(CandidatePaper(self.body["Document Title"][index], self.body["Abstract"][index],
            self.body["Year"][index],self.body["PDF Link"][index], self.body["label"][index],
            self.body["doi"][index],self.body["nCites"][index],self.body["authorCount"][index],
            self.body["aggregationType"][index]))

        return self.papers
    
    
    def extractVocabulary(self):
        ### Combine title and abstract for training ###########
        contentAll = [paper.documentTitle + " " + paper.abstract for paper in self.papers]

        contentPos = [paper.documentTitle + " " + paper.abstract for paper in
                  self.papers if paper.getIsCandidate()]
        
        contentNeg = [paper.documentTitle + " " + paper.abstract for paper in
                  self.papers if not paper.getIsCandidate()]

        # vocPos = self.getRelevantWords(contentPos)
        vocNeg = self.getRelevantWords(contentNeg)
        self.voc = self.getRelevantWords(contentAll)
        
        s = set(vocNeg)
        # exclusiveVocPos = [x for x in vocPos if x not in s]
        #print(vocPos)
        #exclusiveVocPos = ["predict","fault","defect"]
        return self.voc
    
    def getRelevantWords(self, content):
        ### Feature selection by tfidf in order to keep vocabulary ###
        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
                                sublinear_tf=False,decode_error="ignore",)
        tfidf = tfidfer.fit_transform(content)
        weight = tfidf.sum(axis=0).tolist()[0]
        kept = np.argsort(weight)[-self.fea_num:]
        # Define the stemmer
        porter = nltk.stem.PorterStemmer()
        voc = np.array(list(tfidfer.vocabulary_.keys()))[np.argsort(list(tfidfer.vocabulary_.values()))][kept]
        tokens_stemmed = [w.replace(w, porter.stem(w)) for w in voc]
        voc = list(dict.fromkeys(tokens_stemmed))
        return voc