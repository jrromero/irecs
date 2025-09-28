
from typing import List
import terminalLogic

class Evaluator:
    """
    Tests terminals from an individual with the set of individuals
    """
    def __init__(self, individual, instances: List = []):
        self.individual = individual
        self.terminals = individual.getSyntaxTree().getTerminals()
        self.instances = instances
        self.satisfiedInst = 0
        self.satisfiedAntc = 0
        self.satisfiedConsq = 0
        self.nInstances = len(instances)
        self.support = 0.0
        self.confidence = 0.0
        self.lift = 0.0

    def evaluate(self, verbose = False):
        """Calculates support and confidence for an individual (rule) with the set of Candidates (instances)"""
        if verbose:
            for terminal in self.terminals:
                print(terminal, end='')
                print(" ", end='')
            print()
        self.satisfiedInst = 0
        self.satisfiedAntc = 0
        self.support = 0.0
        self.confidence = 0.0

        for paper in self.instances :
            antcResult, consqResult, ruleResult = self.evaluatePaper(paper)
            
            if antcResult:
                self.satisfiedAntc += 1
            if ruleResult:
                self.satisfiedInst += 1
            if consqResult:
                self.satisfiedConsq += 1

            if verbose:  
                print("Antecedent result: ",end="")
                print(antcResult,end=" | ")
                print("Consequent result: ",end="")
                print(consqResult,end=" | ")                
                print("Rule result: ",end="")                
                print(ruleResult)
        if self.satisfiedInst > 0:
            self.support = self.satisfiedInst / self.nInstances
            if self.satisfiedAntc > 0 :
                self.confidence = self.satisfiedInst / self.satisfiedAntc
            self.lift = (self.satisfiedInst*self.nInstances) / (self.satisfiedAntc * self.satisfiedConsq)
            if verbose:  
                print("Support: ",end="")
                print(self.support,end=" | ")
                print("Confidence: ",end="")                
                print(self.confidence)
                print("-----")
        
        self.individual.support = self.support
        self.individual.confidence = self.confidence
        self.individual.lift = self.lift
        self.individual.suppAnt = self.satisfiedAntc
        self.individual.suppCon = self.satisfiedConsq
        self.individual.laplace = self.calculateLaplaceAccuracy(self.individual, self.satisfiedAntc, self.satisfiedInst)

        return self.individual
    
    def evaluatePaper(self, paper, verbose = False):
        """Calculates support and confidence for an individual (rule) with a single paper"""
        if verbose:
            for terminal in self.terminals:
                print(terminal, end='')
                print(" ", end='')
            print()
        
        total = len(self.terminals)
        i = 0            
        evaluations = []
        pendingTerminals = []
        while i < total:
            if hasattr(terminalLogic, self.terminals[i].code):
                if(str(self.terminals[i].code).startswith("cmp")):
                    comparator = self.terminals[i]
                    values = []
                    for j in range(int(comparator.nOperators)):
                        i += 1
                        if str(self.terminals[i].code).startswith("op"):
                            values.append(getattr(terminalLogic, self.terminals[i].code)(paper)) 
                        else:  # if it is not an operator, then it is a literal value already replaced during individual creation
                            values.append(self.getLiteralValue(self.terminals[i]))
                    result = getattr(terminalLogic, comparator.code)(values)
                    evaluations.append(result)
                    # print(result)
                else: # relational logical operator is set to be apply
                    pendingTerminals.append(self.terminals[i])
                i += 1
        
        antcResult = evaluations[0]
        consqResult = evaluations[evaluations.__len__()-1]
        
        #If there are pending terminals to evaluate
        if pendingTerminals.__len__() > 0 :
            #evaluate backwards
            pendingTerminals.reverse() 
            antcEvaluations = evaluations[0:evaluations.__len__()-1] #evaluate backwards and remove consq result
            antcEvaluations.reverse()
            #now execute pending terminals (e.g.:and, or, not, ...)                      

            antcResult = True            
            lastEvaluation = 0
            for i in range(pendingTerminals.__len__()):
                nOperators = int(pendingTerminals[i].nOperators)
                currentResult = getattr(terminalLogic, pendingTerminals[i].code)(antcEvaluations[lastEvaluation:lastEvaluation+nOperators])
                lastEvaluation = lastEvaluation+nOperators-1
                antcEvaluations[lastEvaluation] = currentResult
                antcResult &= currentResult

        ruleResult = antcResult and consqResult
        return antcResult, consqResult, ruleResult     

    def getLiteralValue(self, operator):
        if operator.type == "int":
            return int(operator.symbol)
        elif operator.type == "bool":
            return bool(operator.symbol)
        else:
            return operator.symbol   

    def calculateLaplaceAccuracy(self, individual, satisfiedAnt, satisfiedRules):        
        numClasses = 2.0	# binary classification
        return (satisfiedRules+1)/(satisfiedAnt+numClasses)
    
