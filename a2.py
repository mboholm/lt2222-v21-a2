import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random

#For lemmatization:
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
wnl = WordNetLemmatizer() 
from nltk.corpus import wordnet as wn

def lemmatizer(wordform, pos_tag):
    """
    Takes a wordform and a part-of-speech (POS) tag and returns a lemma. 

    Note: the words of the data (seemigly) is tagged using Penn Treebank convention for POS. Since WordNetLemmatizer uses Word Net's tags the tag for adjectives ("j" in Penn) is converted to Word Net's convention ("a" for adjective). For adverbs, nous and verbs tags are the same in both notations.
    """

    if pos_tag[0] == "j": #i.e. the Penn tag for adjectives
        pos_tag=wn.ADJ #i.e. "a"

    lemma = wnl.lemmatize(wordform, pos_tag[0])
    return lemma

#For removing punctuations:
import string as s

# Function for Part 1
def preprocess(inputfile):
    """ 
    Takes a file object (a list of words from a corpus) and returns a list of dictionaries that represent features of each word. 
    """

    data=[]

    for line in inputfile.readlines()[1:]: #we do not need the headings
        w=line.strip("\n").split("\t")
        form=w[2].lower()
        if form not in s.punctuation: #skip punctuations
            d={}
            d["id"]=w[0]
            d["sent"]=w[1]            
            pos=w[3].lower()
            d["pos"]= pos
            if pos[0] in ["a", "r", "n", "v"]: #only lemmatize adjectives, adverbs, nouns and verbs
                d["lemma"]=lemmatizer(form, pos)
            else:
                d["lemma"]=form
            d["ne_tag"]=w[4]
            data.append(d)

    return data

# Code for part 2
class Instance:
    """ 
    Represents a named entitiy (NE). 

    Attributes: the class of the NE and the features of the NE (i.e. words in the context of NE)
    """

    def __init__(self, neclass, features):
        self.neclass = neclass
        self.features = features

    def __str__(self):
        return "Class: {} Features: {}".format(self.neclass, self.features)

    def __repr__(self):
        return str(self)

def create_instances(data, pos=False):
    """ 
    Takes a list of dictinaries that represent (features of) words (instances) and returns the class and features of those instances which are "named entities" (NE). 

    Two types of features are possible to represent with create_instances: 
    1. words within the same sentence as NE, being within five words to the lft of NE and five words to the right. This is the default feature representation of NEs.
    2. words and POS-tags of those words, within the same context as in 1. In order to implement create_instances with this setting set argument pos to True.
    """

    N=len(data) #the size of the data

    instances = []

    i=0
    while i < N:
        word_d=data[i]
        if word_d["ne_tag"][0] == "B":
            context=[]

            # identifying features in the left context
            j=0
            for p in range(5):
                if i-j <= 0:
                    break
                elif data[i-j]["sent"]!=word_d["sent"]:
                    break
                else:
                    j+=1

            for previous in data[(i-j):(i-1)]:
                context.append(previous["lemma"])
                if pos==True:
                    context.append(previous["pos"])

            # skipping through multi-word NEs
            skip=1
            if i==N-1: #i.e. if we are on the last one
                skip=0

            while data[(i+skip)]["ne_tag"][0]=="I":
                i+=1

            # identifying features in the righthand context
            j=0
            for p in range(5):
                if i+j >= N:
                    break
                elif data[i+j]["sent"]!=word_d["sent"]:
                    break
                else:
                    j+=1
            
            for next_up in data[i:(i+j)]:
                context.append(next_up["lemma"])
                if pos==True:
                    context.append(next_up["pos"]) 

            instance=Instance(neclass=word_d["ne_tag"][-3:], features=context)
            instances.append(instance)

        i+=1

    return instances

# Code for part 3

def word_finder(instances):
    """
    Takes a list of NE instances and returns a key to each feature of the instance.
    """

    vocy=[] #the vocabulary (features)

    for ins in instances:
        for f in ins.features:
            if f not in vocy:
                vocy.append(f)

    wf={term:key for term,key in zip(vocy, range(len(vocy)))} #making a feature-key association

    return wf

def create_table(instances):
    """ 
    Takes a list of instances of the class Instance and returns a Pandas DataFrame with class and features. 
    """

    l_ins=len(instances)
    i_matrix=[]
    vocy=word_finder(instances)

    for ins in instances:
        row={}
        row["class"]=ins.neclass

        for f in ins.features:
            key=vocy[f]
            if key in row:
                row[key]+=1
            else:
                row[key]=1

        i_matrix.append(row)
    
    df=pd.DataFrame(i_matrix)
    first_column = df.pop('class') #in order to put 'class' in first column
    df.insert(0, 'class', first_column)

    df=df.fillna(0) #https://datatofish.com/replace-nan-values-with-zeros/

    return df

def ttsplit(bigdf, test_proportion=0.2, backtrack=False):
    """
    Takes a matrix (Pandas DataFrame) of class and features and plit it into:
    1. the training features (train_x)
    2. the training classes (train_y)
    3. the test features (test_x)
    4. the test classes (test_y)

    Assumes that proportion of training data is the complement of the test proportion.

    Reference: this function is designed based on code in Demo 2 - Learning with support vector machines.ipynb for the course LT2222 (Univ. Gothenburg) (https://canvas.gu.se/courses/41213/files/folder/demos?preview=4268097)

    A note on backtracking instances:
    In the Jupyter Notebook there is a call for test_y[0]. If test_y is returned as a pandas dataframe, this call will result in a keyword error. Therefore, train_x etc. are instead returned as numpy arrays. However, for bonus part A, we need to backtrack instances in order to analyse why we get false predictions. Keeping the dataframe format is useful for this.

    """

    negroups = bigdf.groupby("class", group_keys=False)
    testdf = negroups.apply(lambda x: x.sample(frac=test_proportion)) #this approach will keep proportions of NE classes from the dataset "intact" (note the contrasting approach of taking one random sample from the data, which would be likely to represent the proportions of the NE classes from the sample, but where it would be possible that small NE classes would not be sampled at all.)
    traindf = bigdf.drop(testdf.index, axis=0) #i.e. the complete data set minus the test set

    if backtrack==True:
    	train_x = traindf.drop('class', axis=1)
    	train_y = traindf['class']
    	test_x = testdf.drop('class', axis=1)
    	test_y = testdf['class']
    else:
    	train_x = traindf.drop('class', axis=1).to_numpy()
    	train_y = traindf['class'].to_numpy()
    	test_x = testdf.drop('class', axis=1).to_numpy()
    	test_y = testdf['class'].to_numpy()    

    return train_x, train_y, test_x, test_y

# Code for part 5
def correct(matrix):
    """
    Summarises accuracy (true positives [TP], total number of predictions, and proportion of TPs of total number of predictions)
    """

    d={x:{"correct":0, "total":0, "accuracy":0} for x in matrix.index}

    for x in matrix.index:
        corr=matrix.loc[x][x]
        tot=matrix.loc[x].sum(axis=0)
        d[x]["correct"]=corr
        d[x]["total"]=tot
        hr=0
        if corr != 0:
            hr=round((corr/tot), 2)
        d[x]["accuracy"]=hr

    return pd.DataFrame(d)

def confusion_matrix(truth, predictions, backtrack=False): 
    """
    Takes an array of true classes and compares with predicted classes, and returns a confusion matrix.
    """

    if backtrack==True:
    	truth=list(truth)
    	predictions=list(predictions)

    #Identifying all NE classes:
    nes=[]
    for ne_class in truth:
        if ne_class not in nes:
            nes.append(ne_class)

    l=len(truth) #... so that we can use the lentgh of one of them

    associations={x:{x: 0 for x in nes} for x in nes}

    for c in range(l):
        associations[truth[c]][predictions[c]]+=1 #adding 1 to the count of the true NE class's association with the predicted NE class

    confusion_matrix=pd.DataFrame(associations)

    ##  Here I have added a few lines that helps 
    ##  interpretation of predictions and 
    ##  errors. 

    summary=correct(confusion_matrix)

    print("="*45)
    print("TABLE 1. Confusion matrix, where true labels \nrepresented vertically and \npredictions horisontally.")
    print("-"*45)
    print(confusion_matrix)
    print("="*45,"\n")

    print("="*67)
    print("TABLE 2. Summary: true positives (correct), total number of \npredictions and proportion TP of total (accuracy).")
    print("-"*67)
    print(summary)
    print("="*67)    

    return confusion_matrix #the return is not really needed for the assignment

# Code for Bonus Part A

##  This function (finding_neverland) is never called within the jupyter notebook
##  submitted for assignment 2. It is not asked for. However, I have used it to 
##  find "examples in the test data on which the classifier classified incorrectly 
##  for those classes", as is asked for in Bońus Part A. I have kept the function 
##  definition here. If considered irrelevant, please ignore it. (The function
##  finding_neverland requires that ttsplit returns pandas DataFrames, not numpy 
##  arrays; i.e. backtrack=True)

def finding_neverland(truth, predictions, ne_class, instances, n=20):
    """
    Identifies instances in which ne_class is identified as something else.
    """

    comparison = truth==predictions
    divergence = comparison.loc[comparison[:]==False]
    neverland = truth.drop(divergence.index, axis=0)
    indices = neverland.loc[neverland[:]==ne_class].index

    hard_instances=[]
    for i in indices:
    	hard_instances.append(instances[i])

    print(*random.sample(hard_instances, n), sep="\n")


# Code for bonus part B
def bonusb(filename):
    """
    Takes a fileame and returns some error calculations (confusion matrix).
    """

    print("Open and preprocessing data ...")
    gmbfile = open(filename, "r")
    inputdata = preprocess(gmbfile)
    gmbfile.close()
    
    print("Extracting instances ...")
    instances = create_instances(inputdata, pos=True) #Note change of parameter pos from part 2
    bigdf = create_table(instances)
    
    print("Split into train and test samples ...")
    train_X, train_y, test_X, test_y = ttsplit(bigdf)
    
    print("Training a classifier ...")
    from sklearn.svm import LinearSVC
    model = LinearSVC(max_iter=1400) # I have set the max_iter rather "untheoretically". First, the default (max_iter=1000; see https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC) throws a ConvergenceWarning: Liblinear failed to converge, increase the number of iterations. To solve this, I increased max_iter with 100 (1100, etc.) and tested the code until it worked.
    model.fit(train_X, train_y)
    
    print("Predicting classes ...")
    train_predictions = model.predict(train_X)
    test_predictions = model.predict(test_X)
    
    print("Confusion matrices:")
    print("For test sample:")
    confusion_matrix(test_y, test_predictions)
    print("\nFor predictions on training sample:")
    confusion_matrix(train_y, train_predictions)
