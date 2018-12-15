# ##########################################################################################
# module        NLP_HW02_Q3_feature_extraction
# ##########################################################################################
# path          ...\NLP_HW02\repository\src\NLP_HW02_Q3_feature_extraction.py
# Purpose       pre-process
# description
# ##########################################################################################
from nltk.chunk import tree2conlltags

def bool2int(bool):
    if bool:           return 1
    else:              return 0

def setBoolVal(bool=None):
    if (bool==None):   return 0.5
    else:              return 1.0*bool2int(bool)

def getPadWordFeatures(token=None, postag=None, pref=""):
    features = {pref + 'token':         token,
                pref + 'postag':        postag,
                pref + 'isdigit':       setBoolVal(None),
                pref + 'isupper':       setBoolVal(None)}
    return features

def getAdjWordFeatures(token=None, postag=None, pref=""):
    # manually selected features for the adjacent word
    features = {pref+'token.lower': token.lower(),
                pref+'postag':      postag,
                pref+'isdigit':     setBoolVal(token.isdigit()),
                pref+'isupper':     setBoolVal(token.isupper())}
    return features

def getWordFeatures(token, postag):
    # manually selected features for the word
    features = {'token.lower':  token.lower(),
                'postag':       postag,
                'isdigit':      setBoolVal(token.isdigit()),
                'isupper':      setBoolVal(token.isupper()),
                'perf1':        token[:1],
                'pref2':        token[:2],
                'suff1':        token[-1:],
                'suff2':        token[-2:]}
    return features

def word2features(sent, i, order):
    # sent      list of 3-tuples, each tuple is <token, POS, NER-tag>
    # i         index for tuple for feature extraction
    # order     number of context words on each side to be added to the features
    token  = sent[i][0]
    postag = sent[i][1]
    # set token features
    features = {}
    features.update(getWordFeatures(token,postag))

    # adding features for order-previous and order successive tokens
    for j in range(order):
        # previous words
        prv = (i - (j + 1))
        nxt = (i + (j + 1))
        prvStr = str(prv)
        nxtStr = str(nxt)

        if (prv >= 0):
            # token exist
            prvToken  = sent[prv][0]
            prvPostag = sent[prv][1]
            features.update(getAdjWordFeatures(prvToken, prvPostag,prvStr))
        else:
            # add pad
            features.update(getPadWordFeatures(token="PAD",postag="BOS", pref=prvStr))
        if (nxt<len(sent)):
            # token exist
            nxtToken  = sent[nxt][0]
            nxtPostag = sent[nxt][1]
            features.update(getAdjWordFeatures(nxtToken, nxtPostag, nxtStr))
        else:
            # add pad
            features.update(getPadWordFeatures(token="PAD",postag="EOS", pref=nxtStr))

    return features

# encoding data set
def sent2features(sent, order):
    return [word2features(sent, i, order) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]# reaching a data point

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def getX(sentenceDataSet,order):
    xOut = []
    for sent in sentenceDataSet:
        sent_vec = tree2conlltags(sent)
        xOut.extend(sent2features(sent_vec,order))
    return xOut

def getY(sentenceDataSet):
    yOut = []
    for sent in sentenceDataSet:
        sent_vec = tree2conlltags(sent)
        yOut.extend(sent2labels(sent_vec))
    return yOut

def getTokens(sentenceDataSet):
    tokenOut =[]
    for sent in sentenceDataSet:
        sent_vec = tree2conlltags(sent)
        tokenOut.extend(sent2tokens(sent_vec))
    return tokenOut

