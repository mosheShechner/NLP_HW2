# ##########################################################################################
# module        NLP_HW02_Q1_feature_extraction
# ##########################################################################################
# path          ...NLP_HW02\NLP_HW02_Q1_feature_extraction.py
# Purpose       pre-process
# description
# ##########################################################################################
from nltk.corpus import conll2002
from nltk.chunk import tree2conlltags

etr = conll2002.chunked_sents('esp.train')  # In Spanish
eta = conll2002.chunked_sents('esp.testa')  # In Spanish
etb = conll2002.chunked_sents('esp.testb')  # In Spanish

dtr = conll2002.chunked_sents('ned.train')  # In Dutch
dta = conll2002.chunked_sents('ned.testa')  # In Dutch
dtb = conll2002.chunked_sents('ned.testb')  # In Dutch

#  data size
#print("esp.train:: type %s; length %d" % (type(etr), len(etr)))
#print("esp.testa:: type %s; length %d" % (type(eta), len(eta)))
#print("esp.testb:: type %s; length %d" % (type(etb), len(etb)))
#print("ned.train:: type %s; length %d" % (type(dtr), len(dtr)))
#print("ned.testa:: type %s; length %d" % (type(dta), len(dta)))
#print("ned.testb:: type %s; length %d" % (type(dtb), len(dtb)))

def bool2int(bool):
    if bool:           return 1
    else:              return 0

def setBoolVal(bool=None):
    if (bool==None):   return 0.0
    else:              return 2.0*(bool2int(bool)-0.5)

def getAdjWordFeatures(token=None, postag=None):
    # manually selected features for the adjacent word
    if (token == None):
        features = ["",
                    postag,
                    0,
                    0]
    else:
        features = [token.lower(),
                    postag,
                    setBoolVal(token.isdigit()),
                    setBoolVal(token.isupper())]
    return features

def getWordFeatures(token, postag):
    # manually selected features for the word
    features = [token.lower(),
                postag,
                setBoolVal(token.isdigit()),
                setBoolVal(token.isupper()),
                token[:1],
                token[:2],
                token[-1:],
                token[-2:]]
    return features

def word2features(sent, i, order):
    # sent      list of 3-tuples, each tuple is <token, POS, NER-tag>
    # i         index for tuple for feature extraction
    # order     number of context words on each side to be added to the features
    token  = sent[i][0]
    postag = sent[i][1]
    # set token features
    features = []
    features.extend(getWordFeatures(token,postag))

    # adding features for order-previous and order successive tokens
    for j in range(order):
        # previous words
        prv = (i - (j + 1))
        nxt = (i + (j + 1))

        if (prv >= 0):
            # token exist
            prvToken  = sent[prv][0]
            prvPostag = sent[prv][1]
            features.extend(getAdjWordFeatures(prvToken, prvPostag))
        else:
            # add pad
            features.extend(getAdjWordFeatures(token=None,postag="BOS"))
        if (nxt<len(sent)):
            # token exist
            nxtToken  = sent[nxt][0]
            nxtPostag = sent[nxt][1]
            features.extend(getAdjWordFeatures(nxtToken, nxtPostag))
        else:
            # add pad
            features.extend(getAdjWordFeatures(token=None,postag="EOS"))

    return features

# reaching a data point
x = etr.__getitem__
print("esp.train:: data point: type %s; value %s" % (type(x), x))

# comparing formats
sent = etr[0]
sent_vec = tree2conlltags(sent)

print("tree format:\n%s" % sent)
print("\n")
print("list-of-tuples format:\n%s" % sent_vec)

# testing the features extraction
order = 2

tokenIndex = 0
print("sample a token: index in sentence: %d; type %s; value %s" % (tokenIndex, type(sent_vec[tokenIndex]), sent_vec[tokenIndex]))
print(word2features(sent_vec,tokenIndex,order))

tokenIndex = len(sent_vec)-1
print("sample a token: index in sentence: %d; type %s; value %s" % (tokenIndex, type(sent_vec[tokenIndex]), sent_vec[tokenIndex]))
print(word2features(sent_vec,tokenIndex,order))


# counter = 1
# for sent in etr:
#     # sent is a sentence in a tree format
#     sent_vec = tree2conlltags(sent)
#     for sent_item in sent_vec:
#         print("data point entry: type %s; value %s" % (type(sent_item), sent_item))
#     counter = counter - 1
#     print(word2features(sent_vec,0,2))
#     if (counter == 0): break
