# ##########################################################################################
# module        NLP_HW02_Q3_B_1_1
# ##########################################################################################
# path          ...\NLP_HW02\repository\src\NLP_HW02_Q3_B_1_1.py
# Purpose       Executions of bullet 3.1.1
# description
# ##########################################################################################
from nltk.corpus import conll2002
from nltk.chunk import tree2conlltags
import NLP_HW02_Q3_feature_extraction as fe

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

# reaching an element
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
print(fe.word2features(sent_vec,tokenIndex,order))

tokenIndex = len(sent_vec)-1
print("sample a token: index in sentence: %d; type %s; value %s" % (tokenIndex, type(sent_vec[tokenIndex]), sent_vec[tokenIndex]))
print(fe.word2features(sent_vec,tokenIndex,order))

# testing encoding
x           = fe.getX(etr, order)
y           = fe.getY(etr)
tokenList   = fe.getTokens(etr)

print("checking encoding")
print("x sample: %s" % x[0])
print("y sample: %s" % y[0])
print("token sample: %s" % tokenList[0])

# counter = 1
# for sent in etr:
#     # sent is a sentence in a tree format
#     sent_vec = tree2conlltags(sent)
#     for sent_item in sent_vec:
#         print("data point entry: type %s; value %s" % (type(sent_item), sent_item))
#     counter = counter - 1
#     print(fe.word2features(sent_vec,0,2))
#     if (counter == 0): break