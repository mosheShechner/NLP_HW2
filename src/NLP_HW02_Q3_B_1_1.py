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
# print("esp.train:: type %s; length %d" % (type(etr), len(etr)))
# print("esp.testa:: type %s; length %d" % (type(eta), len(eta)))
# print("esp.testb:: type %s; length %d" % (type(etb), len(etb)))
# print("ned.train:: type %s; length %d" % (type(dtr), len(dtr)))
# print("ned.testa:: type %s; length %d" % (type(dta), len(dta)))
# print("ned.testb:: type %s; length %d" % (type(dtb), len(dtb)))

# reaching an element
# print("reaching an element:")
# x = etr[0]
# print("esp.train:: data point (sentence): type %s; value %s" % (type(x), x))

sent = etr[0]
sent_vec = tree2conlltags(sent)
order = 2

# comparing formats
# print("tree format:\n%s" % sent)
# print("\n")
# print("list-of-tuples format:\n%s" % sent_vec)

# testing the features extraction - single word
print("->testing the features extraction - single word")

tokenIndex = 0
print("sample a token; index in sentence: %d; type %s; value %s" % (tokenIndex, type(sent_vec[tokenIndex]), sent_vec[tokenIndex]))
print(fe.word2features(sent_vec,tokenIndex,order))
print("")

tokenIndex = len(sent_vec)-1
print("sample a token; index in sentence: %d; type %s; value %s" % (tokenIndex, type(sent_vec[tokenIndex]), sent_vec[tokenIndex]))
print(fe.word2features(sent_vec,tokenIndex,order))
print("")

# testing encoding
print("->testing encoding")
x           = fe.getX(etr, order)
y           = fe.getY(etr)
tokenList   = fe.getTokens(etr)

print("x sample: %s" % x[0])
print("y sample: %s" % y[0])
print("token sample: %s" % tokenList[0])
