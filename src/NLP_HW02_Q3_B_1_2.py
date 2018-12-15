# ##########################################################################################
# module        NLP_HW02_Q3_B_1_2
# ##########################################################################################
# path          ...\NLP_HW02\repository\src\NLP_HW02_Q3_B_1_2.py
# Purpose       Executions of bullet 3.1.1
# description
# ##########################################################################################
from nltk.corpus import conll2002
import pycrfsuite
import NLP_HW02_Q3_feature_extraction as fe

etr = conll2002.chunked_sents('esp.train')  # In Spanish
eta = conll2002.chunked_sents('esp.testa')  # In Spanish
etb = conll2002.chunked_sents('esp.testb')  # In Spanish

dtr = conll2002.chunked_sents('ned.train')  # In Dutch
dta = conll2002.chunked_sents('ned.testa')  # In Dutch
dtb = conll2002.chunked_sents('ned.testb')  # In Dutch

order = 2
print("-> encoding training data")
X_train     = fe.getX(etr, order)
Y_train     = fe.getY(etr)
tokenList   = fe.getTokens(etr)

print("-> testing encoding")
print("data size:  |X_train| = %d; type = %s" % (len(X_train), type(X_train)))
print("label size: |Y_train| = %d" % len(Y_train))
print("x sample: %s" % X_train[0])
print("y sample: %s" % Y_train[0])
print("token sample: %s" % tokenList[0])

print("-> setting pycrfsuite trainer")
trainer = pycrfsuite.Trainer(verbose=False)
print(trainer.params())

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier
    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

print("-> setting data to trainer")
trainer.append(X_train,Y_train)
trainer.train('conll2002-esp.crfsuite')

print("-> encoding test data")
X_test      = fe.getX(eta, order)
Y_test      = fe.getY(eta)

print("-> setting tagger")
tagger = pycrfsuite.Tagger()
tagger.open('conll2002-esp.crfsuite')


example_sent = eta[0]
print(' '.join(fe.getSentTokens(example_sent)), end='\n\n')

print("Predicted:", ' '.join(tagger.tag(fe.getSentX(example_sent, order))))
print("Correct:  ", ' '.join(fe.getSentY(example_sent)))

example_sent = eta[2]
print(' '.join(fe.getSentTokens(example_sent)), end='\n\n')

print("Predicted:", ' '.join(tagger.tag(fe.getSentX(example_sent, order))))
print("Correct:  ", ' '.join(fe.getSentY(example_sent)))
