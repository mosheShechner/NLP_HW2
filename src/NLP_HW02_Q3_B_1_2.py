# ##########################################################################################
# module        NLP_HW02_Q3_B_1_2
# ##########################################################################################
# path          ...\NLP_HW02\repository\src\NLP_HW02_Q3_B_1_2.py
# Purpose       Executions of bullet 3.1.2
# description
# ##########################################################################################
import pycrfsuite
import scipy
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nltk.corpus import conll2002

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

# print("-> testing encoding")
# print("data size:  |X_train| = %d; type = %s" % (len(X_train), type(X_train)))
# print("label size: |Y_train| = %d; type = %s" % (len(Y_train), type(X_train)))
# print("x sample: %s" % X_train[0])
# print("y sample: %s" % Y_train[0])
# print("token sample: %s" % tokenList[0])

print("-> setting pycrfsuite trainer")
trainer = pycrfsuite.Trainer(verbose=False)

print("-> setting trainer parameters")
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
token_test  = fe.getTokens(eta)

class_names = set(Y_test)

print("-> setting tagger")
tagger = pycrfsuite.Tagger()
tagger.open('conll2002-esp.crfsuite')

example_sent = eta[0]
print("")
print(' '.join(fe.getSentTokens(example_sent)), end='\n\n')
print("Predicted:", ' '.join(tagger.tag(fe.getSentX(example_sent, order))))
print("Correct:  ", ' '.join(fe.getSentY(example_sent)))

# example_sent = eta[2]
# print("")
# print(' '.join(fe.getSentTokens(example_sent)), end='\n\n')
# print("Predicted:", ' '.join(tagger.tag(fe.getSentX(example_sent, order))))
# print("Correct:  ", ' '.join(fe.getSentY(example_sent)))

print("-> predicting test data'eta'")
Y_pred      = tagger.tag(X_test)

print("real tag list # %d;" % len(Y_test))
print("pred tag list # %d;" % len(Y_pred))
test_class_names = set(Y_test)
pred_class_names = set(Y_pred)
print(test_class_names)
print(pred_class_names)


print("printing few results")

print("token    : %s" % token_test[0])
print("x        : %s" % X_test[0])
print("real     : %s" % Y_test[0])
print("predicted: %s" % Y_pred[0])

print("token    : %s" % token_test[1])
print("x        : %s" % X_test[1])
print("real     : %s" % Y_test[1])
print("predicted: %s" % Y_pred[1])


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(y_true)
    y_pred_combined = lb.fit_transform(y_pred)

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )


def measureModel(train, test, order, experimentName="defExperimentName", verbose=False):
    # encode training data
    if verbose: print("-> encoding train data")
    X_train = fe.getX(train, order)
    Y_train = fe.getY(train)

    # train a classifier
    if verbose: print("-> setting trainer parameters")
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    if verbose: print("-> setting data to trainer")
    trainer.append(X_train, Y_train)
    if verbose: print("-> training; classifier saved on ./%s" % experimentName)
    trainer.train(experimentName)

    # encode test data
    if verbose: print("-> encoding test data")
    X_test = fe.getX(test, order)
    Y_test = fe.getY(test)

    # tag test data (prediction)
    if verbose: print("-> tagging test data with trained classifier %s" % experimentName)
    tagger = pycrfsuite.Tagger()
    tagger.open(experimentName)
    Y_pred = tagger.tag(X_test)

    # print peasured results:
    print("\n\nModel performance classification results for experiment: %s" % experimentName)
    print(bio_classification_report(Y_test, Y_pred))


# measure models order={0,1,2} for {Spanish, Dutch}
measureModel(etr, eta, 0, experimentName="Lang=Spanish_X_Order=0", verbose=False)
measureModel(etr, eta, 1, experimentName="Lang=Spanish_X_Order=1", verbose=False)
measureModel(etr, eta, 2, experimentName="Lang=Spanish_X_Order=2", verbose=False)

measureModel(dtr, dta, 0, experimentName="Lang=Dutch_X_Order=0", verbose=False)
measureModel(dtr, dta, 1, experimentName="Lang=Dutch_X_Order=1", verbose=False)
measureModel(dtr, dta, 2, experimentName="Lang=Dutch_X_Order=2", verbose=False)