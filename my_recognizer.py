import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = []
    guesses = []

    test_words_with_lengths = test_set.get_all_Xlengths().values()

    for test_word, lengths in test_words_with_lengths:

            best_score = float("-inf")
            best_guess = None
            best_log_likelihood = {}

            # for given test word iterate through all the trained words
            # and see which one's model fits best to this test word
            for trained_word, hmm_model in models.items():
                try:
                    score = hmm_model.score(test_word, lengths)
                    best_log_likelihood[trained_word] = best_score
                except:
                    score = float("-inf")
                    best_log_likelihood[trained_word] = score

                if score > best_score:
                    best_score = score
                    best_guess = trained_word

            # for this test word store the best guess and proceed to the next one
            guesses.append(best_guess)
            # store also the log-likelihoods all our guesses (including the best one)
            probabilities.append(best_log_likelihood)

    return probabilities, guesses
