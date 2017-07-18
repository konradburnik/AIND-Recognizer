import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

class ScoredModel(tuple):
    '''
    A tuple where only the first component is compared with '<' and '>'.
    '''
    def __lt__(self, other):
        return self[0] < other[0]

    def __gt__(self, other):
        return self[0] > other[0]

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def bic_score(self, num_states):
        """
            Calculate the BIC score.
        """
        # ftp://metron.sta.uniroma1.it/RePEc/articoli/2002-LX-3_4-11.pdf
        hmm_model = self.base_model(num_states)

        N = len(self.X)
        logN = np.log(N)
        logL =  hmm_model.score(self.X, self.lengths)
        f = hmm_model.n_features
        p = num_states ** 2 * 2 * num_states * f - 1
        return -2 * logL + p * logN, hmm_model

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            best_score, best_model = min([ScoredModel(self.bic_score(num_states))
                                            for num_states in range(self.min_n_components, self.max_n_components + 1)])
            return best_model

        except Exception as e:
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def dic_score(self, num_states):
        # https://discussions.udacity.com/t/dic-score-calculation/238907
        hmm_model = self.base_model(num_states)

        LogL = hmm_model.score(self.X, self.lengths)
        collected_scores = [hmm_model.score(X, lengths)
                                for word, (X, lengths) in self.hwords.items()
                                    if word != self.this_word]
        mu = np.mean(collected_scores)
        return  LogL - mu, hmm_model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            best_score, best_model = max([ScoredModel(self.dic_score(num_states))
                                            for num_states in range(self.min_n_components, self.max_n_components + 1)])
            return best_model

        except Exception as e:
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def cv_score(self, num_states):
        """
        Calculate the average log likelihood of cross-validation folds using the KFold class
        :return: tuple of the mean likelihood and the model with the respective score
        """
        collected_scores = []

        # Handle the number of folds
        num_seq = len(self.sequences)
        if num_seq == 0:
            raise ValueError("Value of num_seq can not be equal to zero!")
        if num_seq == 1:
            return float("-inf"), self.base_model(num_states)
        else:
            num_splits = math.ceil(math.log(num_seq)) + 1

        split_method = KFold(n_splits=num_splits)

        for train_idx, test_idx in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(train_idx, self.sequences)

            hmm_model = self.base_model(num_states)

            X, lengths = combine_sequences(test_idx, self.sequences)

            collected_scores.append(model.score(X, lengths))
        return np.mean(collected_scores), hmm_model

    def select(self):
        """ select the best model for self.this_word based on
        CV score for n between self.min_n_components and self.max_n_components
        It is based on log likehood
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            best_score, best_model = max([ScoredModel(self.cv_score(num_states))
                                            for num_states in range(self.min_n_components, self.max_n_components + 1)])
            return best_model

        except Exception as e:
            return self.base_model(self.n_constant)
