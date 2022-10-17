import numpy as np
from random import seed
import matplotlib
import data
import private
from tqdm import trange

from pathlib import Path

import typer


app = typer.Typer()

def main(
    n_nodes:int = 4000,
    n_iter:int = 10,
    n_trials:int = 10,
    seed:int = 1,
    L: float = .4,
    prefix: str = "huge"
):
    ### Load dataset

    np.random.seed(seed)

    X_train, X_test, y_train, y_test = data.load("Houses")
    print("Successfully load dataset")
    print(X_train.shape)



    def score(y):
        # defining score to be able to evaluate the model on the test set during the training
        def evaluation(theta):

            from sklearn.linear_model._base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator
            class OurEval(BaseEstimator, LinearClassifierMixin):
                def __init__(self):
                    self.intercept_ = np.expand_dims(theta[-1], axis=0)
                    self.coef_ = np.expand_dims(theta[:-1], axis=0)
                    self.classes_ = np.unique(y)

                def fit(self, X, y):
                    pass

            myeval = OurEval()

            return myeval.score(X_test, y_test) 

        return evaluation

    # names for saving the results
    name_central = "result/"+prefix+"central"
    name_muffliato = "result/"+prefix+"muffliato"
    name_privacy = "result/"+prefix+"privacy"

    freq_eval = 1
    gamma = .7
    sigma = 1.5

    scores_central = np.zeros((n_trials, n_iter))
    scores_muffliato = np.zeros((n_trials, n_iter))
    obj_central = np.zeros((n_trials, n_iter))
    obj_muffliato = np.zeros((n_trials, n_iter))

    privacy_muffliato = np.zeros((n_trials, n_nodes))

    for i in trange(n_trials):
        central = private.CentralLogisticRegression(gamma, n_iter, n_nodes, sigma,  score=score, freq_obj_eval=freq_eval, L=L)
        central.fit(X_train, y_train)
        scores_central[i] = central.scores_
        obj_central[i] = central.obj_list_

        muffliato = private.MuffliatoLogisticRegression(gamma, n_iter, n_nodes, sigma, score=score, freq_obj_eval=freq_eval, L=L)
        muffliato.fit(X_train, y_train)
        scores_muffliato[i] = muffliato.scores_
        obj_muffliato[i] = muffliato.obj_list_
        privacy_muffliato[i] = muffliato.privacy_loss


    # Saving data
    np.save(name_muffliato, scores_muffliato)
    np.save(name_central, scores_central)
    np.save(name_privacy, privacy_muffliato)

if __name__ == "__main__":
    typer.run(main)