import numpy as np

class ConformalPredictor():
  """
  Inductive conformal prediction based on "Tutorial On Conformal Prediction" by Shafer & Vovk (p. 388).
  Extended with Mondrian Conformal Prediction option based on "Mondrian Conformal Regressors" by Boström & Johansson, and https://gist.github.com/dsleo/2880882b5e1c1feab677c4cf421e806d
  """
  def __init__(self, alphas, y, mondrian_taxonomy=None):
    """
    alphas: non-conformity measures (n_samples).
    y: targets (n_samples)
    mondrian_taxonomy: the samples categories (n_samples)
    """
    super()

    self.alphas = alphas
    self.y = y
    self.mondrian_taxonomy = mondrian_taxonomy 

    return
  
  def predict(self, y_hat, confidence_level, mondrian=False, mondrian_category=None):
    """
    Retrieve a prediction region for y_hat
    confidence_level: e.g. 0.99
    """

    significance_level = 1 - confidence_level

    prediction_region = []

    classes = np.unique(self.y)

    for y in classes:
      # get non-conformity score for y_hat
      max_prob_not_y = np.max(np.delete(y_hat.detach().numpy(),y))
      ai = max_prob_not_y - y_hat[y].item()

      # non-conformity scores
      if mondrian:
        if mondrian_category is not None:
          if self.mondrian_taxonomy is None:
            raise ValueError("Expected self.mondrian_taxonomy, but found None")
          a = self.alphas[np.logical_and(self.mondrian_taxonomy == mondrian_category, self.y == y)]
        else:
          a = self.alphas[self.y == y]
      else:
        a = self.alphas
      
      # calculate p-score
      c = np.count_nonzero(a >= ai)
      p_score = c / len(a)

      if p_score > significance_level:
        prediction_region.append(y)
    # END: for
      
    return prediction_region
