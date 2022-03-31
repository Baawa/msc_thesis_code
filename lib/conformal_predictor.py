import torch

class ConformalClassifier():
  """
  Inductive conformal prediction based on "Tutorial On Conformal Prediction" by Shafer & Vovk (p. 388).
  Extended with Mondrian Conformal Prediction option based on "Mondrian Conformal Regressors" by Boström & Johansson, and https://gist.github.com/dsleo/2880882b5e1c1feab677c4cf421e806d
  """
  def __init__(self, alphas, y, mondrian=False, mondrian_taxonomy=None):
    """
    alphas: non-conformity measures (n_samples).
    y: targets (n_samples)
    mondrian_taxonomy: the samples categories (n_samples)
    """
    super()

    self.alphas = alphas
    self.y = y
    self.mondrian = mondrian
    self.mondrian_taxonomy = mondrian_taxonomy 

    return
  
  def predict(self, y_hat, confidence_level, mondrian_category=None):
    """
    Retrieve a prediction region for y_hat
    confidence_level: e.g. 0.99
    """

    significance_level = 1 - confidence_level

    prediction_region = []

    classes = torch.unique(self.y)

    for y in classes:
      ai = get_nonconformity_measure_for_classification(y_hat.detach(), y)

      # non-conformity scores
      if self.mondrian:
        if mondrian_category is not None:
          if self.mondrian_taxonomy is None:
            raise ValueError("Expected self.mondrian_taxonomy, but found None")
          a = self.alphas[torch.logical_and(self.mondrian_taxonomy == mondrian_category, self.y == y)]
        else:
          a = self.alphas[self.y == y]
      else:
        a = self.alphas
      
      # calculate p-score
      c = torch.count_nonzero(a >= ai)
      p_score = c / len(a)

      if p_score > significance_level:
        prediction_region.append(y)
    # END: for
      
    return prediction_region

def get_nonconformity_measure_for_classification(y_proba: torch.Tensor, y_true: int):
  """non-conformity measure according to "Reliable diagnosis of acute abdominal pain with conformal prediction" (cp_in_medicine)"""
  y_probas_for_false_classes = torch.cat([y_proba[0:y_true], y_proba[y_true+1:]])
  max_prob_not_y = torch.max(y_probas_for_false_classes)
  alpha = max_prob_not_y - y_proba[y_true].item()

  return alpha