import torch

class ConformalClassifier():
  """
  Inductive conformal prediction based on "Tutorial On Conformal Prediction" by Shafer & Vovk (p. 388).
  Extended with Mondrian Conformal Prediction option based on "Mondrian Conformal Regressors" by Boström & Johansson, and https://gist.github.com/dsleo/2880882b5e1c1feab677c4cf421e806d
  """
  def __init__(self, alphas, y, mondrian=False, mondrian_taxonomy=None, cumulative_taxonomy=False):
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
    self.cumulative_taxonomy = cumulative_taxonomy

    return
  
  def predict(self, alphas, confidence_level, mondrian_category=None):
    """
    Retrieve a prediction region for the provided nonconformity measures 
    \nOBS! Only single samples allowed
    \nconfidence_level: e.g. 0.99
    """

    significance_level = 1 - confidence_level

    prediction_region = []

    classes = torch.unique(self.y)

    for y in classes:
      ai = alphas[y]

      # non-conformity scores
      if self.mondrian:
        if mondrian_category is not None:
          if self.mondrian_taxonomy is None:
            raise ValueError("Expected self.mondrian_taxonomy, but found None")
          if self.cumulative_taxonomy:
            a = self.alphas[torch.logical_and(self.mondrian_taxonomy <= mondrian_category, self.y == y)]
          else:
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

def get_nonconformity_measure_for_classification(y_proba: torch.Tensor, version="v1"):
  """Returns nonconformity measure for all possible classes"""
  alphas = []
  for i in range(y_proba.shape[0]):
    if version == "v1": # non-conformity measure according to "Reliable diagnosis of acute abdominal pain with conformal prediction" (cp_in_medicine)
      y_probas_for_false_classes = torch.cat([y_proba[0:i], y_proba[i+1:]])
      max_prob_not_y = torch.max(y_probas_for_false_classes)
      a = max_prob_not_y - y_proba[i].item()
    elif version == "v2":
      a = 1 - y_proba[i].item()
    else:
      raise ValueError("Invalid version")
    
    alphas.append(a)

  return torch.tensor(alphas)