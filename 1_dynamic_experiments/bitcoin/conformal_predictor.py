import torch


def get_nonconformity_measure_for_classification(y_proba: torch.Tensor, version="v2"):
    """Returns nonconformity measure for all possible classes"""
    if version == "v1":  # non-conformity measure according to "Reliable diagnosis of acute abdominal pain with conformal prediction" (cp_in_medicine)
        alphas = []
        for i in range(y_proba.shape[0]):
            y_probas_for_false_classes = torch.cat(
                [y_proba[0:i], y_proba[i+1:]])
            max_prob_not_y = torch.max(y_probas_for_false_classes)
            a = max_prob_not_y - y_proba[i].item()
            alphas.append(a)
        alphas = torch.tensor(alphas)
    elif version == "v2":
        alphas = 1 - y_proba
    else:
        raise ValueError("Invalid version")

    return alphas


class InductiveConformalClassifier():
    """
    Inductive conformal prediction based on "Tutorial On Conformal Prediction" by Shafer & Vovk (p. 388).
    """

    def __init__(self, alphas, num_classes):
        """
        alphas: non-conformity measures (n_samples).
        """
        super()

        self.alphas = alphas
        self.num_classes = num_classes

        return

    def predict(self, alphas, confidence_level):
        """
        Retrieve a prediction region for the provided nonconformity measures 
        \nOBS! Only single samples allowed
        \nconfidence_level: e.g. 0.99
        """

        significance_level = 1 - confidence_level

        prediction_region = []

        for y in range(self.num_classes):
            ai = alphas[y]

            # non-conformity scores
            a = self.alphas

            # calculate p-score
            c = torch.count_nonzero(a >= ai)
            p_score = c / len(a)

            if p_score > significance_level:
                prediction_region.append(y)
        # END: for

        return prediction_region

class MondrianConformalClassifier():
    """
    Extended with Mondrian Conformal Prediction option based on "Mondrian Conformal Regressors" by Boström & Johansson, and https://gist.github.com/dsleo/2880882b5e1c1feab677c4cf421e806d
    """

    def __init__(self, alphas, y):
        """
        alphas: non-conformity measures (n_samples).
        y: targets (n_samples)
        """
        super()

        self.alphas = alphas
        self.y = y

        return

    def predict(self, alphas, confidence_level):
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
            a = self.alphas[self.y == y]

            # calculate p-score
            c = torch.count_nonzero(a >= ai)
            p_score = c / len(a)

            if p_score > significance_level:
                prediction_region.append(y)
        # END: for

        return prediction_region

class NodeDegreeMondrianConformalClassifier():
    """
    Novel MCP using the node degree instead of the class.
    """

    def __init__(self, alphas, y, node_degrees):
        """
        alphas: non-conformity measures (n_samples).
        y: targets (n_samples)
        node_degrees: the samples node degrees (n_samples)
        """
        super()

        self.alphas = alphas
        self.y = y
        self.node_degrees = node_degrees

        return

    def predict(self, alphas, confidence_level, node_degree):
        """
        Retrieve a prediction region for the provided nonconformity measures 
        \nOBS! Only single samples allowed
        \nconfidence_level: e.g. 0.99
        """

        significance_level = 1 - confidence_level

        prediction_region = []

        classes = torch.unique(self.y)

        _alphas = self.alphas[self.node_degrees == node_degree]

        for y in classes:
            ai = alphas[y]

            # non-conformity scores
            a = _alphas

            # calculate p-score
            c = torch.count_nonzero(a >= ai)
            p_score = c / len(a)

            if p_score > significance_level:
                prediction_region.append(y)
        # END: for

        return prediction_region

class NodeDegreeWeightedConformalClassifier():
  """
  Inspired by "Conformal prediction beyond exchangeability" (2022)
  """
  def __init__(self, alphas, y, node_degrees):
    """
    alphas: non-conformity measures (n_samples).
    y: targets (n_samples)
    node_degrees: the samples node_degrees (n_samples)
    """
    super()

    self.alphas = alphas
    self.y = y
    self.node_degrees = node_degrees

    return
  
  def predict(self, alphas, node_degree, confidence_level):
    """
    Retrieve a prediction region for the provided nonconformity measures 
    \nOBS! Only single samples allowed
    \nconfidence_level: e.g. 0.99
    """

    significance_level = 1 - confidence_level

    prediction_region = []

    classes = torch.unique(self.y)
    
    # weights
    max_degree = torch.max(torch.cat((self.node_degrees, torch.tensor([node_degree]))))
    normalized_degrees = self.node_degrees / max_degree
    weights = 1 - torch.abs((node_degree/max_degree) - normalized_degrees)
    sum_weights = torch.sum(weights) + 1
    
    cal_normalized_weights = weights / sum_weights
    sample_normalized_weight = 1 / sum_weights

    for y in classes:
      ai = alphas[y] * sample_normalized_weight

      # non-conformity scores
      a = self.alphas * cal_normalized_weights
      
      # calculate p-score
      c = torch.count_nonzero(a >= ai)
      p_score = c / len(a)

      if p_score > significance_level:
        prediction_region.append(y)
    # END: for
      
    return prediction_region

class EmbeddingDistanceWeightedConformalClassifier():
  """
  Inspired by "Conformal prediction beyond exchangeability" (2022)
  """
  def __init__(self, alphas, y, node_embeddings):
    """
    alphas: non-conformity measures (n_samples).
    y: targets (n_samples)
    node_embeddings: the samples node_embeddings (n_samples, embedding_size)
    """
    super()

    self.alphas = alphas
    self.y = y
    self.node_embeddings = node_embeddings

    return
  
  def predict(self, alphas, node_embedding, confidence_level):
    """
    Retrieve a prediction region for the provided nonconformity measures 
    \nOBS! Only single samples allowed
    \nconfidence_level: e.g. 0.99
    """

    significance_level = 1 - confidence_level

    prediction_region = []

    classes = torch.unique(self.y)
    
    # euclidean ditance without sqrt for faster calculations
    embedding_distance = torch.sum((self.node_embeddings-node_embedding)**2, dim=1)

    max_distance = torch.max(embedding_distance)
    normalized_distance = embedding_distance / max_distance
    
    cal_normalized_weights = 1 - normalized_distance
    sample_normalized_weight = 1

    for y in classes:
      ai = alphas[y] * sample_normalized_weight

      # non-conformity scores
      a = self.alphas * cal_normalized_weights
      
      # calculate p-score
      c = torch.count_nonzero(a >= ai)
      p_score = c / len(a)

      if p_score > significance_level:
        prediction_region.append(y)
    # END: for
      
    return prediction_region

class LegacyConformalClassifier():
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
                        raise ValueError(
                            "Expected self.mondrian_taxonomy, but found None")

                    if self.cumulative_taxonomy:
                        a = self.alphas[torch.logical_and(
                            self.mondrian_taxonomy <= mondrian_category, self.y == y)]
                    else:
                        a = self.alphas[torch.logical_and(
                            self.mondrian_taxonomy == mondrian_category, self.y == y)]

                else:  # class-based mondrian
                    a = self.alphas[self.y == y]

            else:  # not mondrian
                a = self.alphas

            # calculate p-score
            c = torch.count_nonzero(a >= ai)
            p_score = c / len(a)

            if p_score > significance_level:
                prediction_region.append(y)
        # END: for

        return prediction_region
