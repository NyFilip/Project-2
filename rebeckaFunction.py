import numpy as np
import matplotlib.pyplot as plt
import dataSet
import nilsFunction

def variance_feature_selection(data, num_features_to_select):

# Select features based on highest variance.

    features = data[:, 1:]  # ignore label column

    variances = np.var(features, axis=0)

    # Get indices of top features with highest variance
    top_features = np.argsort(variances)[-num_features_to_select:]

    # Shift indices by 1
    selected_features = list(top_features + 1)

    return selected_features
