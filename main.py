from models.ahp import ahp
from models.wsm import wsm
from models.topsis import topsis
from models.promethee import promethee, make_linear_fn, make_usual_fn, make_u_shape_fn, make_v_shape_fn, make_level_fn, make_gaussian_fn
import numpy as np
import pandas as pd

# Load dataset
file_path = '../data/data.csv'
data = pd.read_csv(file_path)

# Select relevant columns for MCDA
selected_columns = ['Revenues ($M)', 'Profits ($M)', 'Assets ($M)']
criteria = data[selected_columns]

# Clean dataset: Check for and handle missing or invalid values
if criteria.isnull().any().any():
    print("Warning: The dataset contains missing values. Filling them with 0.")
    criteria = criteria.fillna(criteria.mean())


# Normalize the data
def normalize(matrix):
    max_values = matrix.max()
    if (max_values == 0).any():
        raise ValueError("Cannot normalize: One or more columns have a maximum value of 0.")
    return matrix / max_values

def normalize_topsis(matrix):
    norm_values = np.sqrt((matrix ** 2).sum(axis=0))  # Euclidean norm for each column
    return matrix / norm_values


normalized_criteria = normalize(criteria)
normalized_topsis= normalize_topsis(criteria)


# Define weights and types
weights = np.array([0.4, 0.3, 0.3])
types = np.array([1, 1, 1])

preference_functions = {
    "Linear": lambda maximize: [make_linear_fn(maximize, q=0.1, p=0.3) for _ in range(criteria.shape[1])],
    "Usual": lambda maximize: [make_usual_fn(maximize) for _ in range(criteria.shape[1])],
    "U-Shape": lambda maximize: [make_u_shape_fn(maximize, q=0.1) for _ in range(criteria.shape[1])],
    "V-Shape": lambda maximize: [make_v_shape_fn(maximize, p=0.3) for _ in range(criteria.shape[1])],
    "Level": lambda maximize: [make_level_fn(maximize, q=0.1, p=0.3) for _ in range(criteria.shape[1])],
    "Gaussian": lambda maximize: [make_gaussian_fn(maximize, s=0.5) for _ in range(criteria.shape[1])]
}

# Generate initial PROMETHEE results with default linear preference function
p_fn_linear = preference_functions["Linear"](maximize=True)
scores_promethee_linear = promethee(criteria.values, p_fn_linear, weights)


#Model scores
scores_wsm = wsm(normalized_criteria, weights)
scores_topsis = topsis(normalized_topsis, weights)
scores_ahp_eigenvector = ahp(normalized_criteria, weights, method=1)
scores_ahp_normalized = ahp(normalized_criteria, weights, method=2)
scores_ahp_geometric = ahp(normalized_criteria, weights, method=3)


# Combine results
results = pd.DataFrame({
    'WSM': scores_wsm.values,  # Align indices
    'TOPSIS': scores_topsis.values,
    'AHP': scores_ahp_eigenvector.values,
    'PROMETHEE (Linear)': scores_promethee_linear
}, index=data['Name'])

results_WSM = pd.DataFrame({
    'WSM': scores_wsm.values,  # Align indices
}, index=data['Name'])

results_AHP_eigenvector = pd.DataFrame({
    'AHP': scores_ahp_eigenvector.values,  # Align indices
}, index=data['Name'])

results_AHP_normalized = pd.DataFrame({
    'AHP': scores_ahp_normalized.values,  # Align indices
}, index=data['Name'])

results_AHP_geometric = pd.DataFrame({
    'AHP': scores_ahp_geometric.values,  # Align indices
}, index=data['Name'])

results_topsis = pd.DataFrame({
    'TOPSIS': scores_topsis.values,  # Align indices
}, index=data['Name'])

results_promethee_linear = pd.DataFrame({
    'PROMETHEE': scores_promethee_linear
}, index=data['Name'])



"""
Explanation of Results
Berkshire Hathaway:

Performs exceptionally well across all criteria, making it the top performer in every method.
Its dominance is reflected in the PROMETHEE method, where it has a very high net flow.
Apple and Amazon:

Both companies also perform well, consistently ranking near the top, especially in methods like WSM and TOPSIS. They excel in profitability and asset management.
Walmart:

Shows a balanced performance but lags behind the top three due to slightly lower profitability metrics.
Costco Wholesale and Cencora:

These companies have weaker performance metrics, which is evident across all methods.
Investment Recommendation
If you had $10,000 to invest:

Berkshire Hathaway would be the optimal choice. Its consistent dominance across criteria suggests robust financial health and strong performance across revenues, profits, and assets.
"""