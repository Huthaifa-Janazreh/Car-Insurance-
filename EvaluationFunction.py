# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn import set_config 
set_config(transform_output='pandas')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report



def classification_metrics(y_true, y_pred, label='', output_dict=False,
                           figsize=(8, 4), normalize='true', cmap='Blues', colorbar=False):
    # Print header and classification report
    header = "-" * 70
    print(header, f" Classification Metrics: {label} ", header, sep='\n')
    print(classification_report(y_true, y_pred, digits=3))
    # Plot confusion matrices
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    # Raw counts
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, normalize=None, cmap='gist_gray', colorbar=colorbar, ax=axes[0]
    )
    axes[0].set_title("Raw Counts")
    # Normalized
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, normalize=normalize, cmap=cmap, colorbar=colorbar, ax=axes[1]
    )
    axes[1].set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    plt.show()
    # Optional: return dictionary of metrics
    if output_dict:
        return classification_report(y_true, y_pred, output_dict=True)

def evaluate_classification(model, X_train, y_train, X_test, y_test,
                         figsize=(6,4), normalize='true', output_dict = False,
                            cmap_train='Blues', cmap_test="Reds",colorbar=False):
  # Get predictions for training data
  y_train_pred = model.predict(X_train)
  # Call the helper function to obtain regression metrics for training data
  results_train = classification_metrics(y_train, y_train_pred, #verbose = verbose,
                                     output_dict=True, figsize=figsize,
                                         colorbar=colorbar, cmap=cmap_train,
                                     label='Training Data')
  print()
  # Get predictions for test data
  y_test_pred = model.predict(X_test)
  # Call the helper function to obtain regression metrics for test data
  results_test = classification_metrics(y_test, y_test_pred, #verbose = verbose,
                                  output_dict=True,figsize=figsize,
                                         colorbar=colorbar, cmap=cmap_test,
                                    label='Test Data' )
  if output_dict == True:
    # Store results in a dataframe if ouput_frame is True
    results_dict = {'train':results_train,
                    'test': results_test}
    return results_dict

def regression_metrics(y_true, y_pred, label = '', verbose= True, output_dict = False):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    # rmse = mean_squared_error(y_true, y_pred, squared = False)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_true, y_pred)
    if verbose == True:
        header = "-"*60
        print(header, f"Regression Metrics: {label}", header, sep = '\n')
        print(f"- MAE = {mae:,.3f}")
        print(f"- MSE = {mse:,.3f}")
        print(f"- RMSE = {rmse:,.3f}")
        print(f"- R² = {r_squared:,.3f}")
    if output_dict == True:
        metrics = {'label':label, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R sqaured': r_squared}
        return metrics
def evaluate_regression(reg, X_train, y_train, X_test, y_test, verbose = True, output_frame = False):
    y_train_pred = reg.predict(X_train)
    results_train = regression_metrics(y_train, y_train_pred, verbose = verbose, output_dict = output_frame, label = 'Training Data')
    
    print()
    
    y_test_pred = reg.predict(X_test)
    results_test = regression_metrics(y_test, y_test_pred, verbose = verbose, output_dict= output_frame, label = 'Test Data')
    if output_frame: 
        results_df = pd.DataFrame([results_train, results_test])
        # results_df = results_df.set_index('label')
        return res

