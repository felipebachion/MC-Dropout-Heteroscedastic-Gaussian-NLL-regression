# -*- coding: utf-8 -*-
"""MC-dropout regression of organic carbon from OSSL MIR spectra. ---- READ ALL THE COMMENTS! 


############################################################################################################################################
General code for MC Dropout + Heteroscedastic Gaussian NLL regression
=============================================================
This is general code that can be used in soil spectroscopy regression applications.
More details about its implementation can be found in our paper:

xxxxxxxxxxxxx


This implementation follows the MC Dropout / heteroscedastic likelihood
framework described by Kendall & Gal (2017) - What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
https://arxiv.org/abs/1703.04977


###########################################################################################################################################


    To run this code, you can download the reduced version of OSSL at:
    https://github.com/felipebachion/MC-Dropout-Heteroscedastic-Gaussian-NLL-regression

#After download the file, ADD THE PATH IN THE SETTINGS SECTION:
    SCRIPT_DIR = Path(r"D:\MC_example_paper").resolve() # IN THIS CASE THE EXCEL FILE IS LOCATED AT D:\MC_example_paper




    You can download the full dataset at:
        https://docs.soilspectroscopy.org/about.html
        
    A paper review can be found at:
        Safanelli, J. L., Hengl, T., Parente, L. L., Minarik, R., Bloom, D. E., Todd-Brown, K., … Sanderman, J. (2025). 
        Open Soil Spectral Library (OSSL): Building reproducible soil calibration models through open development and community engagement.
        PloS One, 20(1), e0296545. doi:10.1371/journal.pone.0296545.



###########################################################################################################################################
Input workbook layout
---------------------
* Column A: organic-carbon reference value.
* Columns B onward: MIR absorbance/transmittance columns named
  ``scan_mir.<wavenumber>_abs`` (for example, ``scan_mir.602_abs``).

The script creates consecutive sample identifiers, preprocesses the spectra
with a first Savitzky-Golay derivative, removes the 2282--2396 cm-1 CO2 band,
and reserves 33% of the samples for external validation using
Kennard-Stone selection.

"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import savgol_filter
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from matplotlib.ticker import PercentFormatter
%matplotlib qt


# =============================================================================
# Settings
# =============================================================================

SCRIPT_DIR = Path(r"D:\MC_example_paper").resolve() #     # Update bellow the path where you downloaded the excel file containing the MIR spectra
EXCEL_FILE = SCRIPT_DIR / "ossl_red.xlsx"
VALIDATION_FRACTION = 0.33
N_MC_PASSES = 50
RANDOM_STATE = 42
Z_90 = 1.645


# =============================================================================
# 1. Load the OSSL spreadsheet
# =============================================================================

def load_ossl_mir(excel_file):
    """Load organic carbon and spectra from the OSSL spreadsheet.

               
    The first spreadsheet column is used as the organic-carbon reference.
    Spectral columns are identified from their names, not from their position,
    so extra metadata columns will not be included accidentally.
    
    
    """
    
    
    data = pd.read_excel(excel_file)



    if data.shape[1] < 2:
        raise ValueError("The workbook must contain one reference column and spectral columns.")

    reference_column = data.columns[0]
    spectral_columns = []
    wavenumbers = []
    pattern = re.compile(r"^scan_mir\.(\d+(?:\.\d+)?)_abs$", re.IGNORECASE)

    for column in data.columns[1:]:
        match = pattern.match(str(column).strip())
        if match is not None:
            spectral_columns.append(column)
            wavenumbers.append(float(match.group(1)))

    if not spectral_columns:
        raise ValueError(
            "No MIR columns were found. Expected headers such as 'scan_mir.602_abs'."
        )

    # Sort spectra by their physical wavenumber, independent of Excel order.
    order = np.argsort(wavenumbers)
    spectral_columns = [spectral_columns[index] for index in order]
    wavenumbers = np.asarray(wavenumbers, dtype=float)[order]

    y = pd.to_numeric(data[reference_column], errors="coerce").to_numpy(dtype=float)
    X = data[spectral_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Keep only samples with a reference value and a complete MIR spectrum.
    valid_rows = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if not np.all(valid_rows):
        removed = np.count_nonzero(~valid_rows)
        print(f"Removing {removed} row(s) with missing or non-numeric values.")
        X = X[valid_rows]
        y = y[valid_rows]

    if X.shape[0] < 3:
        raise ValueError("At least three complete samples are required.")

    # IDs identify the samples remaining after incomplete rows are removed.
    sample_ids = np.arange(1, X.shape[0] + 1)

    print(f"Reference column: {reference_column}")
    print(f"Loaded {X.shape[0]} samples and {X.shape[1]} MIR variables.")
    print(f"Wavenumber range: {wavenumbers[0]:g} to {wavenumbers[-1]:g} cm^-1")

    return sample_ids, X, y, wavenumbers


# =============================================================================
# 2. Spectral preprocessing and inspection plot
# =============================================================================

def preprocess_spectra(X, wavenumbers):
    """Calculate first derivatives and remove the CO2 absorption region."""


    # First derivative: same preprocessing idea as the supplied ISIS example.
    X_derivative = savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)

    # Remove CO2 physically, rather than using indices from a different spectrum.
    keep = ~((wavenumbers >= 2282) & (wavenumbers <= 2396))
    if not np.any(~keep):
        print("Warning: no variables were found in the 2282--2396 cm^-1 CO2 region.")

    return X_derivative[:, keep], wavenumbers[keep]


def plot_spectra(X, wavenumbers):
    """Plot all preprocessed OSSL MIR spectra."""
    plt.figure(figsize=(10, 6))
    for spectrum in X:
        plt.plot(wavenumbers, spectrum, color="blue", alpha=0.3)

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("First-derivative intensity")
    plt.title("Preprocessed spectra from Open Soil Spectral Library")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. Kennard-Stone external validation split
# =============================================================================

def kennard_stone_split(X, validation_fraction=0.33):
    """Return calibration and validation indices using Kennard-Stone selection.

    Kennard-Stone first chooses two spectra that are far apart, then repeatedly
    adds the sample furthest from the selected set. This makes the validation
    set span the spectral space rather than being a random subset.
    """
    n_samples = X.shape[0]
    n_validation = int(np.ceil(validation_fraction * n_samples))
    n_validation = min(max(n_validation, 2), n_samples - 1)

    squared_norms = np.sum(X ** 2, axis=1, keepdims=True)
    distances_squared = squared_norms + squared_norms.T - 2 * X @ X.T
    distances_squared = np.maximum(distances_squared, 0.0)

    first, second = np.unravel_index(np.argmax(distances_squared), distances_squared.shape)
    selected = [int(first), int(second)]
    available = np.ones(n_samples, dtype=bool)
    available[selected] = False

    while len(selected) < n_validation:
        candidates = np.flatnonzero(available)
        nearest_selected_distance = distances_squared[np.ix_(candidates, selected)].min(axis=1)
        next_sample = candidates[np.argmax(nearest_selected_distance)]
        selected.append(int(next_sample))
        available[next_sample] = False

    validation_indices = np.sort(np.asarray(selected, dtype=int))
    calibration_indices = np.flatnonzero(available)
    return calibration_indices, validation_indices


# =============================================================================
# 4. MC Dropout model and heteroscedastic Gaussian loss
# =============================================================================

def heteroscedastic_loss(y_true, y_pred):
    # This makes the loss safe for either (batch_size,) or (batch_size, 1) y.
    y_true = tf.reshape(y_true, [-1])

    mean = y_pred[:, 0]
    log_variance = y_pred[:, 1]
    precision = tf.exp(-log_variance)

    loss = 0.5 * log_variance + 0.5 * precision * tf.square(y_true - mean)
    return tf.reduce_mean(loss)


def build_model(input_dim, dropout_rate=0.10, learning_rate=0.001):
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation="relu")(inputs)
    x = Dropout(dropout_rate)(x, training=True)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout_rate)(x, training=True)

    mean = Dense(1)(x)
    log_variance = Dense(1)(x)
    outputs = Concatenate()([mean, log_variance])

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=heteroscedastic_loss)
    return model


def optimise_hyperparameters(X, y):
    """Select settings by five-fold CV within calibration samples only."""
    dropout_rates = [0.05, 0.10, 0.20]
    learning_rates = [0.0005, 0.001]
    epochs_list = [50, 100, 150]
    batch_sizes = [64, 128]

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_rmse = np.inf
    best_parameters = None

    for dropout_rate in dropout_rates:
        for learning_rate in learning_rates:
            for epochs in epochs_list:
                for batch_size in batch_sizes:
                    fold_rmse = []

                    for train_idx, cv_idx in kfold.split(X):
                        model = build_model(
                            input_dim=X.shape[1],
                            dropout_rate=dropout_rate,
                            learning_rate=learning_rate,
                        )
                        model.fit(
                            X[train_idx],
                            y[train_idx],
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0,
                        )

                        # Mean predictions are used for model selection.
                        prediction = model(X[cv_idx], training=False).numpy()[:, 0]
                        fold_rmse.append(np.sqrt(mean_squared_error(y[cv_idx], prediction)))

                    mean_rmse = np.mean(fold_rmse)
                    print(
                        f"Dropout={dropout_rate}, LR={learning_rate}, "
                        f"Epochs={epochs}, Batch={batch_size}, CV RMSE={mean_rmse:.4f}"
                    )

                    if mean_rmse < best_rmse:
                        best_rmse = mean_rmse
                        best_parameters = {
                            "dropout_rate": dropout_rate,
                            "learning_rate": learning_rate,
                            "epochs": epochs,
                            "batch_size": batch_size,
                        }

    print("\nBest hyperparameters:", best_parameters)
    print(f"Best CV RMSE: {best_rmse:.4f}")
    return best_parameters


def predict_mc(model, X, n_mc=N_MC_PASSES):
    """Return mean prediction and epistemic, aleatoric, predictive SDs."""
    predictions = np.array([model(X, training=True).numpy() for _ in range(n_mc)])
    means = predictions[:, :, 0]
    log_variances = predictions[:, :, 1]

    prediction = np.mean(means, axis=0)
    epistemic_variance = np.var(means, axis=0)
    aleatoric_variance = np.mean(np.exp(log_variances), axis=0)
    predictive_variance = epistemic_variance + aleatoric_variance

    return (
        prediction,
        np.sqrt(epistemic_variance),
        np.sqrt(aleatoric_variance),
        np.sqrt(predictive_variance),
    )


# =============================================================================
# 5. Train, validate, and save predictions
# =============================================================================

#%% 5.1 Load and preprocess the spreadsheet data
tf.keras.utils.set_random_seed(RANDOM_STATE)

sample_ids, X_raw, y, wavenumbers = load_ossl_mir(EXCEL_FILE)
X, wavenumbers = preprocess_spectra(X_raw, wavenumbers)
plot_spectra(X, wavenumbers)


#%% 5.2 Create calibration and external-validation datasets
calibration_idx, validation_idx = kennard_stone_split(X, VALIDATION_FRACTION)

X_calibration = X[calibration_idx]
y_calibration = y[calibration_idx]
X_validation = X[validation_idx]
y_validation = y[validation_idx]

print(
    f"Kennard-Stone split: {len(calibration_idx)} calibration samples; "
    f"{len(validation_idx)} validation samples."
)


#%% 5.3 Optimise model hyperparameters using calibration samples
best_parameters = optimise_hyperparameters(X_calibration, y_calibration)


#%% 5.4 Train the final model calibration samples
model = build_model(
    input_dim=X_calibration.shape[1],
    dropout_rate=best_parameters["dropout_rate"],
    learning_rate=best_parameters["learning_rate"],
)

model.fit(
    X_calibration,
    y_calibration,
    epochs=best_parameters["epochs"],
    batch_size=best_parameters["batch_size"],
    verbose=1,
)


#%% 5.5 Predict external-validation samples and calculate performance
prediction, epistemic, aleatoric, predictive = predict_mc(model, X_validation)

validation_rmse = np.sqrt(mean_squared_error(y_validation, prediction))
validation_r2 = r2_score(y_validation, prediction)

print(f"External validation RMSE: {validation_rmse:.4f}")
print(f"External validation R2: {validation_r2:.4f}")


#%% 5.6 Calculate 90% predictive intervals
# The uncertainty values returned by predict_mc are standard deviations.
# A 90% Gaussian prediction interval uses z = 1.645.
predictive_interval_half_width = Z_90 * predictive
pi90_lower = prediction - predictive_interval_half_width
pi90_upper = prediction + predictive_interval_half_width
picp_90 = np.mean((y_validation >= pi90_lower) & (y_validation <= pi90_upper))

print(f"90% prediction-interval coverage (PICP): {picp_90:.1%}")


#%% 5.7 Save validation predictions and uncertainty estimates
results = pd.DataFrame(
    {
        "sample_id": sample_ids[validation_idx],
        "organic_carbon_reference": y_validation,
        "organic_carbon_prediction": prediction,
        "epistemic_sd": epistemic,
        "aleatoric_sd": aleatoric,
        "predictive_sd": predictive,
        "pi90_lower": pi90_lower,
        "pi90_upper": pi90_upper,
    }
)

results_file = SCRIPT_DIR / "ossl_red_validation_predictions.csv"
results.to_csv(results_file, index=False)
print(f"Validation predictions saved to: {results_file}")


#%% 5.8 Plot external-validation predictions and 90% predictive intervals
plt.figure(figsize=(6, 6))
plt.errorbar(
    y_validation,
    prediction,
    yerr=predictive_interval_half_width,
    fmt="o",
    alpha=0.7,
    capsize=2,
    label="Spectroscopy Uncertainty (z = 1.645)",
)

limits = [min(y_validation.min(), prediction.min()), max(y_validation.max(), prediction.max())]
plt.plot(limits, limits, "k--", label="1:1 line")
plt.xlabel("Reference organic carbon")
plt.ylabel("Predicted organic carbon")
plt.title("External validation: 90% predictive interval")
plt.text(
    0.05,
    0.95,
    f"$R^2_{{val}}$ = {validation_r2:.3f}\nRMSEP = {validation_rmse:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
)
plt.legend()
plt.tight_layout()
plt.show()


#%% 5.9 Reliability plot and probability integral transform (PIT) histogram
# PIT values should be uniformly distributed when predictive uncertainties are
# well calibrated. The shaded region is a 90% binomial confidence band.
y_std = np.clip(predictive, 1e-6, None)
u = norm.cdf((y_validation - prediction) / y_std)

quantiles = np.linspace(0.05, 0.95, 19)
empirical = np.array([(u <= quantile).mean() for quantile in quantiles])

n_validation = len(u)
standard_error = np.sqrt(quantiles * (1 - quantiles) / n_validation)
reliability_lower = np.clip(quantiles - Z_90 * standard_error, 0, 1)
reliability_upper = np.clip(quantiles + Z_90 * standard_error, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.fill_between(
    quantiles,
    reliability_lower,
    reliability_upper,
    alpha=0.2,
    label="90% confidence band",
)
ax.plot(quantiles, empirical, marker="o", label="Model")
ax.plot([0, 1], [0, 1], "k--", label="Ideal")
ax.set_xlabel("PI level", fontsize=12)
ax.set_ylabel("PICP", fontsize=12)
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_title("Reliability plot", fontsize=14)
ax.legend()
ax.grid(True)

ax = axes[1]
ax.hist(u, bins=20, density=True, alpha=0.7)
ax.axhline(1, linestyle="--", color="black", label="Ideal uniform density")
ax.set_xlabel("Percentile", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.set_title("PIT histogram", fontsize=14)
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(
    SCRIPT_DIR / "ossl_red_reliability_pit_90.tiff",
    dpi=600,
    format="tiff",
    pil_kwargs={"compression": "tiff_lzw"},
)
plt.show()


#%% 5.10 Plot 90% epistemic and aleatoric uncertainty intervals
mean_epistemic = np.mean(epistemic)
mean_aleatoric = np.mean(aleatoric)
plot_limits = [min(y_validation.min(), prediction.min()), max(y_validation.max(), prediction.max())]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].errorbar(
    y_validation,
    prediction,
    yerr=Z_90 * epistemic,
    fmt="o",
    color="blue",
    alpha=0.5,
    label="90% epistemic interval",
)
axes[0].plot(plot_limits, plot_limits, "k--")
axes[0].set_xlabel("Reference organic carbon", fontsize=12)
axes[0].set_ylabel("Predicted organic carbon", fontsize=12)
axes[0].set_title("Epistemic uncertainty (model uncertainty)", fontsize=14)
axes[0].legend()
axes[0].grid(False)
axes[0].text(
    0.05,
    0.95,
    f"Mean epistemic SD: {mean_epistemic:.4f}",
    transform=axes[0].transAxes,
    verticalalignment="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
)

axes[1].errorbar(
    y_validation,
    prediction,
    yerr=Z_90 * aleatoric,
    fmt="o",
    color="green",
    alpha=0.5,
    label="90% aleatoric interval",
)
axes[1].plot(plot_limits, plot_limits, "k--")
axes[1].set_xlabel("Reference organic carbon", fontsize=12)
axes[1].set_ylabel("Predicted organic carbon", fontsize=12)
axes[1].set_title("Aleatoric uncertainty (data noise)", fontsize=14)
axes[1].legend()
axes[1].grid(False)
axes[1].text(
    0.05,
    0.95,
    f"Mean aleatoric SD: {mean_aleatoric:.4f}",
    transform=axes[1].transAxes,
    verticalalignment="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
)

fig.tight_layout()
fig.savefig(
    SCRIPT_DIR / "ossl_red_epistemic_aleatoric_90.tiff",
    dpi=600,
    format="tiff",
    pil_kwargs={"compression": "tiff_lzw"},
)
plt.show()
