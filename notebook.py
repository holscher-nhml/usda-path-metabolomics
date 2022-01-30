#!/usr/bin/env python
# coding: utf-8

# Notes
# - Please see README.md for additional information.
#
# - This notebook requires Python 3.7+. Install dependencies via "pip install -r requirements.txt".
#
# - All plt.show() lines have been commented out so the script can run in "headless" mode.
#   Figures are saved locally.
#
# - "Fast mode" is enabled, signifcantly reducing the model selection time
#    at the cost of no cross-validation. Fast mode can be disabled to produce
#    accurate results. Fast mode was *NOT* used by the authors of the manuscript.
#
# - The notebook has been slightly modified compared to the original to not rely on
#   external dependencies that are not distributed by the authors.
#
# Author: Aditya S. Mansharamani, adityam5@illinois.edu
# This code, along with any accompanying source code files ONLY,
# are released under the license specified in LICENSE.md.

# ## Metabolomics Data - Difference Model

# ## Setup, imports, & formatting

# In[1]:


# imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import numpy as np
from IPython.display import Markdown, display
from sklearn.preprocessing import QuantileTransformer
import csv
import pickle

np.random.seed(1)

# formatting
# get_ipython().run_line_magic('load_ext', 'lab_black')
sns.set()
params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (15, 10),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}
plt.rcParams.update(params)


# ## Load Raw Data

# In[2]:


# Loads data files
def load_data(foods=None, split_grains=False):

    # Load files
    metadata = pd.read_csv("./metadata_fuzzed.csv").set_index("Key")
    met_baseline = pd.read_csv("./metabolomics_baseline_fuzzed.csv").set_index("Key")
    met_end = pd.read_csv("./metabolomics_end_fuzzed.csv").set_index("Key")

    # Print raw data shape
    print(
        f"Raw metabolomics baseline rows: {len(met_baseline)}, end rows: {len(met_end)}"
    )
    print(
        f"Raw metabolomics baseline columns: {len(met_baseline.columns)}, end columns: {len(met_end.columns)}"
    )
    print(
        f"\tIntersection of columns: {len(set(met_baseline.columns).intersection(set(met_end.columns)))}"
    )

    # Subset foods with supplied regex
    if foods is not None:
        met_baseline = met_baseline.filter(regex=foods, axis=0)
        met_end = met_end.filter(regex=foods, axis=0)
    else:
        print("No food filter regex supplied!")
        pass

    # Split grains if needed
    if split_grains:

        def update_index(i):
            # Ignore all non-grains
            if "Grains" not in i:
                return i
            # Map all nograins to nobarley for now
            if "NoGrains" in i:
                return i.replace("Grains", "Barley")
            # Map Grains --> Barley or Oats
            metadata_row = metadata.loc[i]
            i = i.split(".")
            return f"{i[0]}.{i[1]}.{metadata_row.Treatment2}"

        # Apply index update
        met_baseline.index = met_baseline.index.map(update_index)
        met_end.index = met_end.index.map(update_index)

        # Extract a copy of all "barley" (i.e. grains) control
        met_baseline_nobarley = met_baseline.filter(like="NoBarley", axis=0).copy()
        met_end_nobarley = met_end.filter(like="NoBarley", axis=0).copy()

        # Change the copy to oats control
        def update_index(i):
            return i.replace("Barley", "Oats")

        # Apply index update
        met_baseline_nobarley.index = met_baseline_nobarley.index.map(update_index)
        met_end_nobarley.index = met_end_nobarley.index.map(update_index)

        # Add copy of grains control to the dataset
        met_baseline = pd.concat([met_baseline, met_baseline_nobarley])
        met_end = pd.concat([met_end, met_end_nobarley])

    # Modify IDs to remove period qualifiers so we can subtract on index
    met_baseline.index = met_baseline.index.map(lambda i: i.replace(".Baseline", ""))
    met_end.index = met_end.index.map(lambda i: i.replace(".End", ""))

    return metadata, met_baseline, met_end


# In[3]:


metadata, met_baseline, met_end = load_data()


# ## Drop/fix mising values

# In[4]:


# Keep features that have proportion of missing values <  p for any food
def drop_missing_values(met_baseline, met_end, metadata, p):

    columns_to_keep_baseline = set()
    columns_to_keep_end = set()

    for study in set(metadata.Study):
        # Select dataset for this study
        met_baseline_study = met_baseline.filter(like=study, axis=0)
        met_end_study = met_end.filter(like=study, axis=0)

        # Compute percent of missing values for the datasets
        p_baseline = met_baseline_study.isnull().sum() / len(met_baseline_study)
        p_end = met_end_study.isnull().sum() / len(met_end_study)

        # Keep all features that have < p percent missing
        # i.e. have > p percent features present
        p_baseline = p_baseline < p
        p_end = p_end < p

        # Subset feature list to only include those features
        p_baseline = p_baseline.where(lambda a: a).dropna().index
        p_end = p_end.where(lambda a: a).dropna().index

        # Add column to keep list
        columns_to_keep_baseline.update(list(p_baseline))
        columns_to_keep_end.update(list(p_end))

    # Subset columns
    met_baseline = met_baseline[list(columns_to_keep_baseline)]
    met_end = met_end[list(columns_to_keep_end)]

    # Print results
    print(
        f"Total number of columns after dropping missing (baseline, end) = {(len(columns_to_keep_baseline), len(columns_to_keep_end))}"
    )

    return met_baseline, met_end


# In[5]:


# Imputes missing values to uniform random values between [0, mm * minimum observed] for every feature
def impute_missing_values(met_baseline, met_end, mm):

    # Compute per-feature minimums for dataset
    met_baseline_feature_mins = np.min(met_baseline, axis=0)
    met_baseline_nan_dict = {}
    met_end_feature_mins = np.min(met_end, axis=0)
    met_end_nan_dict = {}

    # Create new datasets that contains random values for each subject for each feature,
    # between 0 and mm * the minimum for that feature
    for feature, minimum in met_baseline_feature_mins.iteritems():
        met_baseline_nan_dict[feature] = np.random.uniform(
            low=0, high=mm * minimum, size=len(met_baseline)
        )

    for feature, minimum in met_end_feature_mins.iteritems():
        met_end_nan_dict[feature] = np.random.uniform(
            low=0, high=mm * minimum, size=len(met_end)
        )

    # Update original dataset with new values for any missing entries
    # Original values should be preserved
    met_baseline_nan = pd.DataFrame(met_baseline_nan_dict)
    met_baseline_nan.index = met_baseline.index

    met_end_nan = pd.DataFrame(met_end_nan_dict)
    met_end_nan.index = met_end.index

    met_baseline.update(met_baseline_nan, overwrite=False)
    met_end.update(met_end_nan, overwrite=False)

    return met_baseline, met_end


# ## Prepare Data

# In[6]:


# Keeps only columns/subjects available in both datasets, and separates datasets into treatment + control
def subset_separate_data(met_baseline, met_end):

    # Compute intersection rows
    row_idxs = met_baseline.index.intersection(met_end.index)
    baseline_index = set(met_baseline.index)
    end_index = set(met_end.index)
    print(f"Missing from end: {baseline_index - end_index}")
    print(f"Missing from baseline: {end_index - baseline_index}")

    # Compute intersection columns
    col_idxs = met_baseline.columns.intersection(met_end.columns)

    # Subset datasets
    met_baseline = met_baseline.loc[row_idxs, col_idxs]
    met_end = met_end.loc[row_idxs, col_idxs]
    print(f"Lengths: {(len(met_baseline), len(met_end))}")
    assert len(met_baseline) == len(met_end)

    # Separate treatment/control rows
    row_idxs_treatment = [idx for idx in row_idxs if ".No" not in idx]
    row_idxs_control = [idx for idx in row_idxs if ".No" in idx]
    assert len(row_idxs_control) + len(row_idxs_treatment) == len(row_idxs)

    print(
        f"Remaining rows for (treatment, control, total): {(len(row_idxs_treatment), len(row_idxs_control), len(row_idxs))}"
    )

    met_baseline_cont = met_baseline.loc[row_idxs_control, col_idxs]
    met_baseline_treat = met_baseline.loc[row_idxs_treatment, col_idxs]
    met_end_cont = met_end.loc[row_idxs_control, col_idxs]
    met_end_treat = met_end.loc[row_idxs_treatment, col_idxs]

    # Extract labels for the treatments
    met_treatments = met_baseline_treat.index.map(lambda i: i.split(".")[-1])
    # Extract labels for the studies
    met_studies = met_baseline_cont.index.map(lambda i: i.split(".")[-1][2:])

    print(
        f"\tTotal number of subjects for (baseline, end) after subsetting = {(len(met_baseline), len(met_end))}"
    )
    print(
        f"\tTreatment subjects, Control subjects: ({len(met_treatments)}, {len(met_studies)})"
    )

    print(f"Total number of columns after subsetting = {len(col_idxs)}")

    return (
        met_baseline,
        met_end,
        met_baseline_cont,
        met_baseline_treat,
        met_end_cont,
        met_end_treat,
        met_treatments,
        met_studies,
    )


# In[7]:


# Subtracts two dataframes
def subtract_data(met_baseline, met_end):
    return met_end - met_baseline


# ## PCA

# In[8]:


def plot_pca(
    X,
    hue,
    title,
    plot_evr=False,
):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, QuantileTransformer

    # Scale input
    X = QuantileTransformer(n_quantiles=len(X)).fit_transform(X)

    # Compute PCA, plot
    pca = PCA(n_components=min(20, X.shape[1]), random_state=1)
    X_t = pca.fit_transform(X)

    if plot_evr:
        fig, (ax_pca, ax_evr) = plt.subplots(2, 1, figsize=(12, 8))
    else:
        fig, ax_pca = plt.subplots(1, 1, figsize=(12, 4))

    plt.tight_layout()
    sns.scatterplot(x=X_t[:, 0], y=X_t[:, 1], hue=hue, ax=ax_pca, s=100)
    ax_pca.set_title(f" {title} PCA")

    # Plot explained variance ratio
    if plot_evr:
        ax_evr.plot(pca.explained_variance_ratio_)
        ax_evr.set_title("Explained Variance Ratio")
        ax_evr.set_xlabel("PC #")
        ax_evr.set_ylabel("Explained Variance Ratio")

    # plt.show()


# # Panel Plots

# In[9]:


def panel_plots(
    met_diff_treat,
    met_treatments,
    met_diff_cont,
    met_studies,
    group1,  # always in the treatment group
    group2,  # could be in the treatment group, or "Control"
    feature_list,
    mapping={},
    color_map={
        "Almond": sns.color_palette()[0],
        "Walnut": sns.color_palette()[1],
        "Control": sns.color_palette()[7],
    },
):

    import matplotlib.ticker as ticker
    from matplotlib import font_manager, rc
    import glob

    # Setup text formatting
    # Disabled in this version of the code since we can't distribute font binaries
    """
    font_path = "../../../../other/myriad-pro-cufonfonts/MYRIADPRO-REGULAR.OTF"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()
    rc("text", usetex=False)  # Can't use custom font with tex
    # Add remaining fonts
    for font in glob.glob("../../../../other/myriad-pro-cufonfonts/*.OTF"):
        font_manager.fontManager.addfont(font)
    """

    assert len(feature_list) >= 1

    # Extract group data
    group1_data = met_diff_treat[met_treatments == group1].copy()
    group1_data["group"] = group1
    group2_data = (
        met_diff_cont[met_studies == group1].copy()
        if group2 == "Control"
        else met_diff_treat[met_treatments == group2].copy()
    )
    group2_data["group"] = group2

    data = pd.concat([group1_data, group2_data])

    # Make plots
    fig, axs = plt.subplots(
        nrows=1, ncols=len(feature_list), figsize=(7.3, 4.5), sharex="all", dpi=300
    )

    fig.suptitle("", size=22)
    for i, (feature, ax) in enumerate(zip(feature_list, axs.flat)):
        sns.boxplot(
            data=data,
            x="group",
            y=feature,
            linewidth=3,
            width=0.6,
            ax=ax,
            palette=color_map,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(feature if feature not in mapping else mapping[feature])
        ax.text(
            0.05,
            0.92,
            "A" if i == 0 else "B",
            transform=ax.transAxes,
            fontsize=16,
            fontweight=1000,
        )
        locator = ticker.MaxNLocator(
            nbins=3, integer=True, symmetric=True, min_n_ticks=4, prune="both"
        )
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ticks = locator()

    fig.suptitle(f"{group1} vs. {group2}", size=22)
    fig.supxlabel("Group", size=20)
    fig.supylabel("Δ Relative concentration", size=20)
    plt.tight_layout()

    fig.savefig(
        f"panel-boxplot-{group1}-{group2}.svg", bbox_inches="tight", format="svg"
    )
    fig.savefig(f"panel-boxplot-{group1}-{group2}.png", bbox_inches="tight")
    plt.close(fig)


# ## Classification

# In[10]:


def classification(X, y, title, X_control=None, y_control=None, fast_mode=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut
    from sklearn.model_selection import cross_val_predict, cross_validate
    from sklearn.model_selection import ParameterGrid, GridSearchCV
    from sklearn.metrics import classification_report, roc_auc_score
    from scikitplot.metrics import plot_confusion_matrix
    import numpy as np

    # naively assume the output classes are in alphabetical order. if something breaks, look here!
    classes = sorted(list(set(y)))

    # setup directory
    fdir = "-".join(c[:3] for c in classes)
    os.makedirs(fdir, exist_ok=True)

    print(f"------- {title} -------")

    param_grid = {
        "n_estimators": [5000 if not fast_mode else 1000],
        "oob_score": [True],
        "n_jobs": [-1],
        "random_state": [1],
        "max_features": [None, "sqrt", "log2"],
        "min_samples_leaf": [1, 3, 5],
    }

    best_rf = None
    best_params = None
    for params in ParameterGrid(param_grid):
        rfc = RandomForestClassifier()
        rfc.set_params(**params)

        # Perform LOO evaluation for this parameter set
        cv_result = cross_validate(
            rfc,
            X.values,
            y,
            scoring=None,
            cv=LeaveOneOut(),
            n_jobs=-1,
            return_estimator=True,
        )

        # Update the best parameters
        estimators = cv_result["estimator"]
        for estimator in estimators:
            if best_rf is None or estimator.oob_score_ > best_rf.oob_score_:
                best_rf = estimator
                best_params = params
        # early exit
        if fast_mode:
            break

    print(
        f"Best params for multi-food classification ({title}) were {best_params}. Fast mode was: {fast_mode}"
    )

    # Cross-val predict probabilities using leave one out and our new best parameters
    rfc = RandomForestClassifier()
    rfc.set_params(**best_params)

    y_proba = cross_val_predict(
        rfc, X.values, y, cv=LeaveOneOut(), n_jobs=-1, method="predict_proba"
    )

    # Convert probs to class scores
    y_pred = [classes[score.argmax()] for score in y_proba]

    # Try to compute ROC AUC if possible
    roc_auc = None
    try:
        if len(classes) > 2:
            roc_auc = roc_auc_score(y, y_proba, multi_class="ovr")
        else:
            roc_auc = roc_auc_score(y, y_proba[:, 1], multi_class="ovr")
    except Exception as e:
        print(e)
        print("Couldn't compute ROC AUC score!")

    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_confusion_matrix(y, y_pred, ax=ax)
    plt.tight_layout()
    plt.title(f"{title} Treatment")
    # plt.show()
    print(classification_report(y, y_pred))
    if roc_auc:
        print(f"ROC AUC = {roc_auc}")

    # Plot feature importance graph
    # rfc.fit(X, y)
    best_feature_idxs = np.argsort(best_rf.feature_importances_)[::-1]
    plt.figure(figsize=(5, 5))
    plt.title("Feature Importances")
    plt.xlabel("Feature #")
    plt.ylabel("Importance")
    plt.plot(sorted(best_rf.feature_importances_, reverse=True))
    # plt.show()
    best_features = X.columns[best_feature_idxs[:10]]
    print(best_features)

    # feautre importances
    best_features_list = list(
        zip(
            [X.columns[idx] for idx in best_feature_idxs],
            [best_rf.feature_importances_[idx] for idx in best_feature_idxs],
        )
    )
    with open(f"{fdir}/{title}-multifood-feature-importances.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["feature", "importance"])
        for idx in best_feature_idxs:
            w.writerow([X.columns[idx], best_rf.feature_importances_[idx]])

    # feature means per group
    X_gb = X.copy().iloc[:, best_feature_idxs]
    X_gb["treatment"] = y
    X_gb.groupby("treatment").mean().to_csv(
        f"{fdir}/{title}-multifood-feature-means.csv"
    )
    X_gb.groupby("treatment").std().to_csv(f"{fdir}/{title}-multifood-feature-stds.csv")

    # plot features
    for feature in best_features:
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.boxplot(
            data=X_gb, x="treatment", y=feature, linewidth=2.5, width=0.4, ax=ax
        )
        ax.set_xlabel("Treatment group")
        ax.set_ylabel("Relative concentration")
        fig.suptitle(feature, size=22)
        fig.savefig(f"{fdir}/{title}-{feature}-boxplot.png", bbox_inches="tight")
        plt.close(fig)

    # Control group classification using the best model
    if X_control is not None:
        y_proba_control = best_rf.predict_proba(X_control.values)
        y_pred_control = [classes[score.argmax()] for score in y_proba_control]

        roc_auc_control = None
        try:
            roc_auc_control = roc_auc_score(
                y_control, y_proba_control[:, 1], multi_class="ovr"
            )
        except:
            print("Couldn't compute ROC AUC score!")

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_confusion_matrix(y_control, y_pred_control, ax=ax)
        plt.tight_layout()
        plt.title(f"Control Group - {title}")
        # plt.show()

        print(classification_report(y_control, y_pred_control))
        if roc_auc_control:
            print(f"ROC AUC = {roc_auc_control}")

    print("-------------------------")

    return best_params, best_features_list


# ## Batch effect removal

# In[11]:


def remove_batch_effect(df_cont, cont_labels, df_treat, treat_labels, n=10):
    from sklearn.preprocessing import QuantileTransformer

    df_treat_columns = df_treat.columns
    df_cont_columns = df_cont.columns
    dfs_cont = []
    dfs_treat = []

    for food in set(treat_labels):
        # control data for this study
        df_cont_food = df_cont[cont_labels == food]
        # treatment data for this study
        df_treat_food = df_treat[treat_labels == food]
        # center the treatment group on the median of the control
        df_treat_food -= df_cont_food.median(0)
        # center the control group on their own median
        df_cont_food -= df_cont_food.median(0)
        # append to main list
        dfs_cont.append(df_cont_food)
        dfs_treat.append(df_treat_food)

    # merge dataframes
    df_cont = pd.concat(dfs_cont)
    df_treat = pd.concat(dfs_treat)

    # df = pd.concat([df_cont, df_treat])
    # qt = QuantileTransformer(n_quantiles=len(df), output_distribution="normal")
    # df[df.columns] = qt.fit_transform(df[df.columns])

    # split dataframes again
    # cont_index = df.index.map(lambda i: "No" in i)
    # df_cont = df.loc[cont_index]
    # df_treat = df.loc[[not a for a in cont_index]]

    # generate control basis vectors
    def decompose_pca(X, n=n):
        U, E, V = np.linalg.svd(X)
        return U[:, :n], E[:n], V[:, :n]

    # control_basis_vectors = {"all": decompose_pca(df_cont, n=2)}
    control_basis_vectors = {}
    mean_vectors = []
    for study in set(cont_labels):
        df_cont_food = df_cont[cont_labels == study]
        # control_basis_vectors[study] = decompose_pca(df_cont_food, n=3)
        mean_vectors.append(df_cont_food.mean(0).values)

    control_mean_vectors = decompose_pca(mean_vectors, n=len(mean_vectors))
    control_basis_vectors["all_mean"] = control_mean_vectors

    # Combines the dictionary of basis vectors into one list
    control_basis_vectors = np.concatenate(
        [control_basis_vectors[key][2] for key in control_basis_vectors], axis=1
    )

    control_basis_transformation = (
        control_basis_vectors
        @ np.linalg.inv((control_basis_vectors.T) @ control_basis_vectors)
        @ control_basis_vectors.T
    )

    # transform treatment
    df_treat = (
        (np.eye(control_basis_transformation.shape[0]) - control_basis_transformation)
        @ (df_treat).T
    ).T

    # transform control
    df_cont = (
        (np.eye(control_basis_transformation.shape[0]) - control_basis_transformation)
        @ (df_cont).T
    ).T

    df_treat.columns = df_treat_columns
    df_cont.columns = df_cont_columns

    return df_cont, df_treat


# ## Per-food logistic regression models

# In[12]:


def per_food_models(X_treat, y_treat, title, X_control, y_control, fast_mode=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut
    from sklearn.model_selection import cross_val_predict, cross_validate
    from sklearn.model_selection import ParameterGrid, GridSearchCV
    from sklearn.metrics import classification_report, roc_auc_score
    from scikitplot.metrics import plot_confusion_matrix
    from sklearn.feature_selection import SelectFromModel
    import numpy as np

    foods = sorted(list(set(y_treat)))

    # Combine datasets
    X = pd.concat([X_treat, X_control])
    y_control = "No" + y_control
    y = list(y_treat) + list(y_control)
    y = np.array(y)

    # Param grid to search for each food
    param_grid = {
        "n_estimators": [5000 if not fast_mode else 1000],
        "oob_score": [True],
        "n_jobs": [-1],
        "random_state": [1],
        "max_features": [None, "sqrt", "log2"],
        # "max_features": ["sqrt", "log2"],
        "min_samples_leaf": [1, 3, 5],
    }

    # per-food best features
    best_features_per_food = {}

    for food in foods:
        print(f"------- {title} - {food} -------")

        # make directory
        fdir = f"{food}"
        os.makedirs(fdir, exist_ok=True)

        # Extract labels/data for this food
        idx = [(food in l) for l in y]
        X_food = X.loc[idx]
        y_food = y[idx]
        print(y_food)

        # Naively assume the output classes are in alphabetical order. if something breaks, look here!
        classes = sorted(list(set(y_food)))

        # Grid search
        best_rf = None
        best_params = None
        for params in ParameterGrid(param_grid):
            rfc = RandomForestClassifier()
            rfc.set_params(**params)

            # Perform LOO evaluation for this parameter set
            cv_result = cross_validate(
                rfc,
                X_food.values,
                y_food,
                scoring=None,
                cv=LeaveOneOut(),
                n_jobs=-1,
                return_estimator=True,
            )

            # Update the best parameters
            estimators = cv_result["estimator"]
            for estimator in estimators:
                if best_rf is None or estimator.oob_score_ > best_rf.oob_score_:
                    best_rf = estimator
                    best_params = params

            # early exit
            if fast_mode:
                break

        print(
            f"Best parameters for {food} single-food model were {best_params}. Fast mode is {('en' if fast_mode else 'dis') + 'abled'}"
        )

        # Cross-val predict probabilities using leave one out and our new best parameters
        rfc = RandomForestClassifier()
        rfc.set_params(**best_params)
        y_proba = cross_val_predict(
            rfc,
            X_food.values,
            y_food,
            cv=LeaveOneOut(),
            n_jobs=-1,
            method="predict_proba",
        )

        # Convert probs to class scores
        y_pred = [classes[score.argmax()] for score in y_proba]

        # Try to compute ROC AUC if possible
        roc_auc = None
        try:
            roc_auc = roc_auc_score(y_food, y_proba[:, 1])
        except Exception as e:
            print(e)
            print("Couldn't compute ROC AUC score!")

        # Plot results
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_confusion_matrix(y_food, y_pred, ax=ax)
        plt.tight_layout()
        plt.title(f"{food} {title} Treatment")
        # plt.show()
        print(classification_report(y_food, y_pred))
        if roc_auc:
            print(f"{food} ROC AUC = {roc_auc}")

        # Plot feature importance graph
        best_feature_idxs = np.argsort(best_rf.feature_importances_)[::-1]
        plt.figure(figsize=(5, 5))
        plt.title(f"{food} Feature Importances")
        plt.xlabel("Feature #")
        plt.ylabel("Importance")
        plt.plot(sorted(best_rf.feature_importances_, reverse=True))
        # plt.show()
        best_features = X_food.columns[best_feature_idxs[:10]]
        print(best_features)

        # feature means per group write-out
        X_food_gb = X_food.copy().iloc[:, best_feature_idxs]
        X_food_gb["group"] = list(map(lambda i: "Control" if "No" in i else i, y_food))
        X_food_gb.groupby("group").mean().to_csv(
            f"{fdir}/{title}-{food}-feature-means.csv"
        )
        X_food_gb.groupby("group").std().to_csv(
            f"{fdir}/{title}-{food}-feature-stds.csv"
        )

        # Feautre importances write out + figure generation
        best_features_list = list(
            zip(
                [X_food.columns[idx] for idx in best_feature_idxs],
                [best_rf.feature_importances_[idx] for idx in best_feature_idxs],
            )
        )
        best_features_per_food[food] = best_features_list
        with open(f"{fdir}/{title}-{food}-feature-importances.csv", "w") as f:
            w = csv.writer(f)
            w.writerow(["feature", "importance"])
            for idx in best_feature_idxs:
                w.writerow([X_food.columns[idx], best_rf.feature_importances_[idx]])

        # plot features
        for feature in best_features:
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.boxplot(
                data=X_food_gb, x="group", y=feature, linewidth=2.5, width=0.4, ax=ax
            )
            ax.set_xlabel("Treatment group")
            ax.set_ylabel("Relative concentration")
            fig.suptitle(feature, size=22)
            fig.savefig(
                f"{fdir}/{title}-{food}-{feature}-boxplot.png", bbox_inches="tight"
            )
            plt.close(fig)

    return best_features_per_food


# # Full Pipeline

# In[13]:


def pipeline(single_foods=None, multi_foods=None, fast_mode=False):

    if fast_mode:
        print("Warning: Fast mode is enabled!")

    # 1. Load Data
    metadata, met_baseline, met_end = load_data(foods=single_foods, split_grains=True)

    # 2. Drop missing values
    met_baseline, met_end = drop_missing_values(met_baseline, met_end, metadata, p=0.2)

    # 3. Impute missing values
    met_baseline, met_end = impute_missing_values(met_baseline, met_end, mm=0.25)

    # 4. Decompose dataset
    (
        met_baseline,
        met_end,
        met_baseline_cont,
        met_baseline_treat,
        met_end_cont,
        met_end_treat,
        met_treatments,
        met_studies,
    ) = subset_separate_data(met_baseline, met_end)

    # 5. Subtract datasets
    met_diff_treat = subtract_data(met_baseline_treat, met_end_treat)
    met_diff_cont = subtract_data(met_baseline_cont, met_end_cont)

    # 6. Plot PCA plots
    # plot_pca(met_baseline_treat, met_treatments, "Treatment Baseline")
    # plot_pca(met_baseline, met_baseline.index.map(lambda i: i.split(".")[-1]), "All Baseline")
    # plot_pca(met_diff_treat, met_treatments, "Treatment Difference")
    # plot_pca(met_diff_cont, met_studies, "Control Difference")
    # plot_pca(met_end_treat, met_treatments, "Treatment End")

    # 7. Remove batch effect
    # met_diff_cont_nc, met_diff_treat_nc = remove_batch_effect(
    #    met_diff_cont, met_studies, met_diff_treat, met_treatments
    # )

    # plot_pca(
    #    met_diff_treat_nc, met_treatments, "Treatment Difference - Batch Effect Removed"
    # )

    # 6. Panel plots for specific foods
    if not fast_mode:
        panel_plots(
            met_diff_treat,
            met_treatments,
            met_diff_cont,
            met_studies,
            "Almond",
            "Control",
            ["C18:1 (9)", "C18:2 (9,12)"],
            mapping={
                "C18:1 (9)": "10-hydroxystearic acid",
                "C18:2 (9,12)": "Linoleic acid",
                "Tocopherol, a": r"α-tocopherol",
            },
        )

        panel_plots(
            met_diff_treat,
            met_treatments,
            met_diff_cont,
            met_studies,
            "Walnut",
            "Control",
            ["5-hydroxyindole-3-acetic acid", "URIC ACID"],
            mapping={
                "5-hydroxyindole-3-acetic acid": "5-HIAA",
                "URIC ACID": "Uric acid",
            },
        )

        panel_plots(
            met_diff_treat,
            met_treatments,
            met_diff_cont,
            met_studies,
            "Almond",
            "Walnut",
            ["5-hydroxyindole-3-acetic acid", "Tocopherol, a"],
            mapping={
                "5-hydroxyindole-3-acetic acid": "5-HIAA",
                "Tocopherol, a": "α-tocopherol",
            },
        )

    # 8. Per-food models
    best_features_per_food = per_food_models(
        met_diff_treat,
        met_treatments,
        "Difference",
        X_control=met_diff_cont,
        y_control=met_studies,
        fast_mode=fast_mode,
    )

    # 9. Multi-food models classification
    if multi_foods is not None:
        met_diff_treat_subset = met_diff_treat[
            [t in multi_foods for t in met_treatments]
        ]
        met_treatments_subset = [t for t in met_treatments if t in multi_foods]
        met_diff_cont_subset = met_diff_cont[[t in multi_foods for t in met_studies]]
        met_studies_subset = [t for t in met_studies if t in multi_foods]
    else:
        met_diff_treat_subset = met_diff_treat.copy()
        met_treatments_subset = met_treatments
        met_diff_cont_subset = met_diff_cont.copy()
        met_studies_subset = met_studies

    # Difference
    _, best_features_multi = classification(
        met_diff_treat_subset,
        met_treatments_subset,
        "Difference",
        X_control=met_diff_cont_subset,
        y_control=met_studies_subset,
        fast_mode=fast_mode,
    )

    return (
        met_diff_treat,
        met_treatments,
        met_diff_cont,
        met_studies,
        best_features_per_food,
        best_features_multi,
    )


# In[20]:


(
    met_diff_treat,
    met_treatments,
    met_diff_cont,
    met_studies,
    best_features_per_food,
    best_features_multi,
) = pipeline(
    single_foods=None,
    multi_foods=["Almond", "Walnut"],
    fast_mode=True,
)


# In[15]:


def correlate_features(
    met_diff_treat,
    met_treatments,
    met_diff_cont,
    met_studies,
    best_features_per_food,
    best_features_multi,
    show_all=False,
):
    from kneed import KneeLocator

    sns.set_theme()
    # TODO  fix filtering so we don't have to do this, and can just run Almond|Walnut single food models with the same feature set
    foods = sorted(list(best_features_per_food.keys()))
    foods = ["Almond", "Walnut"]

    # Plot feature importance graphs for selection
    fig, axs = plt.subplots(
        nrows=1, ncols=len(foods) + 1, figsize=(15, 5), sharex=False, sharey=False
    )
    axs = axs.flat
    for food, ax in zip(foods, axs):
        feature_imps = list(map(lambda i: i[1], best_features_per_food[food]))
        kneedle = KneeLocator(
            np.arange(len(feature_imps)),
            feature_imps,
            S=1.0,
            curve="convex",
            direction="decreasing",
        )
        elbow = kneedle.elbow

        ax.xaxis.get_major_locator().set_params(integer=True)
        sns.lineplot(data=feature_imps[: elbow * 2], ax=ax)
        ax.axvline(x=elbow, linestyle="dotted", color="grey")
        ax.set_title(food)

    feature_imps = list(map(lambda i: i[1], best_features_multi))
    kneedle = KneeLocator(
        np.arange(len(feature_imps)),
        feature_imps,
        S=1.0,
        curve="convex",
        direction="decreasing",
    )
    elbow = kneedle.elbow
    sns.lineplot(data=feature_imps[: elbow * 2], ax=axs[-1])
    axs[-1].axvline(x=elbow, linestyle="dotted", color="grey")
    axs[-1].set_title("Multi-food")
    axs[-1].xaxis.get_major_locator().set_params(integer=True)

    fig.show()

    # Plot correlation between single-food features and multi-food features
    # What data to look at? All of it?

    # Get correlation matrix

    met_diff_treat_foods = met_diff_treat.filter(regex="|".join(foods), axis=0)
    met_diff_cont_foods = met_diff_cont.filter(regex="|".join(foods), axis=0)
    met_diff_foods = pd.concat([met_diff_treat_foods, met_diff_cont_foods])
    corr = met_diff_foods.corr()

    # Rows = single food features
    # Columns = multi food features
    corr_rows_per_food = {}
    for food in foods:
        if food == "Almond":
            cutoff = 5
        elif food == "Walnut":
            cutoff = 5
        if show_all:
            cutoff = None
        top_features = list(map(lambda i: i[0], best_features_per_food[food][:cutoff]))
        if show_all:
            print(f"{food} has {len(best_features_per_food[food])} features")
        corr_rows_per_food[food] = corr.filter(items=top_features, axis=0)

    top_features_names_multi = list(map(lambda i: i[0], best_features_multi))[
        : 2 if not show_all else None
    ]
    corr2 = pd.concat(list(corr_rows_per_food.values()))
    corr2 = corr2.filter(items=top_features_names_multi)
    corr2.to_csv(f"correlation-{','.join(foods)}-vs-{'-'.join(foods)}.csv")

    mask = np.triu(np.ones_like(corr2, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr2,
        cmap=cmap,
        # vmax=1,
        # mask=mask,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )
    ax.set_xlabel("Multi-food features")
    ax.set_ylabel("Single-food features")

    fig.show()

    return corr2


# In[21]:


correlate_features(
    met_diff_treat,
    met_treatments,
    met_diff_cont,
    met_studies,
    best_features_per_food,
    best_features_multi,
    show_all=True,
)


# In[ ]:
