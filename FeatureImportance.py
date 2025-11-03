def plot_importance(importances, top_n=None,  figsize=(8,6)):
    # sorting with asc=false for correct order of bars
    if top_n==None:
        ## sort all features and set title
        plot_vals = importances.sort_values()
        title = "All Features - Ranked by Importance"
    else:
        ## sort features and keep top_n and set title
        plot_vals = importances.sort_values().tail(top_n)
        title = f"Top {top_n} Most Important Features"
    ## plotting top N importances
    ax = plot_vals.plot(kind='barh', figsize=figsize)
    ax.set(xlabel='Importance',
            ylabel='Feature Names',
            title=title)
    ## return ax in case want to continue to update/modify figure
    return ax

    def plot_importances(importances, top_n=None, head_or_tail='tail', figsize=(8, 6)):
    # Sort importances in ascending order
    importances_sorted = importances.sort_values(ascending=True)

    # Select top_n features from head or tail
    if top_n:
        if head_or_tail == 'head':
            selected = importances_sorted.head(top_n)
        else:
            selected = importances_sorted.tail(top_n)
    else:
        selected = importances_sorted

    # Plot
    ax = selected.plot(kind='barh', figsize=figsize)
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.show()

def get_importances(model, feature_names=None, sort=False, ascending=True):

    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    series = pd.Series(importances, index=feature_names, name="Feature Importance")

    if sort:
        series = series.sort_values(ascending=ascending)

    return series



# Example usage 

# from sklearn.inspection import permutation_importance

# r = permutation_importance(RandomF_defult_pipe, X_test, y_test,n_repeats =10, random_state = 42)

# permutation_importances = pd.Series(r['importances_mean'],index=X_test.columns, name = 'permutation importance')
# permutation_importances = permutation_importances.sort_values(ascending=False)
# permutation_importances


