import shap
import numpy as np
import warnings;warnings.filterwarnings("ignore")

def _get_model_name(model):
    """
        Returns a string with the name of a sklearn model
            model: Sklearn stimator class
    """
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
        name = "Pipeline_" + str(estimator)[:str(estimator).find("(")]
    else: 
        name = str(model)[:str(model).find("(")]
    return name 

def plot_cv_score(X, y, models_list, cv = 5, scoring = None, refit = True, verbose = True):
    """ 
        X: numpy_array/pandas dataframe n_rows, m_features
        y: numpy_array/pandas dataframe n_rows
        Plots min, max and avg kfold crosval_score for a list of models
    
    """

    
    
    names, scores, min_score, max_score, mean_score = list(), list(), list(), list(), list()

    for i, model in enumerate(models_list):
        t0 = time.time()
        name = _get_model_name(model)
        names.append(name)

        if refit:
            model.fit(X, y)
        
        score = cross_val_score(model, X, y, cv = cv, scoring = scoring, n_jobs= -1)

        min_score.append(np.min(score))
        max_score.append(np.max(score))
        mean_score.append(np.mean(score))
        scores.append(score)
        t1 = time.time()
        
        if verbose:
            print(f"Iteration: {i} done in {round((t1-t0)/60,2)} minutes")
            print(f"Mean score for model: {names[i]}: {mean_score[i]}")
        
            
    
    frame_summary = pd.DataFrame({'Min':min_score, 'Average': mean_score, 'Max': max_score,}, index = names).sort_values(by = 'Average')

    frame_scores = pd.DataFrame(np.vstack(scores).T, columns = names) 


    fig, ax  = plt.subplots(1,2, figsize = (15,7))

    frame_summary.plot.barh(edgecolor = 'black', ax = ax[0], cmap = 'RdYlBu')
    ax[0].legend(loc = 'best')
    ax[0].set_xlabel("Score")

    frame_scores.boxplot(ax = ax[1])
    ax[1].set_title("Model scores distribution")
    ax[1].set_ylabel("Score")
    ax[1].tick_params(labelrotation=90)
    
    
def plot_importances(estimator, X, y, scoring = None, n_repeats = 5, plot_results = True, n_jobs = -1):
    """
    Computes permutation feature importance for a given model
    """
    pimp = permutation_importance(estimator= estimator, X= X, y = y, n_repeats= n_repeats, n_jobs = n_jobs)
    
    df = pd.DataFrame({"Mean performance decrease":pimp.importances_mean}, index = X.columns).sort_values(by = "Mean performance decrease")
    
    if plot_results:
        fig, ax = plt.subplots(figsize = (10,5))

        df.plot.barh(ax = ax, edgecolor = "black", cmap = "RdYlBu")
        ax.set_title("Importances")
    else:
        return df


def plot_results(X, y, estimator, prefit = True, color_test_set = True):
    """
    """
    if color_test_set:
        color_positive = "darkgreen"
        color_negative = "darkred"
        label = "Test"
    else:
        color_positive = "steelblue"
        color_negative = "orange"
        label = "Train"
    if not prefit:
        estimator.fit(X,y)
    y_pred = estimator.predict_proba(X)[:,1]
    fpr,tpr,_ = roc_curve(y_true = y, y_score = y_pred)
    
    fig, ax = plt.subplots(1,3, figsize = (17,5))
    
    ax[0].hist(y_pred[y == 0], color = color_positive, alpha = .5, edgecolor = color_positive, bins = "auto", label = "positive class")
    ax[0].hist(y_pred[y == 1], color = color_negative, alpha = .5, edgecolor = color_negative, bins = "auto", label = "negative class")
    ax[0].set_title(f"Class distribution on [{label}]")
    ax[0].set_ylabel("Number of samples")
    ax[0].set_xlabel("Model probabilities")
    ax[0].legend()
    
    ax[1].plot(fpr, tpr, color = "darkred", label = "roc")
    ax[1].plot([0,1], [0,1], linestyle = "--", color = "black")
    ax[1].set_title(f"Area under roc curve\nScore: {round(roc_auc_score(y_true = y, y_score = y_pred),3)}")
    ax[1].set_xlabel("false positive rate")
    ax[1].set_ylabel("true positive rate")
    ax[1].legend()
    
    ks, p_value = ks_2samp(y_pred[y == 0], y_pred[y == 1])
    
    sns.distplot(y_pred[y==0], hist = False, kde_kws={"cumulative":True}, rug = False, color = color_positive, ax = ax[2], label = "positive class")
    sns.distplot(y_pred[y==1], hist = False, kde_kws={"cumulative":True}, rug = False, color = color_negative, ax = ax[2], label = "negative class")
    ax[2].set_title(f"Class separation\ks: {round(ks,3)}")
    
    plt.tight_layout()


def plot_shap(frame, features):
    """
    """
    import shap
    shap_values = np.vstack(frame.shap_values)
    shap.summary_plot(shap_values, frame[features],plot_size = [15,10],max_display = 30)
    
    