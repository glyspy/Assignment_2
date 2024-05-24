import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from scipy import stats
import optuna
from optuna.integration import OptunaSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, average_precision_score

def dict_to_df(dictionary, model):
    """"
    Function for the conversion of a dictionary into a dataframe

    Input:
    - Dictionary to convert
    - Model (classifier) name (str)

    Output:
    - DataFrame
    """
    model_dict = dictionary[model]
    model_df = pd.DataFrame(model_dict)

    return model_df


##-------------------------------------------------------------------------------##
## BOXPLOTS FOR METRICS ##

def metric_box_plots(all_dfs, classifier_names, title, range_list):
    """"
    Function that returns a figure of subplots with boxplots of the metric scores
    per classifier

    Input:
    - List of DataFrames
    - List of classifier names
    - Title (str) for the plot
    - Value range for x axis

    Output:
    - Figure object
    """
    scores_fig = make_subplots(rows=len(all_dfs), cols=1, subplot_titles=classifier_names)

    # Iterate through each data frame
    for df_index, df in enumerate(all_dfs):
        # Iterate through each metric in the data frame
        for metric in df.columns:
            box_plot = go.Box(
                x=df[metric],
                orientation='h',  
                name=metric  
            )
        
            scores_fig.add_trace(box_plot, row=df_index + 1, col=1)

    scores_fig.update_layout(
        title=title,
        title_x=0.5,
        showlegend=False, 
        height=1200,
        width=450,
        template='simple_white',
        font=dict(family="Arial", size=10),
        margin=dict(l=10, r=10, t=80, b=30, pad=10)
    )

    for i in range(1, len(all_dfs) + 1):
        scores_fig.update_xaxes(range=range_list, row=i, col=1)
    
    return scores_fig


##-------------------------------------------------------------------------------##
## FEAUTURE BOXPLOTS ##

def boxplot_feats_plot(df, title):
    """"
    Function for the creation of the features subplots

    Input:
    - DataFrame object
    - Title (str) for the plot

    Output:
    - Figure object
    """
    feats_box = make_subplots(rows=1, cols=len(df.columns) - 1,
                              horizontal_spacing=0.07)

    for i, column in enumerate(df.columns[1:-1]):
        feats_box.add_trace(go.Box(y=df[column], name=column, showlegend=False), row=1, col=i + 1)

    feats_box.update_layout(title=title, title_x=0.5,
                            height=450,
                            template='simple_white',
                            margin=dict(l=140, r=0, t=80, b=70, pad=10),
                            font=dict(family="Arial", size=15))

    return feats_box


##-------------------------------------------------------------------------------##
## HEATMAP PLOT ##

def correlation_heatmap(df, title, drop_col):
    """"
    Function for that creates a heatmap out of the correlation coefficients of the features

    Input:
    - DataFrame object
    - Title (str) for the plot
    - Column names (str) to drop from the DF

    Output:
    - Figure object
    """
    corr_matrix = df.drop(columns=drop_col).corr()

    heatmap = go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        showscale=True,
        colorbar=dict(title='Correlation')
    )

    heatmap_fig = go.Figure(data=[heatmap])
    
    heatmap_fig.update_yaxes(tickangle=-40)
    heatmap_fig.update_xaxes(tickangle=-40)
    heatmap_fig.update_layout(
        title=title,
        title_x=0.5,
        template='simple_white',
        height=570,
        width=600,
        font=dict(family="Arial", size=15)
    )
    
    return heatmap_fig


##-------------------------------------------------------------------------------##
## PCA PLOT ##

def pca_plot(df, features_rm, target, class_names, title, n_components=2):
    """"
    Function for the creation of PCA plots

    Input:
    - DataFrame object
    - Features/column names (str) to remove
    - Target (y)
    - Class names (str)
    - Title (str) for the plot
    - Number of principal components for the analysis

    Output:
    - Figure object
    """

    X = df.drop(columns=features_rm)
    y = df[target]
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['target'] = y
    
    pca_fig = go.Figure()
    
    unique_classes = pca_df['target'].unique()
    colors = ['#2554a1', '#faba25'] 

    for idx, class_value in enumerate(unique_classes):
        class_df = pca_df[pca_df['target'] == class_value]
        pca_fig.add_trace(
            go.Scatter(
                x=class_df['PC1'],
                y=class_df['PC2'],
                mode='markers',
                name=class_names.get(class_value, str(class_value)), 
                marker=dict(
                    size=8,
                    color=colors[idx],
                    opacity=0.8
                ),
                text=class_df['target'],
                hoverinfo='text'
            )
        )
    
    pca_fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title='PC1',
        yaxis_title='PC2',
        template='simple_white',
        width=550,
        height=450,
        font=dict(family="Arial", size=15),
        legend=dict(title='Target')
    )
    
    return pca_fig


##-------------------------------------------------------------------------------##
## OUTLIER FUNCTION ##

def outlier_mask(df, threshold=1.5):
    """"
    Function for the creation of a filtering mask for outlier detection

    Input:
    - DataFrame object
    - Threshold for the detemination of the lower and upper bounds

    Output:
    - Mask object to further perform filtering based on the original (True=outlier)
    """
    mask_df = pd.DataFrame(index=df.index)
    
    for column in df.columns:
        # Calculate Q1, Q3, and IQR for the column
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate lower and upper bounds for the column
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        mask_df[column] = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return mask_df


##-------------------------------------------------------------------------------##
## CLASSIFIER EVALUATION ##

def evaluate_clfs(data_frames, classifier_names, priorities, top_n):
    """
    Evaluate the classifiers based on priority metrics and calculate confidence intervals.

    Input:
    - data_frames (list of pd.DataFrame): List of data frames with classifier metrics.
    - classifier_names (list of str): List of classifier names.
    - priorities (list of str): List of priority metrics.
    - top_n (int): Number of top classifiers to consider.

    Output:
    - str: Best classifier based on the highest priority score.
    - dict: Dictionary with confidence intervals for the top classifiers.
    - dict: Dictionary with widths of the confidence intervals for the top classifiers.
    - dict: Dictionary with median metrics for each classifier.
    """
    # Initialize dictionary for median metrics
    median_metrics = {}

    # Calculate the median of each metric in the data frame for each classifier
    for classifier, df in zip(classifier_names, data_frames):
        median_metrics[classifier] = df.median()

    # Function to calculate a priority score for a classifier
    def calculate_priority_score(median_metrics, priorities):
        return sum(median_metrics[metric] for metric in priorities)

    # Calculate priority scores for each classifier
    priority_scores = {classifier: calculate_priority_score(median, priorities)
                       for classifier, median in median_metrics.items()}

    # Print the overall score of all classifiers based on priority metrics
    print("Overall Scores of Classifiers based on Priority Metrics:")
    for classifier, score in priority_scores.items():
        print(f"{classifier}: {score}")

    # Find the classifier with the highest priority score
    best_classifier = max(priority_scores, key=priority_scores.get)

    # Determine the top classifiers based on the priority scores
    top_classifiers = [classifier for classifier, score in priority_scores.items()
                       if score >= sorted(priority_scores.values())[-top_n]]

    # Initialize dictionaries for confidence intervals and CI widths
    confidence_intervals = {}
    ci_widths = {}

    # Calculate confidence intervals for the top classifiers
    for classifier in top_classifiers:
        df = data_frames[classifier_names.index(classifier)]

        ci_dict = {}
        widths_dict = {}

        # Calculate 95% confidence interval and width for each metric
        for metric in df.columns:
            ci_low, ci_high = stats.t.interval(0.95, len(df[metric]) - 1, loc=np.mean(df[metric]), scale=stats.sem(df[metric]))
            ci_dict[metric] = (ci_low, ci_high)
            ci_width = ci_high - ci_low
            widths_dict[metric] = ci_width

        # Store confidence intervals and CI widths in dictionaries
        confidence_intervals[classifier] = ci_dict
        ci_widths[classifier] = widths_dict

    # Return the results
    return best_classifier, confidence_intervals, ci_widths, median_metrics


##-------------------------------------------------------------------------------##
## PLOT MEDIAN SCORES ##

def plot_median_scores(median_metrics, title):
    classifiers = list(median_metrics.keys())
    metrics = list(next(iter(median_metrics.values())).index)

    # Create a figure
    fig = go.Figure()

    # Define custom colors for each classifier
    custom_colors = {
        'Logistic Regression': '#9A8822',
        'Gaussian NB': '#F5CDB4',
        'k-NN': '#f59b93',
        'LDA': '#FDDDA0',
        'SVM': '#74A089'
    }

    # Add bars for each classifier
    for classifier in classifiers:
        # Get the median metrics for the classifier
        medians = median_metrics[classifier]
        
        # Add a bar trace for each metric
        fig.add_trace(go.Bar(
            x=metrics,
            y=medians,
            name=classifier,
            marker_color=custom_colors.get(classifier, 'rgb(150, 150, 150)')  # Default color if not specified
        ))

    # Update the layout of the plot
    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title='Metrics',
        yaxis_title='Median Score',
        barmode='group',  # Group the bars together
        template='simple_white',
        legend_title='Classifier',
        font=dict(family="Arial", size=14)
    )
    
    return fig


##-------------------------------------------------------------------------------##
## Data preprocess ##

def filter_zeros(data, cols=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']):
    """
    Filter out rows with zeros in specified columns.
    
    Parameters:
        data (pd.DataFrame): The raw data to preprocess.
        cols (list of str): List of column names to check for zeros.
    
    Returns:
        pd.DataFrame: The filtered data.
    """
    # Create a mask to identify rows with zeros in specified columns
    mask = (data[cols] == 0).any(axis=1)
    
    # Filter out rows with zeros
    data_filtered = data[~mask]
    
    # Remove the 'ID' column if it exists
    if 'ID' in data_filtered.columns:
        data_filtered = data_filtered.drop(columns=['ID'])
    
    return data_filtered

def remove_outliers(data, threshold=1.5):
    """
    Remove outliers using a custom threshold.
    
    Parameters:
        data (pd.DataFrame): The raw data to preprocess.
        threshold (float): The threshold for outlier detection.
    
    Returns:
        pd.DataFrame: The data with outliers removed.
    """
    # Example outlier detection using IQR
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    mask = (data < lower_bound) | (data > upper_bound)
    return data[~mask.any(axis=1)]

def replace_null(data):
    """
    Replace missing values with the mean based on the 'Outcome' column.

    Parameters:
        data (pd.DataFrame): The raw data to preprocess.

    Returns:
        pd.DataFrame: The data with missing values replaced with group means.
    """
    # Define the column name for the target variable
    target_col = 'Outcome'

    # Group the data by the 'Outcome' column and calculate the mean of each group
    group_means = data.groupby(target_col).mean()
    
    # Define a function to replace missing values in a row based on the group mean
    def fill_missing_values(row):
        outcome = row[target_col]
        # Fill missing values in the row with the mean of the corresponding group
        row_filled = row.fillna(group_means.loc[outcome])
        return row_filled
    
    # Apply the function to each row in the data
    data_filled = data.apply(fill_missing_values, axis=1)
    
    return data_filled

