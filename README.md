# Binary Classification of Diabetes dataset
This repository involves an assignment for the course "Machine Learning in Computational Biology"of the MSc program in ["Data Science & Information Tachnologies"](https://dsit.di.uoa.gr) and the specialization of "Bioinformatics - Biomedical Data Science" offered by the National and Kapodistrian University of Athens.

## Summary
Diabetes is a prevalent chronic disease with significant health implications, making early diagnosis crucial for effective management and treatment.
This study focuses on employing machine learning classifiers for the binary classification of diabetes based on medical features.
Through a nested cross-validation approach, hyperparameters of various classifiers are optimized for improved model performance.
Additionally, preprocessing steps and class balancing techniques are explored to assess their impact on classification.
The study finds that Logistic Regression is the best performing classifier, demonstrating resilience against outliers and robustnessacross different datasets and balancing methods.
This research contributes to the field by providing insights into the effectiveness of machine learning techniques for diabetes classification.

# Repository structure
- data: contains the diabetes dataset used
- models: best model from analysis
- notebooks: jupyter notebooks for EDA, nested CV implementation, final model training, and pipeline for final model testingss
- plots: all generated plots from the analysis
- src: contains source code for the `class` object of nested-CV and secondary script for custom functions used throught the different stages of this project

More details on the analysis of the project are described on the [technical report](Glykeria_Spyrou_Report_MLCB_Assignment2.pdf).
