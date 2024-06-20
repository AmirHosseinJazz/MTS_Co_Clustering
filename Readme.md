# Title : Thesis project on co-clustering of multivariate time series

## Introduction

Welcome to the repository for my thesis project on co-clustering of multivariate time series! In this project, I explore the application of co-clustering techniques to analyze and group multivariate time series data.

## Motivation

The analysis of multivariate time series data is a challenging task due to the complex dependencies and correlations between variables. Traditional clustering methods may not be suitable for this type of data, as they often overlook the temporal aspect and fail to capture the underlying patterns.

Co-clustering, also known as biclustering, is a powerful technique that simultaneously clusters both the rows (time series) and columns (variables) of a dataset. By considering the joint clustering of time series and variables, co-clustering can reveal hidden patterns and structures that are not apparent when analyzing each dimension separately.

## Objectives

The main objectives of this thesis project are:

1. Implement and evaluate different co-clustering algorithms for multivariate time series data.
2. Compare the performance of co-clustering methods with traditional clustering approaches.
3. Apply co-clustering techniques to real-world datasets and analyze the discovered patterns.
4. Investigate the interpretability and usefulness of the co-clustering results for time series analysis tasks.

## Repository Structure

This repository is organized as follows: ## Repository Structure

    - `Clustering/`: Containst the traditional clustering methods
        - grid search files implemented a full grid search on hyperparameters for implemented models
        - complex network approach: Implementing complex networks on preprocessed data
        - embedded_complex_network: Implementing complex networks on latent representations
        - embedded_clustering: Implementing traditional clustering approaches on latent represntations
        - complex_network_eval: aggregating the results from the three  approaches and output in 'clustering_evaluation.csv' file
    - `Co-Clustering/`: Contains bi-clustering approaches
    - `CCTN/`: Contains the CCTN network [Coupled clustering of time-series and Networks by Liu et al]
    - `TGAN/`: Contains the trained models for both the TGAN model creation, preprocessing and artifacts.
    - `TVAE/`: Contains the trained models for both the TVAE model creation, preprocessing and artifacts.
    - `wandb/`: logs for TVAE and TGAN models
    - `Evaluation/`: Contains the evaluation results and generated synthetic data.
    - `Generated/`: Contains the generated latent representations for each of TVAE and TGAN models
    - `Data/`: Contains the data from the real-world application (due to privacy concerns this will be removed later on)
    - `README.md`: The main documentation file providing an overview of the project and instructions for getting started.
    Please note that the `tvae` and `tgan` folders are two different approaches to create synthetic data. In this project, their encoder and generator components have been used to create a summarized encoding of the multivariate time series.

## Conclusion

By applying co-clustering techniques to multivariate time series data, this thesis project aims to contribute to the field of time series analysis and provide insights into the underlying structures and patterns of complex datasets. I hope that this research will be valuable for researchers and practitioners working in various domains, such as finance, healthcare, and environmental monitoring.

Feel free to explore the repository and reach out if you have any questions or suggestions!
