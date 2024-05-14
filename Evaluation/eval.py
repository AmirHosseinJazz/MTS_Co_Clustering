import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tslearn.metrics import dtw_path_from_metric, dtw, cdist_dtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import seaborn as sns
import pandas as pd
import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")
from util import load_data

#Method 1: Marginal Distribution Comparison
def marginal_distribution_comparison(real_data, generated_data,model_name,feature_names):
    num_features = real_data.shape[2]
    ks_stats=[]
    for feature_index in range(num_features):
        if os.path.exists(f'./marginal_dist/{model_name}') == False:
            os.makedirs(f'./marginal_dist/{model_name}')
        plt.figure(figsize=(10, 6))
        
        # Flatten the data for the current feature
        real_flattened = real_data[:, :, feature_index].flatten()
        generated_flattened = generated_data[:, :, feature_index].flatten()
        
        # KDE plot
        sns.kdeplot(real_flattened, label='Real', color='blue', bw_adjust=0.5)
        sns.kdeplot(generated_flattened, label='Generated', color='red', bw_adjust=0.5)
        
        plt.title(f'Distribution Comparison for Feature {feature_names[feature_index]}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f'./marginal_dist/{model_name}/feature_{feature_names[feature_index]}_distribution_comparison.png')

        # Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(real_flattened, generated_flattened)
        ks_stats.append({'feature_index': feature_names[feature_index], 'ks_stat': ks_stat, 'p_value': p_value})
        pd.DataFrame(ks_stats).to_csv(f'./marginal_dist/{model_name}/ks_stats.csv', index=False)

#Method 2: Multivariate Distribution Comparison
def multivariate_distribution_comparison(real_data, generated_data,model_name,feature_names,principal_components=2):
    if os.path.exists(f'./dist_compare/{model_name}') == False:
        os.makedirs(f'./dist_compare/{model_name}')
    num_features = real_data.shape[2]

    # Reshape data for comparison
    real_data_reshaped = real_data.reshape(-1, num_features)
    generated_data_reshaped = generated_data.reshape(-1, num_features)

    # 1. Pairwise Correlation Matrix Comparison
    print("Pairwise Correlation Matrix Comparison")
    real_corr = np.corrcoef(real_data_reshaped, rowvar=False)
    generated_corr = np.corrcoef(generated_data_reshaped, rowvar=False)
    plt.figure(figsize=(36, 25))
    plt.subplot(1, 2, 1)
    sns.heatmap(real_corr, cmap='coolwarm', vmin=-1, vmax=1, annot=True,
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Real Data Correlation Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Plot for generated data correlation matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(generated_corr, cmap='coolwarm', vmin=-1, vmax=1, annot=True,
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Generated Data Correlation Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f'./dist_compare/{model_name}/correlation_matrix_comparison.png')

    # 2. Principal Component Analysis (PCA) Visualization
    print("Principal Component Analysis (PCA) Visualization")
    pca = PCA(n_components=principal_components)
    real_data_pca = pca.fit_transform(real_data_reshaped)
    generated_data_pca = pca.transform(generated_data_reshaped)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(real_data_pca[:, 0], real_data_pca[:, 1], alpha=0.5, label='Real')
    plt.scatter(generated_data_pca[:, 0], generated_data_pca[:, 1], alpha=0.5, label='Generated')
    plt.title('PCA of Real and Generated Data')
    plt.legend()
    plt.savefig(f'./dist_compare/{model_name}/pca_comparison.png')

    # 3. t-SNE Visualization
    print("t-SNE Visualization")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    real_data_tsne = tsne.fit_transform(real_data_reshaped)
    generated_data_tsne = tsne.fit_transform(generated_data_reshaped)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(real_data_tsne[:, 0], real_data_tsne[:, 1], alpha=0.5, label='Real')
    plt.scatter(generated_data_tsne[:, 0], generated_data_tsne[:, 1], alpha=0.5, label='Generated')
    plt.title('t-SNE of Real and Generated Data')
    plt.legend()
    plt.savefig(f'./dist_compare/{model_name}/tsne_comparison.png')

    # 4. Multivariate Gaussian Distribution Comparison
    print("Multivariate Gaussian Distribution Comparison")
    real_mean = np.mean(real_data_reshaped, axis=0)
    real_cov = np.cov(real_data_reshaped, rowvar=False)
    
    generated_mean = np.mean(generated_data_reshaped, axis=0)
    generated_cov = np.cov(generated_data_reshaped, rowvar=False)
    
    # print(f'Real Data Mean: {real_mean}')
    # print(f'Generated Data Mean: {generated_mean}')
    
    # print(f'Real Data Covariance Matrix:\n{real_cov}')
    # print(f'Generated Data Covariance Matrix:\n{generated_cov}')
    
    # Optional: Visualize the difference in means and covariances
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(real_mean)), real_mean, alpha=0.5, label='Real')
    plt.bar(range(len(generated_mean)), generated_mean, alpha=0.5, label='Generated')
    plt.title('Mean Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(real_cov.flatten())), real_cov.flatten(), alpha=0.5, label='Real')
    plt.bar(range(len(generated_cov.flatten())), generated_cov.flatten(), alpha=0.5, label='Generated')
    plt.title('Covariance Matrix Comparison')
    plt.legend()
    plt.savefig(f'./dist_compare/{model_name}/mean_cov_comparison.png')

#Method 3: Temporal Dynamics Comparison
def temporal_dynamics_comparison(real_data, generated_data,model_name,feature_names, max_lag=40, ):
    if os.path.exists(f'./temporal_dynamics/{model_name}') == False:
        os.makedirs(f'./temporal_dynamics/{model_name}')
    num_features = real_data.shape[2]

    for feature_index in range(num_features):
        # Extract the time series for the current feature across all cases
        real_series = real_data[:, :, feature_index].flatten()
        generated_series = generated_data[:, :, feature_index].flatten()
        
        plt.figure(figsize=(14, 6))
        
        # ACF for real data
        plt.subplot(2, 2, 1)
        plot_acf(real_series, lags=max_lag, ax=plt.gca(), title=f'ACF - Real Data (Feature {feature_names[feature_index]})')
        
        # ACF for generated data
        plt.subplot(2, 2, 2)
        plot_acf(generated_series, lags=max_lag, ax=plt.gca(), title=f'ACF - Generated Data (Feature {feature_names[feature_index]})')
        
        # PACF for real data
        plt.subplot(2, 2, 3)
        plot_pacf(real_series, lags=max_lag, ax=plt.gca(), title=f'PACF - Real Data (Feature {feature_names[feature_index]})')
        
        # PACF for generated data
        plt.subplot(2, 2, 4)
        plot_pacf(generated_series, lags=max_lag, ax=plt.gca(), title=f'PACF - Generated Data (Feature {feature_names[feature_index]})')
        
        plt.tight_layout()
        plt.savefig(f'./temporal_dynamics/{model_name}/temporal_dynamics_feature_{feature_names[feature_index]}.png')

def dwt(real_data, generated_data,model_name,):
    if os.path.exists(f'./sequence_level/{model_name}') == False:
        os.makedirs(f'./sequence_level/{model_name}')
    num_cases = real_data.shape[0]
    dtw_distances = []

    for i in range(num_cases):
        real_sequence = real_data[i]
        generated_sequence = generated_data[i % generated_data.shape[0]]  # Handle different lengths
        
        distance, _ = fastdtw(real_sequence, generated_sequence, dist=euclidean)
        dtw_distances.append(distance)
    plt.figure(figsize=(10, 6))
    sns.histplot(dtw_distances, kde=True, color='blue')
    plt.title('Distribution of DTW Distances')
    plt.xlabel('DTW Distance')
    plt.ylabel('Frequency')
    plt.savefig(f'./sequence_level/{model_name}/dtw_distances.png')
    pd.DataFrame(dtw_distances).to_csv(f'./sequence_level/{model_name}/dtw_distances.csv', index=False)
    pd.DataFrame([np.mean(dtw_distances), np.median(dtw_distances), np.std(dtw_distances)], index=['Mean', 'Median', 'Standard Deviation'], columns=['Value']).to_csv(f'./sequence_level/{model_name}/dtw_stats.csv')

def discrete_frechet_dist(P, Q):
    n = len(P)
    m = len(Q)
    ca = np.full((n, m), -1.0)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        if i == 0 and j == 0:
            ca[i, j] = euclidean(P[0], Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), euclidean(P[i], Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), euclidean(P[0], Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), euclidean(P[i], Q[j]))
        else:
            ca[i, j] = float('inf')
        return ca[i, j]

    return c(n - 1, m - 1)

def shape_based_distances(real_data, generated_data,model_name):
    num_cases = real_data.shape[0]
    euclidean_distances = []
    frechet_distances = []
    sbd_distances = []

    for i in range(num_cases):
        real_sequence = real_data[i]
        generated_sequence = generated_data[i % generated_data.shape[0]]  # Handle different lengths

        # Compute Euclidean Distance
        euclidean_dist = np.linalg.norm(real_sequence - generated_sequence)
        euclidean_distances.append(euclidean_dist)

        # Compute Frechet Distance
        frechet_dist = discrete_frechet_dist(real_sequence, generated_sequence)
        frechet_distances.append(frechet_dist)

        # Compute Shape-Based Distance (SBD)
        sbd_dist, _ = dtw_path_from_metric(real_sequence, generated_sequence, metric="sqeuclidean")
        sbd_distances.append(sbd_dist)

    plt.figure(figsize=(18, 6))
    
    # Euclidean Distance
    plt.subplot(1, 3, 1)
    sns.histplot(euclidean_distances, kde=True, color='blue')
    plt.title('Distribution of Euclidean Distances')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    
    # Frechet Distance
    plt.subplot(1, 3, 2)
    sns.histplot(frechet_distances, kde=True, color='green')
    plt.title('Distribution of Frechet Distances')
    plt.xlabel('Frechet Distance')
    plt.ylabel('Frequency')
    
    # Shape-Based Distance (SBD)
    plt.subplot(1, 3, 3)
    sns.histplot(sbd_distances, kde=True, color='red')
    plt.title('Distribution of Shape-Based Distances')
    plt.xlabel('Shape-Based Distance')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'./sequence_level/{model_name}/shape_based_distances.png')
    distances=[]
    distances.append({'euclidean_mean': np.mean(euclidean_distances), 'euclidean_median': np.median(euclidean_distances), 'euclidean_std': np.std(euclidean_distances),})
    distances.append({'frechet_mean': np.mean(frechet_distances), 'frechet_median': np.median(frechet_distances), 'frechet_std': np.std(frechet_distances),})
    distances.append({'sbd_mean': np.mean(sbd_distances), 'sbd_median': np.median(sbd_distances), 'sbd_std': np.std(sbd_distances),})
    pd.DataFrame(distances).to_csv(f'./sequence_level/{model_name}/shape_based_distances_stats.csv', index=False)

# def regression_metrics(real_data, generated_data,model_name):
#     if os.path.exists(f'./predictive_performance/{model_name}') == False:
#         os.makedirs(f'./predictive_performance/{model_name}')
#     num_features = real_data.shape[2]
#     metrics = []

if __name__=="__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Evaluate marginal distribution comparison')
    parser.add_argument('--model_name', type=str, default='TimeGAN_model1', help='Name of the model')
    parser.add_argument('--method', type=str, help='Evaluation method')
    args = parser.parse_args()
    print(f"model_name: {args.model_name}")
    params = {}
    if 'GAN' in args.model_name:

        try:
            with open(f'../TGAN/saved_models/{args.model_name}/config.txt', 'r') as f:
                for line in f:
                    key, value = line.split(':')
                    # check if the parameter is an integer
                    try:
                        params[key.strip()] = int(value.strip())
                    except:
                        params[key.strip()] = value.strip()
        except Exception as E:
            raise Exception(f"Could not load the configuration file: {E}")
    elif 'VAE' in args.model_name:
        try:
            with open(f'../TVAE/saved_models/{args.model_name}/config.txt', 'r') as f:
                for line in f:
                    key, value = line.split(':')
                    # check if the parameter is an integer
                    try:
                        params[key.strip()] = int(value.strip())
                    except:
                        params[key.strip()] = value.strip()
        except Exception as E:
            raise Exception(f"Could not load the configuration file: {E}")

    np.random.seed(42)  # For reproducibility
    #### Shape (num_samples, num_features, num_timesteps)
    if params['input_dim'] == 12:
        real_data,feature_names=load_data('../Data/PreProcessed/12var/df12.xlsx')
    elif params['input_dim'] == 29:
        real_data,feature_names=load_data('../Data/PreProcessed/29var/df29.xlsx')
    #read numpy from npz file

    generated_data=np.load(f'../Generated/{args.model_name}/generated_samples.npy')
    # print('Shape of generated_data:', generated_data.shape)
    # print('Sample', generated_data[0])
    # print(feature_names)

    # randomly pick 100 samples from real_data
    real_data = real_data[np.random.choice(real_data.shape[0], 100, replace=False)]
    # print('randomly picked samples from real_data to match the dimension of generated_data')
    # print('Shape of real_data:', real_data.shape)
    # print('Sample', real_data[0])


    if args.method == 'marginal_dist':
        print("Method: Marginal Distribution Comparison")
        print("Results are saved in the folder 'marginal_dist' and subfolder with model name")
        marginal_distribution_comparison(real_data, generated_data, args.model_name,feature_names)
    if args.method == 'dist_compare':
        multivariate_distribution_comparison(real_data, generated_data, args.model_name,feature_names)
        print("Method: Multivariate Distribution Comparison")
        print("Results are saved in the folder 'dist_compare' and subfolder with model name")
    if args.method == "temporal":
        temporal_dynamics_comparison(real_data, generated_data, args.model_name,feature_names)
        print("Method: Temporal Dynamics Comparison")
        print("Results are saved in the folder 'temporal_dynamics' and subfolder with model name")
    if args.method == "sequence_level":
        print("Method1: Dynamic Time Warping (DTW) Comparison")
        dwt(real_data, generated_data, args.model_name)
        print("Method2: Shape-Based Distances Comparison")
        shape_based_distances(real_data, generated_data, args.model_name)
        print("Results are saved in the folder 'sequence_level' and subfolder with model name")
    
    if args.method =="all":
        print("Method: Marginal Distribution Comparison")
        print("Results are saved in the folder 'marginal_dist' and subfolder with model name")
        marginal_distribution_comparison(real_data, generated_data, args.model_name,feature_names)
        print("Method: Multivariate Distribution Comparison")
        print("Results are saved in the folder 'dist_compare' and subfolder with model name")
        multivariate_distribution_comparison(real_data, generated_data, args.model_name,feature_names)
        print("Method: Temporal Dynamics Comparison")
        print("Results are saved in the folder 'temporal_dynamics' and subfolder with model name")
        temporal_dynamics_comparison(real_data, generated_data, args.model_name,feature_names)
        print("Method1: Dynamic Time Warping (DTW) Comparison")
        dwt(real_data, generated_data, args.model_name)
        print("Method2: Shape-Based Distances Comparison")
        shape_based_distances(real_data, generated_data, args.model_name)
        print("Results are saved in the folder 'sequence_level' and subfolder with model name")
    # if args.method =="predictive":
    #     print("Method: Predictive Performance Comparison")
    #     print("Results are saved in the folder 'predictive_performance' and subfolder with model name")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")