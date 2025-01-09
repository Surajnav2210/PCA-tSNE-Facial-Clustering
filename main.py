from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def load_and_center_dataset(filename):
    x = np.load(filename)
    mux = np.mean(x, axis=0)
    x_centered = x - mux
    return x_centered

def get_covariance(dataset):
    samples = dataset.shape[0]
    datasetT = np.transpose(dataset)
    covariance_matrix = np.dot(datasetT, dataset) / (samples - 1)
    return covariance_matrix

def get_eig(S, m):
    eigen_values, eigen_vectors = eigh(S, subset_by_index=(S.shape[0] - m, S.shape[0] - 1))
    eigen_values = eigen_values[::-1]
    eigen_vectors = eigen_vectors[:, ::-1]
    return np.diag(eigen_values), eigen_vectors

def get_eig_prop(S, prop):
    eigen_values, eigen_vectors = eigh(S)
    eigen_values = eigen_values[::-1]
    total_variance = np.sum(eigen_values)
    explained_variance = np.cumsum(eigen_values) / total_variance
    num_selected = np.searchsorted(explained_variance, prop) + 1
    Lambda, U = get_eig(S, num_selected)
    return Lambda, U

def project_image(image, U):
    U_transpose = np.transpose(U)
    alpha = np.dot(U_transpose, image)
    reconstruction = np.dot(U, alpha)
    return reconstruction

def display_image(orig, proj):
    orig = orig.reshape(32, 32).T
    proj = proj.reshape(32, 32).T

    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3), ncols=2)
    ax1.set_title("Original")
    ax2.set_title("Projection")

    img1 = ax1.imshow(orig, aspect='equal', origin='upper')
    img2 = ax2.imshow(proj, aspect='equal', origin='upper')

    plt.colorbar(img1, ax=ax1)
    plt.colorbar(img2, ax=ax2)
    return fig, ax1, ax2

def plot_scree(eigen_values):
    explained_variance_ratio = eigen_values / np.sum(eigen_values)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.title('Scree Plot')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.pause(1)

def tsne_visualization(data, labels=None):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=10)
    plt.title('t-SNE Visualization')
    if labels is not None:
        plt.colorbar(scatter)
    plt.show()

def compute_reconstruction_quality(original, reconstructed):
    original = (original - original.min()) / (original.max() - original.min())
    reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())

    psnr_value = psnr(original, reconstructed, data_range=1.0)
    ssim_value = ssim(original.reshape(32, 32), reconstructed.reshape(32, 32), data_range=1.0)

    return psnr_value, ssim_value

def perform_clustering(data, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, labels)
    print(f"Silhouette Score for clustering: {silhouette_avg:.2f}")
    return labels

def main():
    print("Starting PCA...")
    x = load_and_center_dataset('YaleB_32x32.npy')
    print(f"Dataset loaded. Shape: {x.shape}")
    S = get_covariance(x)

    print("Generating Scree Plot...")
    eigen_values, _ = eigh(S)
    plot_scree(eigen_values[::-1])

    print("Applying PCA with 90% variance retention...")
    Lambda, U = get_eig_prop(S, 0.90)

    print("Reconstructing an image...")
    original_image = x[0]
    reconstructed_image = project_image(original_image, U)

    psnr_value, ssim_value = compute_reconstruction_quality(original_image, reconstructed_image)
    print(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.2f}")

    print("Displaying Original and Reconstructed Images...")
    fig, ax1, ax2 = display_image(original_image, reconstructed_image)
    plt.pause(1)

    print("Performing t-SNE Visualization...")
    projected_data = x @ U
    print("Performing Clustering...")
    labels = perform_clustering(projected_data, num_clusters=5)
    tsne_visualization(projected_data, labels)

main()
