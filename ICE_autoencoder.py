import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import tensorflow as tf
from tensorflow.keras import layers

# Load the data (assuming they are stored as .npy files)
epilepsy_data = np.load('/Users/trashnavadi/Documents/Data_Analysis/2023/analyses/2024/deep_learning/reshaped_ICE_subj_FC_mat/big_matrix_epilepsy.npy', allow_pickle=True)
control_data = np.load('/Users/trashnavadi/Documents/Data_Analysis/2023/analyses/2024/deep_learning/reshaped_ICE_subj_FC_mat/big_matrix_CTRL.npy', allow_pickle=True)

# Reshape the matrices in each cell to vectors
epilepsy_vectors = np.array([x.flatten() for x in epilepsy_data])
control_vectors = np.array([x.flatten() for x in control_data])

# Normalize the data (optional, if not already normalized)
# epilepsy_vectors = normalize(epilepsy_vectors)
# control_vectors = normalize(control_vectors)

# Create and train an autoencoder
nROIs = 56  # Size of the hidden layer
input_dim = epilepsy_vectors.shape[1]

autoencoder = tf.keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(nROIs, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(epilepsy_vectors, epilepsy_vectors, epochs=50, batch_size=32, shuffle=True, validation_data=(control_vectors, control_vectors))

# Encode the data
encoder = tf.keras.Model(autoencoder.input, autoencoder.layers[0].output)
encoded_epilepsy = encoder.predict(epilepsy_vectors)
encoded_control = encoder.predict(control_vectors)

# Perform analysis on the encoded data
mean_encoded_epilepsy = np.mean(encoded_epilepsy, axis=0)
std_encoded_epilepsy = np.std(encoded_epilepsy, axis=0)

mean_encoded_control = np.mean(encoded_control, axis=0)
std_encoded_control = np.std(encoded_control, axis=0)

# Display the results
print('Mean and Standard Deviation of Encoded Epilepsy Data:')
print(mean_encoded_epilepsy)
print(std_encoded_epilepsy)

print('Mean and Standard Deviation of Encoded Control Data:')
print(mean_encoded_control)
print(std_encoded_control)

# K-means clustering
num_clusters = 2
kmeans_epilepsy = KMeans(n_clusters=num_clusters).fit(encoded_epilepsy)
kmeans_control = KMeans(n_clusters=num_clusters).fit(encoded_control)

epilepsy_cluster_centers = kmeans_epilepsy.cluster_centers_
control_cluster_centers = kmeans_control.cluster_centers_

# Compare clustering results from autoencoder with your k-means results from before
distances_epilepsy = cdist(epilepsy_cluster_centers, epilepsy_cluster_centers)
distances_control = cdist(control_cluster_centers, control_cluster_centers)

# Display the distances for comparison
print('Distances between cluster centers (Epilepsy):')
print(distances_epilepsy)

print('Distances between cluster centers (Control):')
print(distances_control)
