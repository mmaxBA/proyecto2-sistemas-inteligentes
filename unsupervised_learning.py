import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt

# Get all images

emoji_types = ['Happy', 'Sad', 'Angry', 'Poo', 'Surprised']
X = []

base_path = os.path.join(os.getcwd(), "images_nor")
image_folders = os.listdir(base_path)
for folder in image_folders:
    folder_path = os.path.join(base_path, folder)
    files = os.listdir(folder_path)
    for file in files:
        img = Image.open(os.path.join(folder_path, file))
        img = img.convert('L')
        img_array = np.array(img)
        x, y = img_array.shape
        img_array = img_array.reshape(x * y)
        X.append(img_array)
        name_images.append(file)

X = np.asarray(X)

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    print(model.children_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)



KMeans

for i in range(1, 5 + 1):
    print(f'n_clusters: {i}')
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    frequencies = {}
    for l in kmeans.labels_:
        if l in frequencies:
            frequencies[l] += 1
        else:
            frequencies[l] = 1
    # print(frequencies.keys)
    for key in frequencies.keys():
        print(f'{key}: {frequencies[key]}')
    print('---------------------------------------------------')

plt.title('n_clusters: 5')
plot the top three levels of the dendrogram
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X)
plot_dendrogram(model, truncate_mode='level', p=2)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
