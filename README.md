# Image-Segmentation
Image Segmentation using K-Means Clustering algorithm 

INTRODUCTION :

Image segmentation is a crucial task in computer vision and image processing,
involving partitioning an image into multiple segments to simplify its representation
and facilitate more meaningful analysis. K-means clustering is a popular
unsupervised learning algorithm used for image segmentation. In this project, we
explore the application of K-means clustering for image segmentation and evaluate
its effectiveness across different types of images, including MRI brain tumor images.


OBJECTIVE :

The primary objective of this project is to demonstrate the effectiveness of K-means clustering in segmenting various types of images, particularly focusing on MRI brain tumor images. We aim to find the optimal number of clusters (k) using the elbow method and apply K-means clustering to segment the images into distinct regions.


METHODOLOGY :

K-means Clustering:
K-means clustering is a partitioning algorithm that aims to divide a dataset into 'k' distinct, non-overlapping clusters. The algorithm works iteratively to assign each data point to the nearest cluster centroid and then updates the centroids based on the mean of the data points assigned to each cluster. This process continues until convergence.
Image Preprocessing:
Before applying K-means clustering, the input images undergo preprocessing steps to enhance their quality and improve segmentation results. This includes:
   
• Color Space Conversion: Converting images from the default BGR color space to RGB.
• Gaussian Blur: Applying Gaussian blur to smooth the image and reduce noise.
• Histogram Equalization: Enhancing image contrast using histogram equalization techniques.
     
  
  Determining Optimal K:
The number of clusters (K) is a crucial parameter in K-means clustering. To determine the optimal value of K, we use the Elbow Method. We compute the inertia (within- cluster sum of squares) for different values of K and select the value where the inertia begins to decrease at a slower rate, indicating the optimal number of clusters.


  IMPLEMENTATION :
  
  The project is implemented using Python programming language and popular libraries such as OpenCV, scikit-learn, and matplotlib. The following steps outline the implementation process:
1. Load the input image.
2. Preprocess the image by applying Gaussian blur and histogram equalization. 3. Determine the optimal value of K using the Elbow Method.
4. Perform K-means clustering with the optimal K value.
5. Segment the image based on the clustering results.
6. Visualize the original and segmented images.


 CHALLENGES FACED :
   
• Selection of Optimal K: Determining the optimal number of clusters (K) using the Elbow Method can be subjective and may require manual inspection.
• Image Preprocessing: Choosing appropriate preprocessing techniques and parameters to enhance image quality without distorting important features.
• Handling Large Datasets: K-means clustering may not scale well to large datasets due to its computational complexity.

      
Libraries that we used: 

 import numpy as np
 import cv2
 from sklearn.cluster import KMeans
 from tkinter import filedialog
 import matplotlib.pyplot as plt
 
  The libraries used in the provided code serve specific purposes and are chosen based on their functionalities and strengths:
  
   1. NumPy (import numpy as np):
       • NumPy is utilized for numerical computations and handling multi- dimensional arrays efficiently.
       • In the code, NumPy is employed to reshape images into arrays, compute differences between inertia values, and determine the optimal value of k using the elbow method.
   2. OpenCV (import cv2):
       • OpenCV is a widely used library for computer vision tasks, including image loading, processing, and manipulation.
       • In this code, OpenCV is employed to load images, convert color spaces, and display images using cv2.imshow().
   3. scikit-learn's KMeans (from sklearn.cluster import KMeans):
       • scikit-learn is a popular machine learning library in Python.
       • The KMeans class from scikit-learn is used for performing k-means
         clustering.
       • It allows us to find the optimal value of k and segment images based
         on pixel values.
   4. tkinter (from tkinter import filedialog):
       • tkinter is a built-in GUI toolkit for Python.
       • The filedialog module from tkinter is utilized to open a file dialog box
         for selecting an image file.
   5. Matplotlib (import matplotlib.pyplot as plt):
       • Matplotlib is a plotting library for Python.
       • It is used here to visualize the inertia values for different values of k
         during the elbow method analysis.
             
  Explanation of Library Usage:
  
• NumPy: Used for numerical operations and efficient handling of image data in array format.
• OpenCV: Employed for image loading, color space conversion, and displaying images.
• scikit-learn: Utilized for k-means clustering, including finding the optimal value of k and segmenting images based on clusters.
• tkinter: Used to create a simple file dialog box for selecting an image file interactively.
• Matplotlib: Employed for visualizing the inertia values to determine the optimal value of k using the elbow method.
            By leveraging these libraries, the code achieves image segmentation using k-means clustering efficiently and effectively. Each library plays a crucial role in different stages of the image segmentation process, from preprocessing and clustering to visualization.


CODE :

import numpy as np
import cv2
from sklearn.cluster import KMeans from tkinter import filedialog import matplotlib.pyplot as plt
def find_optimal_k(image_path):
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) pixel_values = image.reshape((-1, 3))
inertia_values = []
k_values = range(2, 11) # Adjust the range of k-values as needed
for k in k_values:
kmeans = KMeans(n_clusters=k) kmeans.fit(pixel_values) inertia_values.append(kmeans.inertia_)
plt.plot(k_values, inertia_values, 'bx-') plt.xlabel('Number of clusters (k)') plt.ylabel('Inertia')
plt.title('Inertia vs. Number of Clusters') plt.show()

# Determine the optimal k using the elbow method
diff = np.diff(inertia_values)
diff_r = diff[1:] / diff[:-1]
optimal_k = k_values[np.argmin(diff_r) + 1] # Add 1 to account for the removed
element in diff_r return optimal_k
def segment_image(image_path, num_clusters):
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixel_values = image.reshape((-1, 3))
kmeans = KMeans(n_clusters=num_clusters) kmeans.fit(pixel_values)
cluster_centers = np.array(kmeans.cluster_centers_, dtype=np.uint8) labels = np.array(kmeans.labels_, dtype=np.uint8)
segmented_image = cluster_centers[labels.flatten()] segmented_image = segmented_image.reshape(image.shape)
return segmented_image
image_path = filedialog.askopenfilename() optimal_k = find_optimal_k(image_path) print("Optimal value of k:", optimal_k)
segmented_image = segment_image(image_path, optimal_k)
cv2.imshow("Original Image", cv2.imread(image_path))
cv2.imshow(f"Segmented Image (K={optimal_k})", cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()


How does K-Means Clustering works :

  Image segmentation using K-means clustering works by partitioning an image into distinct regions or segments based on the similarity of pixel values. Here's how the process generally works:
   1. Initialization:
       • Choose the number of clusters 𝑘k to partition the image into. This represents the number of segments or regions you want to identify in the image.
 
       • Initialize 𝑘k cluster centroids randomly or using some predefined method.
   2. Assignment Step:
       • Assign each pixel in the image to the nearest cluster centroid based on some distance metric, typically Euclidean distance.
       • Each pixel is associated with the cluster centroid it is closest to in terms of color similarity.
   3. Update Step:
       • Recalculate the cluster centroids by computing the mean of all pixels assigned to each cluster.
       • This step effectively moves the centroids to the average position of the pixels in their respective clusters.
   4. Convergence:
       • Repeat the assignment and update steps iteratively until convergence.
       • Convergence occurs when the cluster assignments and centroids no
         longer change significantly between iterations, or after a predetermined number of iterations.
   5. Segmentation:
       • Once convergence is reached, each pixel in the image is assigned to one of the 𝑘k clusters.
       • The image is segmented into 𝑘k regions or segments, with pixels in each segment having similar color characteristics.
      
 Key Concepts:
 
• Cluster Centroids: The center points of the clusters, representing the average color values of the pixels in each cluster.
• Cluster Assignment: The process of assigning each pixel to the nearest cluster centroid based on distance metrics like Euclidean distance.
• Cluster Update: The process of recalculating the cluster centroids based on the current assignments, effectively updating the center points to better represent the cluster's 
  data.
• Convergence: The point at which the algorithm stops iterating because the cluster assignments and centroids have stabilized.
           
           
Advantages:
           
• Simple and easy to implement.
• Scalable to large datasets.
• Can handle a wide range of data types and dimensions.
• Provides fast computation, especially for low-dimensional data.

  
Limitations:

• Sensitivity to the initial choice of centroids.
• Requires a predefined number of clusters 𝑘k.
• May converge to local optima, leading to suboptimal results.
• Assumes clusters are spherical and of equal size, which may not always be the
case.


 APPLICATONS:
 
   1. Medical Image Analysis:
      • Tumor Detection: Segmentation of medical images, such as MRI or CT scans, can aid in the detection and analysis of tumors and abnormalities within the body.
      • Organ Segmentation: It can be used to segment different organs or tissues within medical images for diagnosis and treatment planning.
   2. Remote Sensing:
      • Land Cover Classification: Satellite or aerial images can be segmented to classify different land cover types, such as forests, water bodies, urban areas, and 
        agricultural land.
      • Vegetation Analysis: Segmentation of vegetation in satellite imagery can help monitor vegetation health, estimate biomass, and track changes in land use over time.
   3. Object Recognition and Tracking:
      • Object Detection: Image segmentation can be used as a preprocessing step for object detection by segmenting the image into regions of interest, making it easier to detect 
        and recognize objects.
      • Motion Tracking: In video analysis, segmentation can help track moving objects by segmenting the foreground from the background.                                         
   4. Biometrics:
      • Fingerprint and Iris Recognition: Segmentation of fingerprint or iris images can help extract features for biometric authentication and identification systems.
         
   5. Image Compression:
      • Region-Based Compression: Segmentation can be used to divide an image into regions with similar characteristics, which can then be compressed more efficiently compared to 
        compressing the entire image uniformly.
   6. Image Editing and Enhancement:
      • Selective Editing: Segmentation allows users to selectively edit or enhance specific regions of an image, such as adjusting brightness, contrast, or color in particular
        areas.

      
RESULTS :
    
Image segmentation using K-Means Clustering yields best results for RGB images
  with objects with different contrast colours. Although it does not give good results
for B&W images such as MRI brain tumor images.
 

CONCLUSION :

In conclusion, image segmentation using K-means clustering is a powerful technique
for partitioning images into meaningful regions. By preprocessing images and
selecting the optimal number of clusters, we can achieve accurate segmentation
results. The project demonstrates the effectiveness of K-means clustering for various
types of images, including medical images, making it a versatile tool for image
analysis tasks.


FUTURE WORK :

Future work may involve exploring advanced segmentation techniques, such as deep
learning-based methods, to improve segmentation accuracy and handle complex
image datasets more effectively. Additionally, integrating interactive user interfaces
for parameter tuning and result visualization could enhance the usability of the
segmentation system.


REFERENCES :

• OpenCV Documentation: https://opencv.org/
• scikit-learn Documentation: https://scikit-learn.org/stable/
• Chat GPT
• Youtube : https://www.youtube.com/@DigitalSreeni
• Github
• GeeksforGeeks
