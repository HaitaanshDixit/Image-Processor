import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage

from skimage import io


# Original Image
image = io.imread('.vscode\Projects\earth.jpg', as_gray=True)

plt.figure(figsize=(9,9))
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()



# Applying Gausssian filter
Gauss_filtered = ndimage.gaussian_filter(image, sigma=3)   # Sigma directly propotional to the blur

plt.figure(figsize=(8, 8))
plt.imshow(Gauss_filtered, cmap='gray')
plt.title('Gaussian Filtered Image')
plt.axis('off')
plt.show()


# Applying Median filter

Median_filtered = ndimage.median_filter(image, size=1)  

plt.figure(figsize=(8, 8))
plt.imshow(Median_filtered, cmap='gray')
plt.title('Median Filtered Image')
plt.axis('off')
plt.show()


# Applying Sobel filter

#Horizontal edges
Sobel_filtered_x = ndimage.sobel(image, axis=0)
#Vertical edges
Sobel_filtered_y = ndimage.sobel(image, axis=1)

#Combining both the edges
Sobel_filtered = np.hypot(Sobel_filtered_x, Sobel_filtered_y)

plt.figure(figsize=(8, 8))
plt.imshow(Sobel_filtered, cmap='gray')
plt.title('Sobel Filtered Image')
plt.axis('off')
plt.show()


#Bilateral filter
from skimage.restoration import denoise_bilateral
Bilateral_filtered = denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15)

plt.figure(figsize=(8, 8))
plt.imshow(Bilateral_filtered, cmap='gray')
plt.title('Bilateral Filtered Image')
plt.axis('off')
plt.show()


#Sharpening the image
from skimage.filters import unsharp_mask
sharpened_image = unsharp_mask(image, radius=1, amount=1)

plt.figure(figsize=(8, 8))
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened image')
plt.axis('off')
plt.show()


#Wiener filter
from scipy.signal import wiener
wiener_filtered = wiener(image, (5, 5))

plt.figure(figsize=(8, 8))
plt.imshow(wiener_filtered, cmap='gray')
plt.title('Enhanced image')
plt.axis('off')
plt.show()







