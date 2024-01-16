import os, os.path
import numpy as np
import cv2
import pydicom as dicom


class FileManager:
    def __init__(self) -> None:
        pass
    
    
    def import_dicom(self, path: str) -> np.ndarray:
        """Import the images from a dicom file.
        
        Keyword arguments:
        path -- The path from which to load the dicom file
        Return: return a list of converted images with the filenames of each one    
        """
        ds = dicom.dcmread(path)
        return ds.pixel_array[:, :, :, 0].transpose(1, 2, 0)


    def _image_batch(self, image_batch, batch_size: int=4) -> np.ndarray:
        """A generator to split the 3D NumPy image array into batch sized chunks.
        
        Keyword arguments:
        image_batch -- the images in a 3D NumPy format
        batch_size -- the size of the 3D batch to yield; the number of images in the 3D array
        Return: return a batch of the 3D NumPy image array
        """
        n_images = len(image_batch[0, 0])
        for ndx in range(0, n_images, batch_size):
            yield image_batch[:, :, ndx:min(ndx + batch_size, n_images)]


    def resize_images(self, image_array: np.ndarray, image_size: tuple=(256, 256)) -> np.ndarray:
        """Resize a batch of images to the specified size.
        
        Keyword arguments:
        image_batch -- the batch of images in a 3D NumPy array. Format is [width, height, image_number]
        image_size -- a tuple specifying the size of the resized image
        Return: the resized batch of images of the same input format
        """
        batch_list = []
        for batch in image_array:
            batch_list.append(cv2.resize(batch, image_size, interpolation = cv2.INTER_AREA))
            
        return np.dstack(batch_list).transpose(2, 0, 1)
    
    
    def _crop_gray_bars(self, image: np.ndarray) -> tuple[int, int]:
        """Crop the gray bars from the image.
           As the gray border have a lot of similar values, the image is cropped by filtering the part with many similar values.
           
        Keyword arguments:
        image_batch -- the image from which to remove the gray bars
        Return: the image with the gray bars removed
        """
        # Downsample the image to cluster similar pixels
        image_downsampled = image // 16
        
        # Count the number of unique values of each columns and keep the largest one for each column
        max_unique = []
        for col in image_downsampled.T:
            values, counts = np.unique(col, return_counts=True)
            max_unique.append(np.max(counts))
        
        # Compute the gradient of the median to identify variations
        grad = np.gradient(max_unique)
        
        # When the gradient of the count decrease, the image begins
        start = np.argmin(grad)
        
        # When the gradient of the count increase, the image ends
        end = np.argmax(grad)
        
        return start, end
    
    
    def _remove_grid(self, image: np.ndarray, threshold: float=5.0, bottom_rows: int=20) -> int:
        """Remove the grid at the bottom of the image.
           The grid is filtered by removing row of the images containing high intensity pixels.
           
        Keyword arguments:
        image_batch -- the image from which to remove the grid
        threshold -- the threshold used to filter the grid
        bottom_rows -- the number of rows at the bottom of the image in which to seek the grid
        Return: the image with the grid removed
        """
        # Get the max values of the bottom rows of the image
        max = np.max(image[-bottom_rows:], axis=1)
        
        # Get the gradient to check for variation in the maximals
        grad = np.gradient(max)[::-1]
        
        # Detect the peaks above the threshold
        peak = np.argwhere(grad > threshold)
        
        # Get the peak that is the highest in the image, to filter the white grid
        no_tick = np.max(peak)
        
        return no_tick
        
    
    def auto_crop(self, image_array: np.ndarray) -> np.ndarray:
        """Crop the images automatically by removing the gray frame and the axis from the dicom images.
           The detection of the gray frame and the axis is done automatically.
        
        Keyword arguments:
        image_batch -- the batch of images in a 3D NumPy array. Format is [width, height, image_number]
        Return: the cropped batch of images of the same input format
        """
        # First, remove the gray bars from the image.
        first_image = image_array[0]
        left, right = self._crop_gray_bars(first_image)
        
        # Then, remove the white grid from the image
        first_image_no_bars = first_image[:, left:right]
        bottom = self._remove_grid(first_image_no_bars)
        
        # Crop the entire 3D array of images
        cropped_image_array = image_array[:, :-bottom, left:right]
        
        return cropped_image_array


    def import_images(self, path: str=os.getcwd()) -> tuple[list[np.ndarray], list[str]]:
        """Import the images from the specified path.
            
        Keyword arguments:
        path -- The path from which to load the images. Default is the current path.
        Return: return a list of found images with the filenames of each one
        """
        imgs = []
        filenames = []
        valid_images = [".jpg", ".png"]
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            filenames.append(f)
            imgs.append(cv2.imread(os.path.join(path,f), 0))
            
        return imgs, filenames


    def save_images_filenames(self, img_list: list[np.ndarray], filenames: list[str], path: str=os.getcwd()) -> None:
        """Save a list of image to a list of filenames
            
        Keyword arguments:
        img_list -- the list of images to be saved
        filenames -- the filenames containing the path and the name of the image.
        path -- The path to which to save the images. Default is the current path.
        """
        # Check if the size of the images list and the filenames matches
        assert len(img_list) == len(filenames), "The number of images and filenames are not the same"
        
        for img, f in zip(img_list, filenames):
            # Change the extension to .png
            f = os.path.splitext(f)[0] + ".png"
            cv2.imwrite(os.path.join(path,f), img)
            
            
    def save_images(self, img_list: list[np.ndarray], filename: str, path: str=os.getcwd()) -> None:
        """Save a list of images, with the filename + the number of the images attached.
            
        Keyword arguments:
        img_list -- the list of images to be saved 
        filename -- the base filename, on which the number of the image is added
        path -- The path to which to save the images. Default is the current path.
        """
        for n, img in enumerate(img_list):
            # Add the number of the image to the filename and the .png extension
            f = f"{os.path.splitext(filename)[0]}-{n}.png"
            cv2.imwrite(os.path.join(path,f), img)
        