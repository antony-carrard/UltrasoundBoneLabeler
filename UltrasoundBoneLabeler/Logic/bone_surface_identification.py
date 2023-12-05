import numpy as np
import cv2


class BoneSurfaceIdentification:
    def __init__(self,
                 threshold: float=0.1,
                 sigma: float=10,
                 bone_width_min: float=0.4,
                 thickness: int=4,
                 color: tuple=(255, 0, 0)) -> None:
        
        self.threshold = threshold
        self.sigma = sigma
        self.bone_width_min = bone_width_min
        self.thickness = thickness
        self.color = color


    def get_contours(self, img: np.ndarray, threshold: float=0.1) -> tuple:
        """identify the contours from the image
        
        Keyword arguments:
        img -- the bone probability mapping image
        threshold -- the threshold to apply to the image before detecting contours
        Return: the tuple containing the contours description
        """
        
        # Trim the values too low
        threshold_int = round(threshold*255)
        ret,thresh = cv2.threshold(img,threshold_int,255,cv2.THRESH_BINARY)

        # Find contours in the image
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return cnts


    def heaviest_contour(self, img: np.ndarray, cnts: tuple) -> np.ndarray:
        """Return the id of the heaviest contour.
        The weight of the contour is the sum of the thresholded contour * the image.
        
        Keyword arguments:
        img -- the bone probability mapping image
        cnts -- the contours from which to select the heaviest one
        Return: the heaviest contour
        """
        
        max_sum = 0
        max_cnt = 0
        
        # If no contour is detected, return None
        if len(cnts) == 0:
            return None
        
        # Iterate on each contour and detect the heaviest
        for i, cnt in enumerate(cnts):
            temp = np.zeros(img.shape, np.uint8)
            cv2.drawContours(temp, cnts, i, 255, cv2.FILLED)
            sum = np.sum(temp*img)
            if sum > max_sum:
                max_sum = sum
                max_cnt = i
        return max_cnt
    
    def gaussian(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """Create a one dimension gaussian array.

        Keyword arguments:
        x -- values on which to apply the gaussian function
        mu -- the mean of the distribution
        sigma -- the variance of the distribution
        Return: the one dimension gaussian to compute the shadow value
        """
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))
    

    def trace_best_line(self, col_start, col_stop, starting_point, best_line, img, threshold, sigma):
        rows = img.shape[0]
        inc = np.sign(col_stop - col_start)
        prev_max = starting_point[0]
        col_count = 0
        for c in range(col_start, col_stop, inc):
            
            # Compute the cost of the column
            index_cost = self.gaussian(np.arange(-prev_max, rows-prev_max), 0, sigma)
            column_cost = img[:, c] * index_cost
            
            # Remove the negative values
            column_cost_thresholded = np.clip(column_cost - threshold, 0, 255)
            
            # Keep the best point of the column
            best_row = column_cost_thresholded.argmax()
            if best_row == 0:
                break
            best_line[best_row, c] = 255
            prev_max = best_row
            col_count += 1
        
        return col_count


    def dynamic_selection(self, img, weighted_contour, threshold=0.1, sigma=10):
        
        best_line = np.zeros(img.shape, np.uint8)
        cols = img.shape[1]
        col_count = 0
        
        # Start from the brightest point
        brightest_point  = np.unravel_index(np.argmax(weighted_contour), weighted_contour.shape)
        best_line[brightest_point] = 255
        
        # Trace to the right side
        col_count += self.trace_best_line(brightest_point[1]+1, cols, brightest_point, best_line, img, threshold, sigma)
        
        # Trace to the left side
        col_count += self.trace_best_line(brightest_point[1]-1, 0, brightest_point, best_line, img, threshold, sigma)
        
        return best_line, col_count
    
    def thicken_line(self, img, thickness):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        img = cv2.dilate(img, kernel, iterations=1)
        return img


    def label_image(self, img: np.ndarray, cnts: tuple) -> np.ndarray:
        """Create the label from the heaviest contour by drawing the brightest pixel of each column of the image from the heaviest contour.
        
        Keyword arguments:
        img -- the bone probability mapping image
        cnts -- the contours from which to select the heaviest one
        Return: the heaviest contour
        """

        # Detect the heaviest contour
        cnt_id = self.heaviest_contour(img, cnts)
        out = np.zeros(img.shape, np.uint8)
        best_line = np.zeros(img.shape, np.uint8)
        
        
        # Iterate on the column of the heaviest contour and draw the brighest pixel on a new image
        if cnt_id is not None:
            cv2.drawContours(out, cnts, cnt_id, 255, cv2.FILLED)
            weighted_contour = out * img
            best_line, col_count = self.dynamic_selection(img, weighted_contour, self.threshold, self.sigma)
            best_line_thickened = self.thicken_line(best_line, self.thickness)
            
        return weighted_contour, best_line_thickened, col_count


    def draw_on_image(self, img: np.ndarray, label: np.ndarray, color: tuple=(255, 0, 0)) -> np.ndarray:
        """Draw the segmented line on the original image for verification.
        
        Keyword arguments:
        img -- the image on which to draw the line
        label -- the labeled image containing the line to draw
        color -- the color of the line to draw. Default is (255, 0, 0) (red in rgb format)
        Return: the image with the line drawn on it
        """
        
        # Trace the segment on the original image
        image_with_segment = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        image_with_segment[label==255] = color
        image_with_segment = cv2.cvtColor(image_with_segment, cv2.COLOR_BGR2RGB) 
        return image_with_segment


    def identify_bone_surface(self, img: np.ndarray) -> np.ndarray:
        """Identify the bone surface from the bone probabaility mapping.
        
        Keyword arguments:
        img -- the image from which to detect the bone surface
        Return: the label containing the surface of the bone
        """
        
        col_count = 0
        image_width = img.shape[1]
        
        # Apply the algorithm to return the new labeled image
        cnts = self.get_contours(img=img, threshold=self.threshold)
        img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        weighted_contour, label, col_count = self.label_image(img=img, cnts=cnts)
                    
        # Check if the width of the segmentation is wide enough to be considered a bone
        if image_width*self.bone_width_min >= col_count:
            traced_line = np.zeros(img.shape).astype(np.uint8)
        else:
            traced_line = label
            
        weighted_contour = cv2.normalize(weighted_contour, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        
        return weighted_contour, label, traced_line
        