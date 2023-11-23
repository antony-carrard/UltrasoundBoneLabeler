import numpy as np
import cv2


class BoneSurfaceIdentification:
    def __init__(self,
                 str_elem_size: tuple=(15, 25),
                 threshold: float=0.1,
                 color: tuple=(255, 0, 0)) -> None:
        
        self.str_elem_size = str_elem_size
        self.threshold = threshold
        self.color = color
        
                 
    def bone_closing(self, img: np.ndarray, str_elem_size: tuple=(15, 25)) -> np.ndarray:
        """Apply a closing on the bone to stick the part together.
        
        Keyword arguments:
        img -- the bone probability mapping image
        str_elem_size -- the size of the structuring element in a tuple format. The first value of the tuple is the height of the ellipse, the second value is the width.
        Return: the closed image
        """
        
        # Generate intermediate image; use morphological closing to keep parts of the bone together
        str_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, str_elem_size)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, str_elem)


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
            cv2.drawContours(temp, cnt, -1, 255, cv2.FILLED)
            temp = temp.astype(np.float64)
            sum = np.sum(temp*img)
            if sum > max_sum:
                max_sum = sum
                max_cnt = i
        return max_cnt

    def trace_best_line(self, col_start, col_stop, starting_point, best_line, img, threshold_int, exp_decay):
        rows = img.shape[0]
        inc = np.sign(col_stop - col_start)
        prev_max = starting_point[0]
        for c in range(col_start, col_stop, inc):
            # Compute the cost of the column
            index_cost = np.exp(-np.abs(np.arange(-prev_max, rows-prev_max)) * exp_decay)
            column_cost = index_cost * img[:, c]
            column_cost_thresholded = np.clip(column_cost - threshold_int, 0, 255)
            best_row = column_cost_thresholded.argmax()
            if best_row == 0:
                break
            best_line[best_row, c] = 255
            prev_max = best_row


    def dynamic_selection(self, img, weighted_contour, threshold=0.1, exp_decay=0.05):
        
        best_line = np.zeros(img.shape, np.uint8)
        threshold_int = round(threshold*255)
        cols = img.shape[1]
        
        # Start from the brightest point
        brightest_point  = np.unravel_index(np.argmax(weighted_contour), weighted_contour.shape)
        best_line[brightest_point] = 255
        
        # Trace to the right side
        self.trace_best_line(brightest_point[1]+1, cols, brightest_point, best_line, img, threshold_int, exp_decay)
        
        # Trace to the left side
        self.trace_best_line(brightest_point[1]-1, 0, brightest_point, best_line, img, threshold_int, exp_decay)
        
        return best_line


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
            out = out.astype(np.float64)
            weighted_contour = out * img
            best_line = self.dynamic_selection(img, weighted_contour)
            
        return best_line


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
        
        # Apply the algorithm to return the new labeled image
        closed_img = self.bone_closing(img=img, str_elem_size=self.str_elem_size)
        cnts = self.get_contours(img=closed_img, threshold=self.threshold)
        label = self.label_image(img=closed_img, cnts=cnts)
        return label
        