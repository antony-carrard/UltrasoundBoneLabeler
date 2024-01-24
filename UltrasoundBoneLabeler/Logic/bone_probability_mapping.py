import numpy as np
import math
import cv2
from scipy import fftpack


class BoneProbabilityMapping:
    def __init__(self,
                 img_dimensions: tuple[int, int],
                 gaussian_kernel_size: int=25,
                 binary_threshold: float=0.2,
                 top_layer: float=0.1,
                 log_kernel_size: int=31,
                 shadow_sigma: float=100.0,
                 shadow_n_sigmas: float=2.0,
                 quad_filter_sigma: float=0.5,
                 quad_filter_wavelength: float=2.0,
                 phase_symmetry_threshold: float=0.01,
                 phase_symmetry_epsilon: float=0.001) -> None:
        
        self.img_dimensions = img_dimensions
        self.gaussian_kernel_size = gaussian_kernel_size
        self.binary_threshold = binary_threshold
        self.top_layer = top_layer
        self.log_kernel_size = log_kernel_size
        self.shadow_sigma = shadow_sigma
        self.shadow_n_sigmas = shadow_n_sigmas
        self.quad_filter_sigma = quad_filter_sigma
        self.quad_filter_wavelength = quad_filter_wavelength
        self.phase_symmetry_threshold = phase_symmetry_threshold
        self.phase_symmetry_epsilon = phase_symmetry_epsilon
        
        # Compute the spherical quadratude filters 
        self.log_gabor_filters, self.riesz_filter = self.spherical_quadratude_filters(sigma=self.quad_filter_sigma, wavelength=self.quad_filter_wavelength)
        
        # Check if optimizing the images dimensions is necessary
        self.dimension_optimization, self.optimal_dimensions = self.check_dimensions()


    def gaussian_filter(self, img: np.ndarray, kernel_size: int=25) -> np.ndarray:
        """Apply a gaussian filter to the input image.

        Keyword arguments:
        img -- the image on which to apply the filter
        kernel_size -- the size of the used kernel (default 25)
        Return: the blurrred imagee
        """
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


    def apply_mask(self, img: np.ndarray, threshold: float=0.2, top_layer: float=0.1) -> np.ndarray:
        """Apply a mask on the input image.
        First, remove the first layers of the image, as it likely contains only soft tissues.
        Then, apply a thresholding on the image to remove non-bony parts.

        Keyword arguments:
        img -- the image on which to apply the mask
        threshold -- the threshold of the mask to apply
        top_layer -- the proportion of the image to remove from the top
        Return: the binarized mask of the image
        """

        # Remove the top layer
        height = round(top_layer * img.shape[0])
        img_no_top = img.copy()
        img_no_top[:height] = 0

        # Apply the threshold
        threshold_int = round(threshold*255)
        ret,thresh = cv2.threshold(img_no_top, threshold_int, 255, cv2.THRESH_BINARY)
        
        # Transform the mask to a boolean type
        thresh = thresh.astype(bool)
        
        return thresh


    def laplacian_of_gaussian(self, img: np.ndarray, kernel_size: int=31) -> np.ndarray:
        """Apply the laplacian of a gaussian (LoG) filter to the input image.

        Keyword arguments:
        img -- the image on which to apply the filter
        kernel_size -- the size of the used kernel (default 31)
        Return: the laplacian of a Gaussian of the image
        """
        laplacian = cv2.Laplacian(img, ddepth=cv2.CV_64F, ksize=kernel_size)

        # Only keep the negative pixels
        laplacian = -laplacian
        laplacian /= np.max(laplacian)
        laplacian = np.clip(laplacian, 0, 1)
        
        return laplacian


    def gaussian(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """Create a one dimension gaussian array.

        Keyword arguments:
        x -- values on which to apply the gaussian function
        mu -- the mean of the distribution
        sigma -- the variance of the distribution
        Return: the one dimension gaussian to compute the shadow value
        """
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))


    def get_optimal_depth(self, current_row: int, total_rows: int, sigma: float, number_of_sigmas: float) -> int:
        """Get the optimal depth of the image to analyse for the shadow value.
        The purpose of this function is to limit the computation to meaningful values of the gaussian array,
        by removing values that are too low.
        The function only keep values by a certain number of sigmas near the mean of the gaussian array.

        Keyword arguments:
        current_row -- the current row being of the image being computed
        total_rows -- the total number of rows of the image, in other words it's height
        sigma -- the variance of the distribution
        number_of_sigmas -- the number of sigmas at which to keep the values
        Return: the optimal depth until which to compute the values
        """
        optimal_depth = round(number_of_sigmas*sigma)
        if total_rows - current_row < optimal_depth:
            optimal_depth = total_rows - current_row
        return optimal_depth


    def shadow_value(self, img: np.ndarray, mask: np.ndarray, sigma: float=100.0, number_of_sigmas: float=2.0) -> np.ndarray:
        """This function applies the shadow value to the image by computing a gaussian weighted accumulation of every pixel.
        The darker a region below a certain pixel is, the brightest the pixel will be, to help highlighting pixel situated above an acoustic shadow.

        Keyword arguments:
        img -- the image on which to apply the filter
        mask -- the binary mask on which the potential bone structure can be. Row where the mask is False are ignored in the computation.
        sigma -- the variance of the distribution of the gaussian
        Return: the shadow value of the image
        """
        # Create the shadow image
        shadow_img = np.ones(img.shape)
        
        # Get the dimensions of the image
        R, C = self.img_dimensions
        
        # Create the gaussian 2D array, with the same width as the image
        G = self.gaussian(np.arange(R), 0, sigma)
        G_2D = np.tile(G, (C, 1)).T
        
        # Get the optimal depth to compute the gaussian
        optimal_depth = round(number_of_sigmas*sigma)
        
        # Compute a precomputed denominator with the optimal depth
        optimal_denominator = np.tile(np.sum(G[:optimal_depth]), (C, 1)).T
        
        # Iterate on every row of the image
        for a in range(R):
            # Skip the row if the mask is empty
            if not (True in mask[a]):
                continue
            depth = self.get_optimal_depth(a, R, sigma, number_of_sigmas)
            I = img[a:a+depth]
            
            # Compute the numerator and denominator of the shadow formula
            numerator = np.sum(G_2D[:depth] * I, axis=0)
            if depth == optimal_depth:
                denominator = optimal_denominator
            else:
                denominator = np.tile(np.sum(G[:depth]), (C, 1)).T
            
            # Compute the shadow value and return it
            shadow_img[a] = numerator / denominator
        
        # Invert the intensity to brighten the pixels with the shadow underneath
        shadow_img /= np.max(shadow_img)
        shadow_img = 1 - shadow_img
        shadow_img = shadow_img**2
        
        return shadow_img


    def spherical_quadratude_filters(self, sigma: float=0.3, wavelength: float=200) -> tuple[list[np.ndarray], np.ndarray]:
        """This function compute the monogenic signals to later compute the local energy, local phase and phase symmetry.

        Keyword arguments:
        img -- the image on which to apply the filter
        sigma -- the variance of the distribution of the monogenic signal.
        Return: the three quadrature filters
        """
        # Get the dimensions of the image
        rows, cols = self.img_dimensions
        
        # Compute the frequency image
        xy = np.mgrid[-rows//2:rows//2, -cols//2:cols//2]
        xy = xy / np.max((rows, cols))
        frequency_x = xy[1]
        frequency_y = xy[0]
        frequency = np.sqrt(frequency_y**2 + frequency_x**2)
        frequency[math.ceil(rows/2), math.ceil(cols/2)] = 1e-300 # Avoid division by zero
        riesz_filter = (1j * frequency_y - frequency_x)/ frequency
        
        # Create extra wavelenght from the specified one
        wavelengths = [wavelength/2, wavelength, wavelength*2]
        
        # Compute the even monogenic signal
        log_gabor_filters = []
        for w in wavelengths:
            centre_frequency = 2*np.pi/w
            log_gabor_filter = np.exp(-np.log(frequency/centre_frequency)**2/(2*np.log(sigma)**2))
            log_gabor_filters.append(log_gabor_filter)
        
        return log_gabor_filters, riesz_filter
    
    
    def check_dimensions(self):
        """This function checks if an optimization of the images dimensions is necessary.
        
        Keyword arguments:
        img -- the image on which to add the padding
        Return: the boolean value of the necessity of the dimension optimization
        """
        # Get the dimensions of the input image
        rows,cols = self.img_dimensions

        # Get the optimal dimensions of the image
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        
        # Return if an optimization is necessary
        return (nrows, ncols) != (rows, cols), (nrows, ncols)


    def optimize_fft_dimensions(self, img: np.ndarray):
        """This function optimize the dimension of an image to accelerate the processing time.
        A padding is added to the sides of the images to to make it divisible by 2, 3 or 5.
        
        Keyword arguments:
        img -- the image on which to add the padding
        Return: the padded image
        """
        # Get the dimensions of the input image
        rows,cols = self.img_dimensions
        nrows,ncols = self.optimal_dimensions

        # Convert the image type if the image is complex
        nimg = np.zeros((nrows,ncols))
        if img.dtype == np.complex128:
            nimg = nimg.astype(np.complex128)
        
        # Add the padding to the image
        nimg[:rows,:cols] = img

        return nimg


    def monogenic_signal(self, img: np.ndarray, log_gabor_filters: list[np.ndarray], riesz_filter: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Create the monogenic signal to later construct the local energy, the local phase and the local symmetry.

        Keyword arguments:
        img -- the original image
        gabor_even -- the even gabor quadrature filter
        gabor_odd_1 -- the first odd gabor quadrature filter
        gabor_odd_2 -- the second odd gabor quadrature filter
        Return: The odd and even monogenic signals
        """
        
        mono_even_scales = []
        mono_odd_scales = []
        
        # If a dimensions optimization is necessary, apply the optimization
        if self.dimension_optimization:
            
            # get the optimal dimensions of the image
            nimg = self.optimize_fft_dimensions(img)
            nriesz_filter = self.optimize_fft_dimensions(riesz_filter)
            nlog_gabor_filters = []
            for log_gabor_filter in log_gabor_filters:
                nlog_gabor_filters.append(self.optimize_fft_dimensions(log_gabor_filter))

            # Build the monogenic signals
            rows,cols = img.shape
            
            for nlog_gabor_filter in nlog_gabor_filters:
                fft_filter = fftpack.fft2(nimg) * fftpack.ifftshift(nlog_gabor_filter)
                mono_even = np.real(fftpack.ifft2(fft_filter))
                mono_odd_complex = fftpack.ifft2(fft_filter * fftpack.ifftshift(nriesz_filter))
                mono_odd_1 = np.real(mono_odd_complex)
                mono_odd_2 = np.imag(mono_odd_complex)
                mono_odd = np.sqrt(mono_odd_1**2 + mono_odd_2**2)

                # Crop the image to get back to the original dimensions of the image
                mono_even = mono_even[:rows,:cols]
                mono_odd = mono_odd[:rows,:cols]
                
                # Add the filters to the scales
                mono_even_scales.append(mono_even)
                mono_odd_scales.append(mono_odd)
                
        else:
            # Build the monogenic signals
            for log_gabor_filter in log_gabor_filters:
                fft_filter = fftpack.fft2(img) * fftpack.ifftshift(log_gabor_filter)
                mono_even = np.real(fftpack.ifft2(fft_filter))
                mono_odd_complex = fftpack.ifft2(fft_filter * fftpack.ifftshift(riesz_filter))
                mono_odd_1 = np.real(mono_odd_complex)
                mono_odd_2 = np.imag(mono_odd_complex)
                mono_odd = np.sqrt(mono_odd_1**2 + mono_odd_2**2)
                
                # Add the filters to the scales
                mono_even_scales.append(mono_even)
                mono_odd_scales.append(mono_odd)
        
        return mono_even_scales, mono_odd_scales


    def local_energy(self, mono_even_scales: list[np.ndarray], mono_odd_scales: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        """Construct the local energy from the monogenic signals.
        
        Keyword arguments:
        mono_even_scales -- the even part of the monogenic signal
        mono_odd_scales -- the odd part of the monogenic signal
        Return: the local energy
        """
        local_energy_scales = []
        
        # Compute the local energy for every scale
        for mono_even, mono_odd in zip(mono_even_scales, mono_odd_scales):
            local_energy_scales.append(np.sqrt(mono_even**2 + mono_odd**2))
        
        # Sum the scales together
        local_energy = np.sum(np.array(local_energy_scales), axis=0)
        
        # Normalize the local energy
        local_energy /= np.max(local_energy)

        return local_energy, local_energy_scales


    def local_phase(self, mono_even_scales: list[np.ndarray], mono_odd_scales: list[np.ndarray]) -> np.ndarray:
        """Construct the local phase from the monogenic signals.
        
        Keyword arguments:
        mono_even -- the even part of the monogenic signal
        mono_odd -- the odd part of the monogenic signal
        Return: the local phase
        """
        mono_odd = np.sum(np.array(mono_odd_scales), axis=0)
        mono_even = np.sum(np.array(mono_even_scales), axis=0)
            
        local_phase = np.arctan2(mono_even, mono_odd)
        
        # Min-max normalize the phase
        local_phase = (local_phase-np.min(local_phase))/(np.max(local_phase)-np.min(local_phase))

        return local_phase


    def phase_symmetry(self,
                       mono_even_scales: list[np.ndarray],
                       mono_odd_scales: list[np.ndarray],
                       local_energy_scales: list[np.ndarray],
                       threshold: float=0.01,
                       epsilon: float=1e-300) -> np.ndarray:
        """Construct the phase symmetry from the monogenic signals.
        
        Keyword arguments:
        mono_even -- the even part of the monogenic signal
        mono_odd -- the odd part of the monogenic signal
        local_energy -- the local energy of the image
        threshold -- only the values above the threshold are kept
        epsilon -- a small number to avoid division by zero
        Return: the phase symmetry
        """
        
        # Scales the threshold for a uint8 image type
        max_uint8 = np.iinfo(np.uint8).max # 255
        scaled_threshold = threshold * max_uint8
        
        phase_symmetry_scales = []
        
        # Compute the phase symmetry for every scale
        for mono_even, mono_odd, local_energy in zip(mono_even_scales, mono_odd_scales, local_energy_scales):
            phase_symmetry = (np.abs(mono_even) - mono_odd - scaled_threshold) / (local_energy + epsilon)
            phase_symmetry_scales.append(phase_symmetry)
            
        # Sum the scales together
        phase_symmetry = np.sum(np.array(phase_symmetry_scales), axis=0)

        # Normalize the image
        phase_symmetry /= np.max(phase_symmetry)

        # Remove the negative part
        phase_symmetry = np.clip(phase_symmetry, 0, 1)

        return phase_symmetry


    def IBS(self, img: np.ndarray) -> np.ndarray:
        """ Compute integrated backscatter by computing the cummulated squared sum of an image column wise.
        
        Keyword arguments:
        img -- the original image
        Return: the integrated backscatter
        """
        # Cumulative sum of each row
        ibs = np.cumsum(img**2, axis=0)
        
        return ibs


    def prob_map(self, img: np.ndarray, mask: np.ndarray, laplacian: np.ndarray, shadow_image: np.ndarray, local_energy: np.ndarray, local_phase: np.ndarray, phase_symmetry: np.ndarray, ibs: np.ndarray) -> np.ndarray:
        """ Compute the bone probability mapping by multiplying all the filtered images.
        
        Keyword arguments:
        img -- the original image
        mask -- the binary mask of the image
        laplacian -- the laplacian of the Gaussian of the image (LoG)
        shadow_image -- the shadow image
        local_energy -- the local energy of the image
        local_phase -- the local phase of the image
        phase_symmetry -- the phase symmetry of the image
        ibs -- the integrated backscatter of the image

        Return: the bone probability mapping
        """
        prob_map = img * mask * laplacian * shadow_image * local_energy * local_phase * phase_symmetry * ibs

        return prob_map


    def apply_all_filters(self, img: np.ndarray) -> np.ndarray:
        """Apply all the filters to the image to find the bone probability mapping.
        
        Keyword arguments:
        img -- the original image
        Return: the bone probability mapping
        """
        gaussian_img = self.gaussian_filter(img=img, kernel_size=self.gaussian_kernel_size)
        mask = self.apply_mask(img=gaussian_img, threshold=self.binary_threshold, top_layer=self.top_layer)
        laplacian = self.laplacian_of_gaussian(img=gaussian_img, kernel_size=self.log_kernel_size)
        gaussian_img = cv2.normalize(gaussian_img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        shadow_image = self.shadow_value(img=gaussian_img, mask=mask, sigma=self.shadow_sigma, number_of_sigmas=self.shadow_n_sigmas)
        mono_even_scales, mono_odd_scales = self.monogenic_signal(img=img, log_gabor_filters=self.log_gabor_filters, riesz_filter=self.riesz_filter)
        local_energy_img, local_energy_scales = self.local_energy(mono_even_scales=mono_even_scales, mono_odd_scales=mono_odd_scales)
        local_phase_img  = self.local_phase(mono_even_scales=mono_even_scales, mono_odd_scales=mono_odd_scales)
        phase_symmetry_img  = self.phase_symmetry(mono_even_scales=mono_even_scales,
                                                  mono_odd_scales=mono_odd_scales,
                                                  local_energy_scales=local_energy_scales,
                                                  threshold=self.phase_symmetry_threshold,
                                                  epsilon=self.phase_symmetry_epsilon)
        ibs = self.IBS(img=gaussian_img)        
        prob_map_img = self.prob_map(img=gaussian_img,
                                      mask=mask,
                                      laplacian=laplacian,
                                      shadow_image=shadow_image,
                                      local_energy=local_phase_img,
                                      local_phase=local_phase_img,
                                      phase_symmetry=phase_symmetry_img,
                                      ibs=ibs)
        
        # While not necessary for the algorithm, multiply the phase-based filters for vizualisation purposes
        phase_multiplied = local_energy_img * local_phase_img * phase_symmetry_img
        
        # Normalize the images to uint8
        gaussian_img = cv2.normalize(gaussian_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        mask = cv2.normalize(mask.astype(np.uint8), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        laplacian = cv2.normalize(laplacian, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        shadow_image = cv2.normalize(shadow_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        local_energy_img = cv2.normalize(local_energy_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        local_phase_img = cv2.normalize(local_phase_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        phase_symmetry_img = cv2.normalize(phase_symmetry_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        ibs = cv2.normalize(ibs, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        prob_map_img = cv2.normalize(prob_map_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        phase_multiplied = cv2.normalize(phase_multiplied, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        
        return gaussian_img, mask, laplacian, shadow_image, local_energy_img, local_phase_img, phase_symmetry_img, ibs, prob_map_img, phase_multiplied
    