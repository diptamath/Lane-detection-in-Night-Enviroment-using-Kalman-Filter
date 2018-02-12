# Lane-detection-in-Night-Enviroment-using-Kalman-Filter

An important milestone in a computer vision approach to autonomous vehicles is finding lane markings on the road. Here, we describe a process to detect lane in night environment.

###Challenges
· low light intensity
· difficult to tune the parameters for various light intensity
· poor edge detection
· Shadows, Sudden high intensity car headlights

##Our Approach
Our method is described in those Steps:
1. We perform gamma correction for each video frame to set the light intensity
2. Region of Interest is cropped from the image, so that we can only look for lanes on ROI part only. It helps to reduce the computational cost and increase the fps.
3. Bilateral filter is applied to cancel out the noise and smoothing the video frames but preserving edges.
4. HSV filter is applied to create a mask for pixels within fixed range
5. After those preprocessing (gamma correction and filtering), we detect edges using Canny edge detector.
6. After that, Hough transformation is used for detection of lines using the edges from the previous step.
7. Detected lines are clustered using DBSCAN as we want only lines across lanes only.
8. Kalman Filtering is applied for better lane detection. Here, we applied as a linear estimator for the clustered lines for the lanes to make it stable and free from any offset errors.  

###Dependencies
Python 3.5
Numpy
OpenCV-Python
Matplotlib
sklearn
Scipy

####Gamma Correction
Here, Gamma Correction is to set the intensity values. It uses a parameter to tune it and also set the intensity level. Basics of gamma correction is here.
 
 
Code:-
def gamma_correction(RGBimage, correct_param = 0.35,equalizeHist = False):
    red = RGBimage[:,:,2]
    green = RGBimage[:,:,1]
    blue = RGBimage[:,:,0]
    
    red = red/255.0
    red = cv2.pow(red, correct_param)
    red = np.uint8(red*255)
    if equalizeHist:
        red = cv2.equalizeHist(red)
    
    green = green/255.0
    green = cv2.pow(green, correct_param)
    green = np.uint8(green*255)
    if equalizeHist:
        green = cv2.equalizeHist(green)
        
    
    blue = blue/255.0
    blue = cv2.pow(blue, correct_param)
    blue = np.uint8(blue*255)
    if equalizeHist:
        blue = cv2.equalizeHist(blue)
    
 
    output = cv2.merge((blue,green,red))
    return output
 
####Region of Interest
Using following code cropped the each video frame to consider only portion for lane detection. Due to this, sudden high illumination due to streetlights, other car headlights can be avoided. This also increase the fps a bit.  
Code:-
def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    #channel_count = img.shape[2]
    # color used to fill polygon
    match_mask_color = 255
    # Fill the polygon with white
    cv2.fillPoly(mask, vertices, (255,255,255))
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
 

####Bilateral filter
A bilateral filter is a non-linear, edge-preserving, and noise-reducing smoothing filter for images. It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels.

####Hough transformation
In this section, we used Hough transformation and some modification to remove horizontal detected lines and lines only oriented to the lane.
Code:
def hough_transform(original, gray_img, threshold, discard_horizontal = 0.4):
    """
    A function fitting lines that intersect >=threshold white pixels
    Input:
    - original - image we want to draw lines on
    - gray_img - image with white/black pixels, e.g. a result of Canny Edge Detection
    - threshold - if a line intersects more than threshold white pixels, draw it
    - discard_horizontal - smallest abs derivative of line that we want to take into account
    Return:
    - image_lines - result of applying the function
    - lines_ok - rho and theta
    """
    lines = cv2.HoughLines(gray_img, 0.5, np.pi / 360, threshold)
    image_lines = original
    lines_ok = [] #list of parameters of lines that we want to take into account (not horizontal)
            
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            #discard horizontal lines
            m = -math.cos(theta)/(math.sin(theta)+1e-10) #adding some small value to avoid dividing by 0
            if abs(m) < discard_horizontal:
                continue
            else:
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(image_lines, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
                lines_ok.append([rho,theta])
        
    lines_ok = np.array(lines_ok)
                    
    return image_lines, lines_ok
  
####Kalman Filter
The Kalman filter is an algorithm that uses noisy observations of a system over time to estimate the parameters of the system (some of which are unobservable) and predict future observations. At each time step, it makes a prediction, takes in a measurement, and updates itself based on how the prediction and measurement compare.More information about it can be found here.

Here, LaneTracker class implements the Kalman filter for lane detection. Firstly, it initializes the State matrix & Measurement matrix size. Then, it calculated transition matrix. We take White Gaussian noise for our system. Using this noise model, we have calculated the error for state and used in estimator to generate the predicted state using that measurement noise. Variation of detected lines along the lanes are averaged out by the kalman filter by adding up the measurement error and previous state. That’s why detected lane marker lines are stable over time and for its predictor property from previous state,  at very low illumination condition, it can be able to detect lanes by remembering previous detected lanes from the previous video frame.
Code:
class LaneTracker: 
    def __init__(self, n_lanes, proc_noise_scale, meas_noise_scale, process_cov_parallel=0, proc_noise_type='white'):
        self.n_lanes = n_lanes
        self.meas_size = 4 * self.n_lanes
        self.state_size = self.meas_size * 2
        self.contr_size = 0
 
        self.kf = cv2.KalmanFilter(self.state_size, self.meas_size, self.contr_size)
        self.kf.transitionMatrix = np.eye(self.state_size, dtype=np.float32)
        self.kf.measurementMatrix = np.zeros((self.meas_size, self.state_size), np.float32)
        for i in range(self.meas_size):
            self.kf.measurementMatrix[i, i*2] = 1
 
        if proc_noise_type == 'white':
            block = np.matrix([[0.25, 0.5],
                               [0.5, 1.]], dtype=np.float32)
            self.kf.processNoiseCov = block_diag(*([block] * self.meas_size)) * proc_noise_scale
        if proc_noise_type == 'identity':
            self.kf.processNoiseCov = np.eye(self.state_size, dtype=np.float32) * proc_noise_scale
        for i in range(0, self.meas_size, 2):
            for j in range(1, self.n_lanes):
                self.kf.processNoiseCov[i, i+(j*8)] = process_cov_parallel
                self.kf.processNoiseCov[i+(j*8), i] = process_cov_parallel
 
        self.kf.measurementNoiseCov = np.eye(self.meas_size, dtype=np.float32) * meas_noise_scale
 
        self.kf.errorCovPre = np.eye(self.state_size)
 
        self.meas = np.zeros((self.meas_size, 1), np.float32)
        self.state = np.zeros((self.state_size, 1), np.float32)
 
        self.first_detected = False
 
    def _update_dt(self, dt):
        for i in range(0, self.state_size, 2):
            self.kf.transitionMatrix[i, i+1] = dt
 
    def _first_detect(self, lanes):
        for l, i in zip(lanes, range(0, self.state_size, 8)):
            self.state[i:i+8:2, 0] = l
        self.kf.statePost = self.state
        self.first_detected = True
 
    def update(self, lanes):
        if self.first_detected:
            for l, i in zip(lanes, range(0, self.meas_size, 4)):
                if l is not None:
                    self.meas[i:i+4, 0] = l
            self.kf.correct(self.meas)
        else:
            if lanes.count(None) == 0:
                self._first_detect(lanes)
 
    def predict(self, dt):
        if self.first_detected:
            self._update_dt(dt)
            state = self.kf.predict()
            lanes = []
            for i in range(0, len(state), 8):
                lanes.append((state[i], state[i+2], state[i+4], state[i+6]))
            return lanes
        else:
            return None
 
 
##Conclusion:
For our dataset, this algorithm perform quite. For very low intensity lights it can be able to detect lanes using Kalman Filter. Detected lane markers are quite stable and robust to intensity and noise variation.
 
                                                             
