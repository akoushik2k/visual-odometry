import cv2 as cv
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

'''
Class to implement visual odometry using stereo images
'''
class VisualOdometry:
    
    #initialize the class
    def __init__(self):
        # path to the images and calibration file
        self.image_path = os.path.dirname(os.path.realpath(__file__))
        self.cali_file = self.image_path + '/zed_300/calib.txt'
        self.gt_file = self.image_path + '/00/00.txt'
        
        # load the images
        self.left_images, self.right_images = self.load_images()
        
        # read the calibration file
        self.l_K, self.l_translation, self.r_K, self.r_translation, = self.read_calibration()

        
        self.curr_left_img = None
        self.curr_right_img = None
        self.curr_disparity = None
        self.next_left_img = None
        self.next_right_img = None
        self.next_disparity = None
        self.homo_matrix = np.eye(4)
        self.trajectory =[]
        self.ground_truth = []
        
        # read ground truth
        self.read_gt()

        self.ground_truth = np.array(self.ground_truth)
        
         
    # decompose projection matrix
    def decomposition(self,p):
        intrinsic_matrix, rotation_matrix, translation_vector, _, _, _, _ = cv.decomposeProjectionMatrix(
            p)


        translation_vector = (translation_vector / translation_vector[3])[:3]

        return intrinsic_matrix, rotation_matrix, translation_vector

    # read ground truth data
    def read_gt(self):
        gt = open(self.gt_file, 'r')
        gt_data = gt.readlines()
        gt.close()
        
        for line in gt_data:
            matrix = np.array(line.split()).reshape(3, 4).astype(float)
            translation = matrix[:, 3]
            
            self.ground_truth.append([translation[0], translation[2]])
            
    # read calibration data
    def read_calibration(self):
        cali = open(self.cali_file, 'r')
        cali_data = cali.readlines()
        cali.close()
        
        P0 = cali_data[0].split()
        
        P0 = np.array(P0[1:]).reshape(3,4).astype(float)
        l_K, l_rotation, l_translation = self.decomposition(P0)

        
        P1 = cali_data[1].split()
        P1 = np.array(P1[1:]).reshape(3,4).astype(float)
        r_K, r_rotation, r_translation = self.decomposition(P1)

       
        return l_K, l_translation, r_K, r_translation
        
    # load images
    def load_images(self):
        left_images = glob.glob(self.image_path + '/zed_300/image_0/*.png')
        right_images = glob.glob(self.image_path + '/zed_300/image_1/*.png')
        
        left_images.sort()
        right_images.sort()
        
        return left_images, right_images
    # extract features from the images
    def extract_features(self, img):

        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)

        img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
        
        return kp,des, img2
    
    # get disparity map
    def get_disparity(self, left_img, right_img):
        num_disparities = 9*16
        block_size = 5


        matcher = cv.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1=8  * block_size ** 2,
                                        P2=32  * block_size ** 2,
                                        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
                                        )

        left_image_disparity_map = matcher.compute(
            left_img, right_img).astype(np.float32)/16
        
        return left_image_disparity_map
    
    # match features
    def match_features(self, curr_left_img,next_left_img):
        #extract features
        curr_kp,curr_des,curr_img = self.extract_features(curr_left_img)
        next_kp,next_des,next_img = self.extract_features(next_left_img)
        
        
        #match features
        bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    
        matches = bf.knnMatch(curr_des, next_des, k=2)
              
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.25*n.distance:
                good.append(m)
        
        return  good, curr_kp, next_kp
    
    # get depth map
    def get_depth(self, disparity, l_K, l_translation, r_K, r_translation):
  
        baseline = r_translation[0] - l_translation[0]
        focal_length = l_K[0,0]
        print(focal_length, baseline)

        disparity[disparity == 0.0] = 0.1
        disparity[disparity == -1.0] = 0.1
        depth_map = np.ones(disparity.shape)
        depth_map = (focal_length * baseline) / disparity
        
        return depth_map
    
    # calculate the pose
    def get_pose(self, firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, matches,depth):
        max_depth = 1000    
        rotation_matrix = np.eye(3)
        translation_vector = np.zeros((3, 1))

        image1_points = np.float32(
                [firstImage_keypoints[m.queryIdx].pt for m in matches])
        image2_points = np.float32(
                [secondImage_keypoints[m.trainIdx].pt for m in matches])

        image1_points = image1_points.reshape(-1, 2)
        image2_points = image2_points.reshape(-1, 2)

        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]

        points_3D = np.zeros((0, 3))
        outliers = []

        for indices, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]

            if z > max_depth:
                outliers.append(indices)
                continue

            x = z*(u-cx)/fx
            y = z*(v-cy)/fy

            points_3D = np.vstack([points_3D, np.array([x, y, z])])

        image1_points = np.delete(image1_points, outliers, 0)
        image2_points = np.delete(image2_points, outliers, 0)

        _, rvec, translation_vector, _ = cv.solvePnPRansac(
            points_3D, image2_points, intrinsic_matrix, None)

        rotation_matrix = cv.Rodrigues(rvec)[0]

        return rotation_matrix, translation_vector, image1_points, image2_points

    
    
    def main(self):
        
        fig = plt.figure()
        viewer = fig.add_subplot(221)
        v2 = fig.add_subplot(222)

        v5 = fig.add_subplot(212)
        v5.set_xlabel('X')
        v5.set_ylabel('Y')
        v5.set_title('Trajectory')

        v5.set_xlim(-1, 1)
        v5.set_ylim(-1, 4)
        trajectory = np.zeros((len(self.left_images), 3, 4))
        trajectory[0] = self.homo_matrix[:3, :]        

        trajectory_plot, = v5.plot([], [], 'bo')
        left_rect = [0.999795, -0.000974, 0.020200,
                    0.000994, 0.999999, -0.000956,
                    -0.020199, 0.000975, 0.999796]
        
        right_rect = [1.000000, -0.000093, -0.000426,
                    0.000093, 1.000000, 0.000966,
                    0.000425, -0.000966, 0.999999]
        
        left_dist = [-0.160933, 0.010858, 0.001155, 0.000977, 0.000000]
        right_dist  = [-0.163547, 0.022941, 0.000450, 0.000903, 0.000000]


        left_dist = np.array(left_dist)
        right_dist = np.array(right_dist)
    
        
        left_rect =  np.array(left_rect).reshape((3, 3)).astype(np.float32)
        right_rect = np.array(right_rect).reshape((3,3)).astype(np.float32)
        
        
        
        plt.ion()
        fig.show() 
        
        features = []
        matches_in_frames = []
        pose_error = []

        for i in range(len(self.left_images)-1):
            
            self.curr_left_img = cv.imread(self.left_images[i], cv.IMREAD_GRAYSCALE)
            self.curr_right_img = cv.imread(self.right_images[i], cv.IMREAD_GRAYSCALE)
            self.next_left_img = cv.imread(self.left_images[i+1], cv.IMREAD_GRAYSCALE)
            self.next_right_img = cv.imread(self.right_images[i+1], cv.IMREAD_GRAYSCALE)
            
            # undistort the images
            self.curr_left_img = cv.undistort(self.curr_left_img, self.l_K, left_dist, None, self.l_K)
            self.curr_right_img = cv.undistort(self.curr_right_img, self.r_K, right_dist, None, self.r_K)
            self.next_left_img = cv.undistort(self.next_left_img, self.l_K, left_dist, None, self.l_K)
            self.next_right_img = cv.undistort(self.next_right_img, self.r_K, right_dist, None, self.r_K)
            

            disparity = self.get_disparity(self.next_left_img,self.next_right_img)
            
            depth = self.get_depth(disparity, self.l_K, self.l_translation, self.r_K, self.r_translation)
            
            matches,curr_pts, next_pts = self.match_features(self.curr_left_img,self.next_left_img)
            
            
            rot,trans,_,_  = self.get_pose(curr_pts, next_pts, self.l_K, matches,depth)  
            
            features.append(len(curr_pts))
            matches_in_frames.append(len(matches))
            
            new_mat = np.eye(4)
            new_mat[:3,:3] = rot
            new_mat[:3,3] = trans.T
            
            
            self.homo_matrix = self.homo_matrix.dot(np.linalg.inv(new_mat))       
            
            # create a graph and plot the x,y of homo matrix in realtime
            x = self.homo_matrix[0, 3]
            y = self.homo_matrix[2, 3]
            # print(self.homo_matrix)
            trajectory[i+1, :, :] = self.homo_matrix[:3, :]
           

            # Adjust plot limits if needed
            v5.relim()
            v5.autoscale_view()
            
            self.trajectory.append([x,y])

            
            
            viewer.clear() # Clears the curious image
            v2.clear()

            viewer.imshow(disparity, cmap='gray') # Loads the new image
            v2.imshow(self.curr_left_img)

            
            trajectory_plot.set_data(*zip(*self.trajectory))
    

            pose_error.append(np.linalg.norm(self.ground_truth[i] - [x,y]))

          

            plt.pause(0.01) # Pauses for 0.1 seconds (necessary for the plot to update
            fig.canvas.draw() # Draws the image to the screen
            


        # plot the features
        print(features)
        # Create a single figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))

        # Plot features
        axes[0].plot(features)
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Number of features')
        axes[0].set_title('Number of features detected in each frame')

        # Plot matches
        axes[1].plot(matches_in_frames)
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('Number of Matches')
        axes[1].set_title('Number of Matches detected in each frame')

        # Plot pose error
        axes[2].plot(pose_error)
        axes[2].set_xlabel('Frame')
        axes[2].set_ylabel('Pose Error')
        axes[2].set_title('Pose Error in each frame')

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show(block=True)  # Add block=True to keep the plot window open
        
                            
        
            
            
        

if __name__ == '__main__':
    vo = VisualOdometry()
    vo.main()

    