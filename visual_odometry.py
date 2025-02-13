import cv2 as cv
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# class to implement visual odometry
class VisualOdometry:
    
    def __init__(self):
        #current file path
        self.image_path = os.path.dirname(os.path.realpath(__file__))
        
        self.cali_file = self.image_path + '/06/calib.txt'
        self.gt_file = self.image_path + '/06/06.txt'
        
        # load images from the left and right camera
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
        
        # read the ground truth
        self.read_gt()
        
        # convert ground_truth to a np array
        self.ground_truth = np.array(self.ground_truth)
        
        
    # decompose the Porjection Matrix
    def decomposition(self,p):
 
        # Decomposing the projection matrix
        intrinsic_matrix, rotation_matrix, translation_vector, _, _, _, _ = cv.decomposeProjectionMatrix(
            p)

        # Scaling and removing the homogenous coordinates
        translation_vector = (translation_vector / translation_vector[3])[:3]

        return intrinsic_matrix, rotation_matrix, translation_vector

    # read the ground truth data
    def read_gt(self):
        gt = open(self.gt_file, 'r')
        gt_data = gt.readlines()
        gt.close()
        
        for line in gt_data:
            matrix = np.array(line.split()).reshape(3, 4).astype(float)
            translation = matrix[:, 3]
    
            self.ground_truth.append([translation[0], translation[2]])
            
    # read the calibration data
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
        
    #load images from left and right camera
    def load_images(self):
        left_images = glob.glob(self.image_path + '/06/image_0/*.png')
        right_images = glob.glob(self.image_path + '/06/image_1/*.png')
        
        left_images.sort()
        right_images.sort()
        
        return left_images, right_images
    
    # extract features from the image
    def extract_features(self, img):
        #detect features
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        # draw the keypoints
        img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0))
        
        return kp,des, img2
    
    # get the disparity map
    def get_disparity(self, left_img, right_img):

        num_disparities = 2*16
        block_size = 15

        # Using SGBM matcher(Hirschmuller algorithm)
        matcher = cv.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1=8  * block_size ** 2,
                                        P2=32  * block_size ** 2,
                                        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
                                        )
        # Disparity map
        left_image_disparity_map = matcher.compute(
            left_img, right_img).astype(np.float32)/16

        return left_image_disparity_map
    
    # match features between two images
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


        
        return  good, curr_kp, next_kp,curr_img
    
    # get the depth map
    def get_depth(self, disparity, l_K, l_translation, r_K, r_translation):
        #get the depth

        baseline = r_translation[0] - l_translation[0]

        focal_length = l_K[0,0]
      
        disparity[disparity == 0.0] = 0.1
        disparity[disparity == -1.0] = 0.1
        depth_map = np.ones(disparity.shape)
        depth_map = (focal_length * baseline) / disparity
        
        return depth_map
    
    # get the pose of the camera
    def get_pose(self, firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, matches,depth):
        max_depth = 1000    
        rotation_matrix = np.eye(3)
        translation_vector = np.zeros((3, 1))

        # Only considering keypoints that are matched for two sequential frames
        image1_points = np.float32(
                [firstImage_keypoints[m.queryIdx].pt for m in matches])
        image2_points = np.float32(
                [secondImage_keypoints[m.trainIdx].pt for m in matches])
        
        # flatten the image points
        image1_points = image1_points.reshape(-1, 2)
        image2_points = image2_points.reshape(-1, 2)
        
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]

        points_3D = np.zeros((0, 3))
        outliers = []

        # Extract depth information to build 3D positions
        for indices, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]


            # We will not consider depth greater than max_depth
            if z > max_depth:
                outliers.append(indices)
                continue

            # Using z we can find the x,y points in 3D coordinate using the formula
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy

            # Stacking all the 3D (x,y,z) points
            points_3D = np.vstack([points_3D, np.array([x, y, z])])

        # Deleting the false depth points
        image1_points = np.delete(image1_points, outliers, 0)
        image2_points = np.delete(image2_points, outliers, 0)

        # Apply Ransac Algorithm to remove outliers
        _, rvec, translation_vector, _ = cv.solvePnPRansac(
            points_3D, image2_points, intrinsic_matrix, None)

        rotation_matrix = cv.Rodrigues(rvec)[0]

        return rotation_matrix, translation_vector, image1_points, image2_points

    
    # main function to run the visual odometry
    def main(self):
        
        fig = plt.figure()
        viewer = fig.add_subplot(221)
        viewer.set_title('Disparity Map')
        v2 = fig.add_subplot(222)
        v2.set_title('Feature Matching')

        v5 = fig.add_subplot(212)
        v5.set_xlabel('X')
        v5.set_ylabel('Y')
        v5.set_title('Trajectory Plot')

        
        v5.set_xlim(-60, 60)
        v5.set_ylim(-200, 400)
        v5.grid(True)

        v5.legend(['Ground Truth', 'Estimated'])
        
        
        trajectory = np.zeros((len(self.left_images), 3, 4))
        trajectory[0] = self.homo_matrix[:3, :]        
        ground_truth_plot, = v5.plot([], [], 'r-')
        trajectory_plot, = v5.plot([], [], 'b-')
 
        

        
        plt.ion() # Turns interactive mode on (probably unnecessary)
        fig.show() # Initially shows the figure
        
        features = []
        matches_in_frames = []
        pose_error = []

        for i in range(len(self.left_images)-1):
         
            self.curr_left_img = cv.imread(self.left_images[i], 0)
            self.curr_right_img = cv.imread(self.right_images[i], 0)
            self.next_left_img = cv.imread(self.left_images[i+1], 0)
            self.next_right_img = cv.imread(self.right_images[i+1], 0)
            

            disparity = self.get_disparity(self.next_left_img,self.next_right_img)
            
            depth = self.get_depth(disparity, self.l_K, self.l_translation, self.r_K, self.r_translation)
            
            matches,curr_pts, next_pts,features_img = self.match_features(self.curr_left_img,self.next_left_img)
            
            new_image = cv.drawMatches(self.curr_left_img, curr_pts, self.next_left_img, next_pts, matches, None)
            
            features.append(len(curr_pts))
            matches_in_frames.append(len(matches))            
            
            rot,trans,_,_  = self.get_pose(curr_pts, next_pts, self.l_K, matches,depth)  
            
        
            new_mat = np.eye(4)
            new_mat[:3,:3] = rot
            new_mat[:3,3] = trans.T
            
            
            self.homo_matrix = self.homo_matrix.dot(np.linalg.inv(new_mat))       
            
            x = self.homo_matrix[0, 3]
            y = self.homo_matrix[2, 3]

            trajectory[i+1, :, :] = self.homo_matrix[:3, :]

            ground_truth_plot.set_data(self.ground_truth[:i+1, 0], self.ground_truth[:i+1, 1])
            
            
            # Adjust plot limits if needed
            v5.relim()
            v5.legend(['Ground Truth', 'Calculated VO'])
            v5.autoscale_view()
            
            self.trajectory.append([x,y])

            
            viewer.clear() # Clears the curious image
            v2.clear()
            
            viewer.imshow(disparity, cmap='gray') # Loads the new image
            v2.imshow(new_image, cmap="gray")

            
            trajectory_plot.set_data(*zip(*self.trajectory))


            plt.pause(0.01) # Pauses for 0.1 seconds (necessary for the plot to update
            fig.canvas.draw() # Draws the image to the screen
            
            # calculate the error between the ground truth and the estimated trajectory
            pose_error.append(np.linalg.norm(self.ground_truth[i] - [x,y]))


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
        

        print('Done')
        
            
            
        

if __name__ == '__main__':
    vo = VisualOdometry()
    vo.main()

    