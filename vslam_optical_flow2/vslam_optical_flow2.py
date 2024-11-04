import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from tf2_ros import TransformBroadcaster
import os
import yaml
import tf_transformations as tf
import time

class CameraViewer(Node):
    def __init__(self):
        super().__init__('vslam_optical_flow2')

        cwd = os.path.dirname(os.path.realpath(__file__))
        cwd = (cwd.split(os.sep)[:-1])
        cwd = (os.sep.join(cwd))

        cwd = cwd + '/../../../../../src/vslam_optical_flow2/params/vslam_params.yaml'

        with open(cwd, 'r') as file:
            self.params = yaml.safe_load(file)

        print(self.params)
        
        self.frame = None
        self.prev_gray = np.array([])
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.quat = [0,0,0,1]
        self.loc = []

        if isinstance(self.params['initial_pose'], str):
            self.inital_pose_sub = self.create_subscription(
                PoseWithCovarianceStamped,
                self.params['initial_pose'],
                self.pose_callback,
                10)
            self.inital_pose_sub
        else:
            self.loc = self.params['initial_pose']
            self.subscription = self.create_subscription(
                Image,
                self.params['image_topic'],
                self.listener_callback,
                10)
            self.subscription

        self.bridge = CvBridge()

    def pose_callback(self, msg):
        _,theta,_ = self.quaternion_to_euler_angle_vectorized(msg.pose.pose.orientation.w,msg.pose.pose.orientation.x,
                                                  msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        self.loc = [msg.pose.pose.position.y, msg.pose.pose.position.x, theta]
        self.destroy_subscription(self.inital_pose_sub)

        self.subscription = self.create_subscription(
            Image,
            self.params['image_topic'],
            self.listener_callback,
            10)
        self.subscription
    
    def euler_to_quaternion(self, yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]
    
    def quaternion_to_euler_angle_vectorized(self, w, x, y, z):
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.degrees(np.arctan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)

        t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
        Y = np.degrees(np.arcsin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.degrees(np.arctan2(t3, t4))

        return X, Y, Z
    
    def create_rotation_matrix(self, yaw):

        roll = 0
        pitch = 0

        Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
        
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
        
        R = np.dot(Rz, np.dot(Ry, Rx))
        
        return R

    def adjust_gamma(self, image, gamma):
        
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)

    def listener_callback(self, msg):
        if self.frame is None:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.frame = self.frame[60:660, 340:940]
            self.prev_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            return
        
        self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.frame = self.frame[60:660, 340:940]

        gamma = np.log(np.mean(self.frame))/np.log(128)

        self.frame = self.adjust_gamma(self.frame, gamma)

        lk_params = dict(winSize=(60, 60), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                   
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        try:
            p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **lk_params)

            self.good_new = p1[st == 1]
            self.good_old = p0[st == 1]

            temp_d1s = []
            temp_d2s = []
            for i, (new, old) in enumerate(zip(self.good_new, self.good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                temp_d1s.append(a - c)
                temp_d2s.append(b - d)

                if 0 > a > 600 or 0 > b > 600 or 0 > c > 600 or 0 > d > 600:
                    return

                self.frame = cv2.arrowedLine(self.frame, (int(c), int(d)), (int(a), int(b)),
                (0, 255, 0), 2)

                self.frame = cv2.circle(self.frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            temp_d1 = np.median(np.array(temp_d1s)) * self.params['calibration_multiplier']
            temp_d2 = np.median(np.array(temp_d2s)) * self.params['calibration_multiplier']

            if np.isnan(temp_d1) or np.isnan(temp_d2):
                print("NaN value occured. Skipping...")
                return

            H, mask = cv2.findHomography(self.good_old, self.good_new, cv2.RANSAC, 5.0)
            theta = np.arctan2(H[1, 0], H[0, 0])
            #print(H[1, 0], H[0, 0])
            self.loc[2] = self.loc[2] + theta
            print(f"Açısal değişim: {np.degrees(self.loc[2])} derece")

            R = self.create_rotation_matrix(self.loc[2])
            V = [temp_d1, temp_d2, 0]
            temp_d1, temp_d2, _ = np.dot(R,V)
        except:
            print("Not enough points to calculate angle")

        
        self.loc[0] = self.loc[0] + temp_d1
        self.loc[1] = self.loc[1] + temp_d2


        self.quat = self.euler_to_quaternion(self.loc[2],0,0)

        broadcaster = TransformBroadcaster(self)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = "map"
        tf_msg.child_frame_id = self.params['tf_frame_name']
        tf_msg.transform.translation.x = self.loc[1] 
        tf_msg.transform.translation.y = self.loc[0] 
        tf_msg.transform.translation.z = 0 + 0.1
        tf_msg.transform.rotation.x = float(self.quat[0])
        tf_msg.transform.rotation.y = float(self.quat[1])
        tf_msg.transform.rotation.z = float(self.quat[2])
        tf_msg.transform.rotation.w = float(self.quat[3])

        broadcaster.sendTransform(tf_msg)
        
        cv2.imshow("Optical Flow Vectors", self.frame)

        self.prev_gray = gray
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    viewer = CameraViewer()
    rclpy.spin(viewer)

    viewer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

