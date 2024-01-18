from torch.utils.data import DataLoader, Dataset, IterableDataset
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from threading import Thread, Lock
from Datasets.ros_classes import image_converter
import cv2
import numpy as np
from .utils import make_intrinsics_layer
import torch
import time

class ROSLoadStream(IterableDataset):
    def __init__(self, stream_source, transform = None, time_out = 2, batch_size=1, num_workers=1, focalx = 480.0, focaly = 480.0, centerx = 480.0, centery = 270.0):
        self.stream_source = stream_source
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.time_out = time_out
        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

        self.img1 = self.stream_source.cv_image_queue.get()
        self.img2 = self.stream_source.cv_image_queue.get()
        self.odom = self.stream_source.odom_queue.get()
        self.image_time = self.stream_source.odom_time_queue.get() # timestamp of img2
        self.thread = Thread(target=self.update, args=([stream_source]), daemon=True)
        self.thread.start()
        self.lock = Lock()
        self.lock_extra = Lock()
        time.sleep(0.5) # wait for the thread to start

    
    def update(self, stream_source):
        is_ros = isinstance(stream_source, image_converter)
        while True:            
            my_im = stream_source.cv_image_queue.get()
            my_im = cv2.resize(my_im, (0,0), fx=1., fy=1.)
            my_time = stream_source.odom_time_queue.get()
            my_odom = stream_source.odom_queue.get()
            orient = np.array([my_odom.pose.pose.orientation.x, my_odom.pose.pose.orientation.y, my_odom.pose.pose.orientation.z]) # this is a hack, our odometry is roll, pitch, yaw, but ROS message is quaternion
            position = np.array([my_odom.pose.pose.position.x, my_odom.pose.pose.position.y, my_odom.pose.pose.position.z])
            if my_time != self.image_time:
                self.img2 = my_im
                self.image_time = my_time
                self.odom = np.hstack((position, orient))
                self.lock.acquire()
            else:    
                print('skip frame bc time stampt is the same as the previous one')
                if not self.lock.locked():
                    self.lock.acquire()
            if self.lock_extra.locked():
                self.lock_extra.release()
            
    def __iter__(self):
        return iter([self.img1, self.img2])
    
    def __next__(self):

        lock_success = self.lock_extra.acquire(timeout=self.time_out)
        if not lock_success:
            raise StopIteration
        
        if not self.thread.is_alive() or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        
        else:
            # im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            res = {'img1': self.img1, 'img2': self.img2}

            h, w, _ = self.img1.shape
            intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
            res['intrinsic'] = intrinsicLayer

            if self.transform:
                res = self.transform(res)

            res['img1'] = res['img1'].unsqueeze(0)
            res['img2'] = res['img2'].unsqueeze(0)
            res['intrinsic'] = res['intrinsic'].unsqueeze(0)
            # res['img1_raw'] = self.img1
            # res['img2_raw'] = self.img2
            #conver self.img2 to tensor and add new dimension at 0
            res['img2_raw'] = torch.from_numpy(self.img1).unsqueeze(0)
            res['img1_raw'] = torch.from_numpy(self.img2).unsqueeze(0)
            res['odom'] = self.odom
            self.img1 = self.img2
            self.lock.release()
            return res

    def __len__(self):
        return len(self.stream)

