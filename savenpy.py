import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

pipeline = rs.pipeline()

#Create a config并配置要流​​式传输的管道
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # 【Skip 5 first frames to give the Auto-Exposure time to adjust 跳过前5帧以设置自动曝光时间】
# for x in range(5):
#     pipe.wait_for_frames()

# # 【Store next frameset for later processing: 存储下一个框架集以供以后处理：】
# frameset = pipe.wait_for_frames()
# depth_frame = frameset.get_depth_frame()


profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

# 创建文件夹
save_path = './save'
os.mkdir(save_path)
os.mkdir(os.path.join(save_path, "color"))
os.mkdir(os.path.join(save_path, "depth"))

# 保存的图片和实时的图片界面
#cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE)
#cv2.namedWindow("save", cv2.WINDOW_AUTOSIZE)
saved_color_image = None # 保存的临时图片
saved_depth_mapped_image = None
saved_count = 0

# 主循环
try:
    while True:
        
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_data = np.asanyarray(aligned_depth_frame.get_data(), dtype="float32")
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #cv2.imshow("live", np.hstack((color_image, depth_mapped_image)))
        key = cv2.waitKey(30)

        # s 保存图片
        if key & 0xFF == ord('s'):
            saved_color_image = color_image
            saved_depth_mapped_image = depth_mapped_image

            # 彩色图片保存为png格式
            cv2.imwrite(os.path.join((save_path), "color", "{}.png".format(saved_count)), saved_color_image)
            # 深度信息由采集到的float16直接保存为npy格式
            np.save(os.path.join((save_path), "depth", "{}".format(saved_count)), depth_data)
            #saved_count=0
            cv2.imshow("save", np.hstack((saved_color_image, saved_depth_mapped_image)))

        # # q 退出
        # if key & 0xFF == ord('q') or key == 27:
        #     cv2.destroyAllWindows()
        #     break    
finally:
    pipeline.stop()