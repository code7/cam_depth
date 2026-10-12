import os
import sys
import cv2
import queue
import numpy as np

import vcamera as vcam

frame_queue = queue.Queue(maxsize=1)

def frame_callback(frame_set):
    if not frame_queue.full():
        frame_queue.put_nowait(frame_set)

def main():
    try:
        vcam.CameraUtils.Init(True)
        cam_infos = vcam.CameraUtils.DiscoverCameras()
        
        if len(cam_infos) == 0:
            print('No cameras found!')
            return  # 直接返回退出
        
        cam = vcam.CameraFactory.GetCameraBySerialNumber(cam_infos[0].serial_number)
        
        status = cam.Connect()
        if not status:
            print("Failed to connect camera: ", status.message())
            return
        
        acq_mode, status = cam.GetFeature('AcquisitionMode')
        if not status:
            print("Failed to get AcquisitionMode: ", status.message())
            cam.Disconnect()
            return
        
        #0：SingleFrame
        #1：MultiFrame
        #2：Continuous
        status = acq_mode.SetValue(2)
        if not status:
            print("Failed to set Continuous AcquisitionMode: ", status.message())
            cam.Disconnect()
            return
        
        status = cam.SetSensorEnabled(vcam.SensorType.Depth, True)
        if not status:
            print("Failed to enable depth sensor")
            cam.Disconnect()
            return
        
        cam.RegisterFrameSetCallback(frame_callback)
        
        if not cam.StartCapture():
            print("Failed to start capture")
            cam.Disconnect()
            return
        
        # 创建深度图像显示窗口
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                try:
                    fs = frame_queue.get_nowait()
                    depth_image = fs.GetImage(vcam.SensorType.Depth)
                    if depth_image.IsValid():
                        depth_width = depth_image.width()
                        depth_height = depth_image.height()
                        # print(f'Depth frame: {depth_width}x{depth_height}')
                        depth_data = depth_image.to_numpy()
                        # 转换深度图为可显示格式
                        depth_vis = cv2.normalize(
                            depth_data, None, 0, 255,
                            cv2.NORM_MINMAX, dtype=cv2.CV_8U
                        )
                        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        cv2.imshow("Depth", depth_colored)
                except queue.Empty:
                    pass
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
            cam.StopCapture()
            cam.Disconnect()
            print('Camera disconnected.')
            
    except Exception as e:
        print(f"Main error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__=='__main__':
    main()