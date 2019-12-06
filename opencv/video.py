import cv2

#获得视频的格式  
videoCapture = cv2.VideoCapture('./video/14K264.mp4')  
#获得码率及尺寸  
fps = videoCapture.get(cv2.CAP_PROP_FPS)  
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),   
    int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))  

#指定写视频的格式, I420-avi, MJPG-mp4  
videoWriter = cv2.VideoWriter('other.mp4', cv2.VideoWriter_fourcc("X", "V", "T", "D"), fps, size)    
#读帧  
success, frame = videoCapture.read()   
index = 0  
while success:
    cv2.imwrite("tmp/{}.jpg".format(index),frame)  
    #cv2.imshow("Oto Video", frame) #显示  
    #cv2.waitKey(1000/int(fps)) #延迟  
    videoWriter.write(frame) #写视频帧  
    success, frame = videoCapture.read() #获取下一帧 
    index += 1 