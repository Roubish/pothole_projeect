# This is the pothole project 
To enviroment to run the project reuqires python 3.8 and yolov5.

there are two files to be exuted one for image other for videos.

#####################################################################
# To run the image file pothole_main_image.py

pass the source file inside pothole_main_image.py for main(img_path="./yolov5/image-rode2.jpg"

pass your ROI inside pothole_main_image.py line no. 115

eg:..,, point=np.array([[8,7],[7,322],[262,322],[576,327],[585,17],[280,8]])

press 'q' to terminate the execution 

######################################################################
# To run the video file pothole_main_video.py

pass the source file inside pothole_main_video.py for main(img_path="./yolov5/image-rode2.jpg"

pass your ROI inside pothole_main_video.py line no. 145

eg:..,, point=np.array([[8,7],[7,322],[262,322],[576,327],[585,17],[280,8]])


python3 pothole_main_video.py | tee result_pothole-5.txt
