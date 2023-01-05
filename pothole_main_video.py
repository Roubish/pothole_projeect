from multiprocessing import parent_process
import torch
import pandas
import cv2
import time
import numpy as np

value=[]
pathole_precentage=[]
def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)  

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    #print(cordinates)
    return labels, cordinates

#plot the BBox and results
def plot_boxes(results, frame,classes):

    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")

    ### looping through the detections
    pathole_aera_value=[]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.1: ### threshold value for detection. We are discarding everything below this value
            # print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            #print(x1, y1, x2, y2)
            x_diff=x2-x1
            y_diff=y2-y1
            area_pathole=x_diff*y_diff
            pathole_aera_value.append(area_pathole)

            text_d = classes[int(labels[i])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
                
            cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

            total_pathole_area=sum(pathole_aera_value)
            # print(total_pathole_area)
    value.append(total_pathole_area)

    return frame,total_pathole_area

#Main function
def main(img_path=None, vid_path=None,vid_out = None):

    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    model =  torch.hub.load('./yolov5', 'custom', source ='local', path='./best.pt',force_reload=True)

    classes = model.names ### class names in string format
    #print(classes)

    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE 

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        frame ,area= plot_boxes(results, frame,classes = classes)

        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## creating a free windown to show the result
        #results.pandas().xyxy[0]

        while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            cv2.imshow("img_only", frame)

    elif vid_path !=None:
        print(f"[INFO] Working with video: {vid_path}")

        ## reading the video
        cap = cv2.VideoCapture(vid_path)


        if vid_out: ### creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        # assert cap.isOpened()
        frame_no = 1
        total_area=0
        area=0
        total=0
        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if frame is None:
                print(' Successful process  ')
                break
            if ret :
                print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                try:
                    results = detectx(frame, model = model)
                    # print(type(results))
                    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                    frame,area = plot_boxes(results, frame,classes = classes)

                    cv2.imshow("vid_out", frame)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                    if vid_out:
                        print(f"[INFO] Saving output video. . . ")
                        out.write(frame)

                except:
                    frame_no -=1
                    pass

                else:
                    # print(total) 
                    pass 
                finally:
                    #print("print")
                    frame_no += 1
                    total_area +=area
                    total = total_area/frame_no
                    #print(total)
                    #print(frame_no)
                    #point=np.array([[567,25],[310,15],[9,14],[29,313],[330,317],[573,321]])
                    # point=np.array([[8,7],[7,322],[262,322],[576,327],[585,17],[280,8]])
                    # point=np.array([[129,92],[420,79],[623,159],[631,352],[4,356],[9,168]])
                    point=np.array([[408,180],[1108,189],[1270,398],[1269,705],[2,668],[162,261]])
                    
                    
                    #[test1.mp4]     [472,285],[756,286],[1091,368],[1249,703],[5,702],[251,349]
                    #pothole-5.mp4    [408,180],[1108,189],[1270,398],[1269,705],[2,668],[162,261]
                    
                    road_area=cv2.contourArea(point)

                    # global total_pathole_area
                    precentage_faulty_road=(int(total)/road_area)
                    faulty_road=round(precentage_faulty_road*100)
                    pathole_precentage.append(faulty_road)
            
                    print("precentage_faulty_road%", faulty_road)  
        
        #print("pathole_precentage",pathole_precentage)
        pathole_precentage_count = len(pathole_precentage)
        print(pathole_precentage_count)
        pathole_precentage_sum = sum(pathole_precentage)
        avg_pathole_precentage = pathole_precentage_sum/pathole_precentage_count
        print("avg_pathole_precentage", int(avg_pathole_precentage), "%")

        print(f"[INFO] Clening up. . . ")
        ### releaseing the writer
        out.release()
        cap.release()
        ## closing all windows
        cv2.destroyAllWindows()

main(vid_path="pothole-5.mp4",vid_out="result_pothole-5.mp4") 
