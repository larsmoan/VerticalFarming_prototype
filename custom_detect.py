#Aims to do the detection itself without using opt and argparser

""" import sys
sys.path.append('/home/lars/Documents/Internship.Picking/src/picker/src/picker/yolov7') """



import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import os




def custom_detect(weights, source, conf_tresh=0.7, iou_tresh= 0.45, view_img=False, save_txt=False, save_img=False, save_conf=False, exist_ok=False, trace=False, imgsz=640, augment=False):
    detections = []

    # Directories - below method is standard for yolov7
    current_path = os.getcwd()

    save_dir = Path(increment_path(Path(str(current_path) + '/src/picker/src/picker/runs/') / 'exp', exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        

    set_logging()
    device = select_device('cpu')


    #Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    
    if trace:
        model = TracedModel(model, device, imgsz)
    
    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    #Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Running inference
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment)[0]


        # Apply NMS
        # This should perhaps take the classes, but as default in nms classes=None?, perhaps use classes = 5?
        pred = non_max_suppression(pred, conf_tresh, iou_tresh)
        #print(pred[0])
        t2 = time_synchronized()

        # Here u can add classifier if you want

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #Appending the detections to the list
                    
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    class_num = int(cls)
                    label = f'{names[int(cls)]} {conf:.2f}'

                    skippable = [1,2,3] #Allows to skip ripe, unripe and toddler predictions with two low confidence
                    # Fix for allowing lower confidence with regards to peduncles - in the future with better model. This code can be removed
                    if class_num in skippable and conf < 0.6:   #Removing low confidence ripe detections
                        continue
                
                    detections.append([c1,c2, class_num, float(f'{conf:.2f}'), label])

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(0)  # 1 millisecond
                cv2.destroyAllWindows()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    
    #print(detections)
    return detections


def parse_detections(detections, conf_tresh_ripe):

    ripe = []
    peduncle = []

    for det in detections:
        if det[2] == 2 and det[3] > conf_tresh_ripe:     # ripe
            ripe.append(det)
        elif det[2] == 1:   # peduncle
            peduncle.append(det)


    return ripe, peduncle
    # should return only the coordinates we want to             



if __name__ == '__main__':

    current_path = os.getcwd()

    custom_detect(str(current_path) + '/yolov7_strawberry.pt', str(current_path) + '/../multi_straw.jpg', 0.2, 0.45, True, False, True)
    