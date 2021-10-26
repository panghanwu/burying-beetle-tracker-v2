from tqdm import tqdm
import numpy as np
import json
import yaml
import cv2
import sys
import os

from utils import app, torch_utils, utils


# ---
# python track.py config.yaml
# ---

if len(sys.argv) == 1:
    print('Please give the config path for the tracking mission.')
    sys.exit()

# load configs
with open(sys.argv[1]) as yf:
    configs = yaml.safe_load(yf)

assert configs['mode'] in ['class', 'id']
label_dict = configs['classes']
batch_size = configs['batch_size']

if configs['device'] == 'cpu':
    device = 'cpu'
else:
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(configs['device'])

# load models
detector = app.Detector(configs['Detector'], device)
segmenter = app.Segmenter(configs['Segmenter'], device)

if configs['mode'] == 'class':
    classifier = app.Classifier(len(label_dict), configs['Classifier'], device)
    tracker = app.Tracker(label_dict, configs['Tracker'])
elif configs['mode'] == 'id':
    tracker = app.IDTracker(configs['IDTracker'])

# load video groups
with open(configs['video_dict']) as jf:
    video_dict = json.load(jf)

total_vids = 0
for v in video_dict.values():
    total_vids += len(v)
    
total_t, n_vid = 0, 0
# for each group
for exp, v_list in video_dict.items():
    
    # create output dir
    out_dir = f'{configs["output_dir"]}/{exp}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    tracker.reset()
    
    for v_path in v_list:
        vid = cv2.VideoCapture(v_path)
        vid_name = os.path.basename(v_path)

        total = int(vid.get(7))
        n_vid += 1
        title = f'[{n_vid}/{total_vids}] {exp}: {vid_name}'
        
        if configs['center_crop_4x3']:
            w, h = int(vid.get(4)*4/3), int(vid.get(4))
        else:
            w, h = int(vid.get(3)), int(vid.get(4))
        
        data = {}
        data['Video'] = {
            'Path': v_path,
            'Crop 4x3': configs['center_crop_4x3'],
            'Width': w,
            'Height': h,
            'Length': total,
            'FPS': vid.get(5)
        }
        
        with tqdm(desc=title, total=total, ncols=100, leave=False) as pbar:
            while vid.isOpened():
                
                f_no = int(vid.get(1))
                success, frame = vid.read()
                if not success:
                    break
                
                # crop
                if configs['center_crop_4x3']:
                    frame = torch_utils.center_crop_4x3_numpy(frame, h)
                
                init_batch = f_no%batch_size==0
                full_batch = (f_no+1)%batch_size==0
                
                # segment
                if f_no%configs['mouse_frequency'] == 0:
                    mouse_area = segmenter.cut(frame, keep_size=True)
                    
                if init_batch:
                    batch_frames = []
                
                batch_frames.append(frame)
                
                if full_batch or (f_no+1)==total:
                    # detect
                    batch_boxes, _ = detector.detect_batch(batch_frames)
                    
                    for i in range(len(batch_frames)):
                        boxes = batch_boxes[i].cpu().numpy()
                        
                        if configs['mode'] == 'class':
                            class_codes = []
                            if len(boxes) > 0:
                                crops = torch_utils.crop_array_images(frame, boxes)

                                # classify
                                _ = classifier.mark_batch(crops)
                                class_codes = classifier.raw.cpu().numpy()
                        
                            # track
                            tracker.track(boxes, class_codes)
                        
                        elif configs['mode'] == 'id':
                            tracker.track(boxes)
                        
                        # on mouse
                        boxes =tracker.trace[-1]['Boxes']
                        on_mouse = torch_utils.is_on_mouse(boxes, mouse_area, threshold=0.1)
                        tracker.trace[-1]['On Mouse'] = on_mouse
                    
                    if full_batch:
                        pbar.update(batch_size)
                    else:
                        pbar.update(total%batch_size)
                        
            data['Tracks'] = tracker.trace
            
            json_name = f'{out_dir}/{os.path.splitext(vid_name)[0]}.json'
            with open(json_name, 'w') as jf:
                json.dump(data, jf, indent=4)
                
            tracker.trace = []
            
            t = pbar.format_dict['elapsed']
            total_t += t
            
        print(f'{title} finished. Excution time: {tqdm.format_interval(t)}')
            
print(f'Tracking Completed. Total excution time: {tqdm.format_interval(total_t)}')