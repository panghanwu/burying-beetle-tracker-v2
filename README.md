# Beetle Tracker v2
Tracker for bury beetle experiments, developed by [Prof. Sheng-Feng Shen's lab](https://brcasesslab.wordpress.com/?fbclid=IwAR1rVhSXoyqQX1BemOx3E894xfAkQ4PDnang-AHWtWuX_iADrF81pH-uwIU). 

![](https://github.com/panghanwu/burying-beetle-tracker-v2/blob/main/materials/demo.gif?raw=true)

## Usage
### 1. Preparation
(1) Download `pth` files
Download the `pth` files from the [link](https://drive.google.com/drive/folders/1QmGPgLXu0AcuZL2yG_U1ljU8w8BZiPIq?usp=sharing), and put the three `pth` files in `data` directory.

(2) Create video dictionary
Create `json` file for target videos as following:

```
"group 1":[
    "/group1_video1_path",
    "/group1_video2_path", ...
]
"group 2":[
    "/group2_video1_path",
    "/group2_video2_path", ...
]
...
```
- Saving groups and paths of videos in parent-child hierarchical structure
- Sorting the paths of videos according to time in each group

### 2. Configuration
Check the `yaml` configuration file (default is `config.yml`):

```
device: 0  # Device to use, "cpu" or cuda number.
mode: "class"  # Mode of tracking, "class" or "id".
batch_size: 48  # Batch size for YOLO model.
center_crop_4x3: True  # Crop frame to 4:3.
mouse_frequency: 30  # Update area of mouse body once per N frames.
video_dict: "data/video-dict.json"  # Path of video dictionary create in the preparation.
output_dir: "out-dir"  # Output directory.

classes:  # Classes, id: name。
    0: "H"
    1: "O"
    2: "X"
    3: "nn"
    4: "ss"
    5: "xx"
```

### 3. Run
Rue `tracker.py` by the command line:

```
python tracker.py config.yml
```
The third command refers to the path of configuration, and can be customized.

### Output

(1) Directory
```
out-dir
    ├─ group 1
        ├─ video1.json
        ├─ video2.json
        ├─ ...
    ├─ group 2
        ├─ video1.json
        ├─ video2.json
        ├─ ...
```

(2) JSON file (format of mode "class")
- `Video`: information of video
  - `Path`
  - `Crop 4x3`
  - `Width`: width of frame
  - `Height`: height of frame
  - `Length`: total frames
  - `FPS` 
- `Tracks`: tracks in List
  - `Label IDs`
  - `Labels`
  - `Boxes`: coordinates in (left, top, right, bottom)
  - `Label Scores`: scores by classifier
  - `IoU Scores`: IoU from previouse and this frame
  - `On Mouse`: whether on mouse

## Training
### 1. Preparation
#### YOLOv4 (beetle detector)
Prepare a data list `txt` file for training as following:
`img_path class_id,left,top,right,bottom class_id,left,top,right,bottom ...`

For example:
```
/0001.jpg 1,0.597917,0.352778,0.120833,0.097222
/0002.jpg 2,0.167708,0.515972,0.077083,0.079167
/0003.jpg 0,0.505729,0.972222,0.076042,0.055556 1,0.174479,0.527083,0.101042,0.079167
...
```

More detials can refer to [documentation](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) of Pytorch YOLO in `1.2 Create Labels`.

#### ResNet (mark classifier)
(1) Prepare directories named by class IDs for the beetle images as following:
```
data_dir
    ├─ -1
    ├─ 0
    ├─ 1
    ├─ 2
    ...
anothor_data_dir
    ├─ -1
    ├─ 0
    ├─ 1
    ├─ 2
    ...
...
```
`-1` directories are for unknow beetles.

(2) Prepare data list `txt` file for training as following:
```
/path/data_dir
/path2/anothor_data_dir
...
```

#### UNet (mouse segmenter)
(1) Prepare directories named by class IDs for the beetle images as following:
```
data_dir
    ├─ img/
    │   ├─ 001.jpg
    │   ├─ 002.jpg
    │   ...
    └─ mask/
        ├─ 001.jpg
        ├─ 002.jpg
        ...
```
In the data directory, there are a `img` and a `mask` directory to store images and their mask pairs in the same names.

(2) Prepare data list `txt` file for training as following:
```
/data_dir/img/001.jpg
/data_dir/img/002.jpg
/data_dir2/img/001.jpg
...
```
List of paths of images in `img` only.

### 2. Configuration
Check the `yaml` configuration file for models.
`Detector` for YOLOv4.
`Classifier` for ResNet.
`Segmenter` for UNet.

### 3. Training
Run commend line:
```
python train.py model-name config.yml comment(optional)
```

Replace `model-name` by:
  - `detector` for YOLOv4
  - `classifier` for ResNet
  - `segmenter` for UNet

### Assessment
You can run `assess.py` for simple assessment of models.

```
python assess.py model-name config.yml comment(optional)
```
