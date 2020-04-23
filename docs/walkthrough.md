# MIMAMO-Net Python Library

Python implementation of client side functions for interacting with the MIMAMO-Net.

## Usage
With inputs of video snippets of a face emotion, process the video with MIMAMO-Net and gives a result of an array (what is the name/term of that result?)
1. Parse the video file with OpenFace
2. Parse the cropped and aligned face images with Resnet 50 feature extraction
3. Use the Image sampler to create face snippets
4. more explanations
5. etc


## Methods
### Component - Image Sampler
`Aff-wild-exps/dataloader.py`
#### *class* `Face_Dataset`
  **Arguments**:
  - root_path
    - type: `str`
    - the path to root dir (just an example, explain more on the purpose of this dir)
  - feature_path
  - annot_dir
  - video_name_list
  - label_name
  - py_level=4
  - py_nbands=2
  - test_mode=False
    - type: `bool`
    - explains why the `False` flag is default and what happens when it's changed to `True`
  - num_phase=12
  - phase_size = 48
  - length=64
  - stride=32
  - return_phase=False
  
  **Functionality**:
  - Parses videos from a directory and return a list (?) (I don't fully know what is the return type of the class `Face_Dataset`)

  **expected returned object**: `iterable` object of lists
  ```
  [[phase_batch_0,phase_batch_1], np.array(imgs), np.array(seq_labels), np.array([start, end]), video_record.video]
  ```
  - [phase_batch_0,phase_batch_1]
    - explanation
  - np.array(imgs)
    - explanation
    - imgs
      - explanation
  - np.array(seq_labels)
  - np.array([start, end])
    - start
    - end
  - video_record.video

  **example usage**: (I don't think this is sufficient explanation of what the code does, need help on this part)
  ```
  # instantiate a Face_Dataset object
  train_dataset = Face_Dataset(root_path, feature_path, annot_dir, video_names, label_name='arousal_valence',  num_phase=12 , phase_size=48, test_mode=True)
  # explanation here
  train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size = 4, 
    num_workers=0, pin_memory=False )
  ```

**example returned object**:
  ```
[
    [
        value1,
        value2
    ],
    object,
    object,
    object
]
  ```  

### Component - Phase difference extraction
`/Aff-wild-exps/utils.py`
#### *class* `Steerable_Pyramid_Phase`
  **Arguments**:
  - example
    - type: `int`
    - (just an example)
  
  **Functionality**:
  - Example (needs input)

  **expected returned object**: `iterable` object of lists (just example)
  ```
  [[phase_batch_0,phase_batch_1], np.array(imgs), np.array(seq_labels), np.array([start, end]), video_record.video]
  ```
  - [phase_batch_0,phase_batch_1]
    - explanation
  - np.array(imgs)
    - explanation
    - imgs
      - explanation
  - np.array(seq_labels)
  - np.array([start, end])
    - start
    - end
  - video_record.video

  **example usage**: (I don't think this is sufficient explanation of what the code does, need help on this part)
  ```
  # objective of this code snipper
  print("hello world")
  ```

**example returned object**:
  ```
[
    [
        value1,
        value2
    ],
    object,
    object,
    object
]
  ```  
