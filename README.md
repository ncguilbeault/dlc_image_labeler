# DeepLabCut (DLC) Image Labeller

This is a custom python application for manually labelling keypoints in video for DeepLabCut (DLC). I built this to have some nice features that I wanted while manually annotating videos for the DLC.

### Features

Many of the features included are also available in the built-in DLC GUI (i.e. labelling keypoints) and the DLC GUI has features that this does not have. However, this GUI includes the following features:

* modifying video brightness/LUT in live view
* stepping forward and backward through he video to label specific frames

### Installation

To install, run the following commands:

```python
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
