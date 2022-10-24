
# Interactive Perturbation Visualization Tool
A preliminary interactive perturbation tool that invites users to edit input source images and better understand the attribution problem. This tool is the result of a two-month summer undergraduate research project, and could be subject to more changes in the future.

## Notebook
### Description
This repository contains two perturbation tools in the two separate ipython notebooks. In both notebooks, users can choose to do one of the four kinds of occlusions (Grey Scale Preservation, Grey Scale Deletion, Gaussian Noise, and Gaussian Blur) by following the instructions in the notebooks.
* [Textual Class List](https://github.com/iceysiyiai/Interactive_Visualization_Tool_Summer2022/blob/main/InteractiveList.ipynb) shows the result of perturbation in the form of texual lists containing the top five classes and respective softmax scores for each. Users use their mouse to highlight the regions in picture that they want to occlude.
* [Green Bar Visualization](https://github.com/iceysiyiai/Interactive_Visualization_Tool_Summer2022/blob/main/InteractiveVisualization.ipynb) visualizes the result of perturbation by the changing length of the green bar at the side of the picture. The green bar's length demonstrates how the confidence level changes after each perturbation.
### Demo
Textual List Results             |   Visualization of results
:-------------------------:|:-------------------------:
<img src="Images/ListDemo.png" alt="Demonstration of the results of project 1" height="200" /> |     <img src="Images/demo1.gif" alt="Visualizatin Demo" height="200" />


## Potential Improvements & Bugs
### Improvements
* For some originally nonsquare images, the cropping transformation could be more precise. For example, the mud puppy fish from the image dataset originally came in a rectangled shape, and the current transformation crops out its head, which makes subsequent editting more inacurate.

Original Source Image         |   Image after Cropping Transformation
:-------------------------:|:-------------------------:
<img src="Images/OriginalFish.png" alt="Original Image" height="100" /> |     <img src="Images/CroppedFish.jpeg" alt="Image after cropping" height="100" />

* The visualization tool currently only allows users to see the interactive changes that occurs to the original class. A possible improvement would include the top five possible classes, and allow users to observe what happens to the five at the same time, instead of only at one.
### Bug
* The greenbar visualizer in visualization demo is only changing after the second mouse perturbation. Though the values in the `out.txt` changes, the length of the greenbar doesn't change accordingly until second perturbation.


## Acknowledgements
Special thanks to the supervision and guidance of [Dr.Ruth Fong](https://www.ruthfong.com/), and help from [Devon Ulrich](https://github.com/devonulrich). Also thanks to the contribution of Zohra Boumhaout.