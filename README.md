# Project: Dogs, Fried Chicken or Blueberry Muffins?
![image](figs/chicken.jpg)
![image](figs/muffin.jpg)

### [Full Project Description](doc/project3_desc.md)

Term: Fall 2017

+ Team 5
+ Team members
	+ Chen, Tiantian
	+ Guo, Yajie
	+ Ni, Jiayu
	+ Tao, Siyi
	+ Zhao, Yufei 

+ Project summary: In this project, except 5000 SIFT feature descriptors, we extracted 960 GIST feature descriptors, which summarizes the gradient information (scales and orientations) for different parts of an image. Then we tested SVM (linear and non-linear), random forest, GBM, XGBoost classification methods to recognize images of dogs versus fried chicken versus blueberry muffins. XGBoost achieves best result, after tuning, test error is around 10.4%.
	
**Contribution statement**:

	Chen, Tiantian: XGBoost model, main.ipynb construction, github organization
	Guo, Yajie: SVM model (linear and non linear)
	Ni, Jiayu: Random forest model
	Tao, Siyi: feature extraction, main.ipynb construction, github organization
	Zhao, Yufei: GBM model




Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
