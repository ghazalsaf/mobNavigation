# mobNavitation

Necessary environment is different in execute.py and myeval.py. We used anaconda to make environment.

## execute.py

Environment is in executeenv.yml file. You can execute execute.py in terminal like :

> $ python execute.py <input_dir> <output_dir>

where <input_dir> means directory of the data_road folder from KITTI dataset.

## myeval.py

This myeval.py file must be in devkit_road folder from KITTI dataset. Environment is in evalenv.yml. You can evaluate our result in terminal like :

> $ python myeval.py <result_dir> <data_road_dir>

where <data_road_dir> means directory of the data_road folder from KITTI dataset.
