import os, sys
import computeBaseline, evaluateRoad

if __name__ == "__main__":

    assert len(sys.argv) == 3, "Usage : python myeval.py <result_dir> <data_road_dir>"

    result_dir = sys.argv[1]
    dataset_dir = sys.argv[2]
    trainDir = os.path.join(dataset_dir, "training")

    evaluateRoad.main(result_dir, trainDir)
