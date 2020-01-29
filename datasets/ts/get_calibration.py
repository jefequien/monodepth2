import os
import argparse
import csv
import pickle

from calibration_manager import CalibrationManager

def main(args):
    
    bag_list = [row for row in csv.reader(open(args.bag_list, 'r'))][1:]

    all_calibs = {}
    for bag_info in bag_list:
        print(bag_info)
        bag_name, map_name, begin, end = bag_info
        cm = CalibrationManager(dataset=bag_name)
        
        calibs = cm.get_cameras()
        all_calibs[bag_name] = calibs

    with open('calibrations.pkl', 'wb') as f:
        pickle.dump(all_calibs, f, protocol=pickle.HIGHEST_PROTOCOL)
    

    with open('calibrations.pkl', 'rb') as f:
        all_calibs = pickle.load(f)
    print(all_calibs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag-list',
                        type=str, help='path to a test image or folder of images',
                        required=True)
    args = parser.parse_args()
    main(args)