import os
import csv
import json
from tqdm import tqdm


class OfflineRunner:

    def __init__(self, model, bag_reader, bag_list):
        self.model = model
        self.bag_reader = bag_reader
        self.bag_list = bag_list
        self.show_viz = True

    def run(self):
        for bag_info in self.bag_list:
            results = self.run_bag(bag_info)
            self.log_results(bag_info, results)
            break
    
    def run_bag(self, bag_info):
        results = {}

        self.model.setup(bag_info)
        if self.show_viz:
            visualizer = LocalizationVisualizer(bag_info)

        for _, data in enumerate(tqdm(self.bag_reader(bag_info))):

            pred = self.model.step(data)

            if self.show_viz:
                viz = visualizer.visualize(data, pred)
                visualizer.show(viz)

            # Collate results
            for k,v in pred.items():
                if k not in results:
                    results[k] = []
                results[k].append(v.tolist())
        return results
    
    def log_results(self, bag_info, results):
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        
        data = {}
        data['bag_info'] = bag_info
        data['results'] = results
        results_fn = os.path.join(self.output_dir, '{}_{}_{}_{}.json'.format(*bag_info))
        with open(results_fn, 'w') as f:
            json.dump(data, f, indent=2)