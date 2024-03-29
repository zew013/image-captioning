

from experiment import Experiment
import sys

# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
# or specify the configuration by running `python main.py task-1-default-config`
if __name__ == "__main__":
    exp_name = 'task-1-default-config'

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        
    mode = 'train'
    if len(sys.argv) > 2:
        mode = sys.argv[2]
        

    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
    if mode == 'train':
        exp.train()
    elif mode == 'test':
        exp.test()
    else:
        exp.val()
