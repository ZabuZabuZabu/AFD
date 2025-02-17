# First, change cwd to path/to/trackeval
#cd /home/zabu/Deep-EIoU/SportsMOT/codes/evaluation/TrackEval
# donnot forget to activate the environment if you want to use anaconda
source activate trackeval
# call the main evalualtion process.
python ./scripts/run_mot_challenge.py --BENCHMARK sportsmot --SPLIT_TO_EVAL val --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --PRINT_CONFIG True --GT_FOLDER ./data/ref --TRACKERS_FOLDER ./data/res --OUTPUT_FOLDER ./output/