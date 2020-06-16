'''

Author: Daniel M. Low
License: Apache 2.0

'''
import argparse

# Run this in terminal to test with toy run
# python3 -i run.py --job_array_task_id=1 --run_modelN=0 --run_version_number=7 --task=binary --toy=True

toy = False
run_version_number = 2 #version
run_modelN = 0 # 1=SGD,2=RBF,3=Extratrees,4=XGBModel,5=MLP (not 0)
run_final_model = True

subsample = 2700  # int, float (0.2) or False. 3000 is bipolar after removing duplicates.
task = 'binary' #'multiclass'

include_subreddits_overN = False #3600 #remove subreddits under this number
subsample_subreddits_overN = False #25000  # Load less, subsample if dataset > N, set to 0 for all
model = 'vector' # or 'sequential' for lstm
dim_reduction = False
cv = 5
features = ['tfidf', 'liwc', 'readability'] #todo: make loading variable, removed, add to args
stem = True

midpandemic_train = False
midpandemic_test = True
subsample_midpandemic_test = 1100
pre_or_post = 'pre'
timestep = None
input_dir = './../../datum/reddit/input/final_features/'
output_dir = './../../datum/reddit/output/'

def str2boolean(s):
	if s == 'False':
		s_new = False
	else:
		s_new = True
	return s_new

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--job_array_task_id',
                    help='default: ${SLURM_ARRAY_TASK_ID} or 1. When using job arrays, this will be set by the bash script by ${SLURM_ARRAY_TASK_ID} or set to 1, which will be substracted below by 1 for zero indexing')
parser.add_argument('--toy', help='run quickly with less labels, parameters and splits')
parser.add_argument('--run_version_number',
                    help='default: 0. if you need to run the same model but output to a different directory, change number')
parser.add_argument('--dim_reduction', help="True or False")
parser.add_argument('--features', help="['liwc']")
parser.add_argument('--stem', help="True or False")
parser.add_argument('--run_modelN', help="True or False")
parser.add_argument('--task', help="binary or multiclass")
args = parser.parse_args()

if args.toy!=None:
	toy = str2boolean(args.toy)

if args.run_version_number != None:
	run_version_number = int(args.run_version_number)

if args.job_array_task_id != None:
	subredditN = int(args.job_array_task_id) - 1

if args.dim_reduction!=None:
	dim_reduction = str2boolean(args.dim_reduction)

if args.stem != None:
	stem= str2boolean(args.stem)

if args.run_modelN!= None:
	run_modelN = int(args.run_modelN)

if args.task!= None:
	task= args.task

print('args: ',args)

# Define default subreddits
subreddits = ['addiction','EDAnonymous','adhd','autism','alcoholism', 'bipolarreddit', 'depression', 'anxiety','healthanxiety', 'lonely', 'schizophrenia', 'socialanxiety', 'suicidewatch']
# subreddits  = ['depression', 'anxiety', 'suicidewatch']
# subreddits_midpandemic = subreddits+['COVID19_support', 'relationships', 'divorce']
if run_version_number in [2,3]:
	subreddits = ['alcoholism', 'bipolarreddit', 'depression', 'healthanxiety', 'lonely', 'schizophrenia', 'socialanxiety', 'suicidewatch']
	subsample = 5600 #for alcoholism

if run_version_number in [4]:
	subreddits = ['healthanxiety', 'lonely', 'socialanxiety', 'suicidewatch']
	subsample = 11000  # for alcoholism

if run_version_number in [5,6]:
	subreddits = ['addiction','EDAnonymous','adhd','autism','alcoholism', 'bipolarreddit', 'depression', 'anxiety','healthanxiety', 'lonely', 'schizophrenia', 'socialanxiety', 'suicidewatch']
	# subreddits = ['depression', 'anxiety', 'suicidewatch']
	subsample = 5600 #for alcoholism

if run_version_number in [7]:
	subreddits = ['addiction', 'EDAnonymous', 'adhd', 'autism', 'alcoholism', 'bipolarreddit', 'bpd','depression', 'anxiety',
	              'healthanxiety', 'lonely', 'schizophrenia', 'socialanxiety', 'suicidewatch']
	# subreddits = ['depression', 'anxiety', 'suicidewatch']
	subsample = 3000  # for alcoholism
	subsample_midpandemic_test = 1100

if run_version_number in [8]:
	subreddits = ['addiction', 'EDAnonymous', 'adhd', 'autism', 'alcoholism', 'bipolarreddit', 'bpd','depression', 'anxiety',
	              'healthanxiety', 'lonely', 'schizophrenia', 'socialanxiety', 'suicidewatch']
	# subreddits = ['depression', 'anxiety', 'suicidewatch']
	subsample = 2700 # for alcoholism
	subsample_midpandemic_test = 900


if toy:
	subsample = 80
	subsample_midpandemic_test = 80
	subreddits = subreddits

