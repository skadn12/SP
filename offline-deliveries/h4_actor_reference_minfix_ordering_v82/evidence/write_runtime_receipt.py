from pathlib import Path
from datetime import datetime
import sys
HOME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HOME))
from common import read,write,bind,now
python='/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'
actions=[('freeze','FREEZE'),('prepare-evaluation','PREPARE_EVALUATION'),('live-trained','LIVE_TRAINED'),('live-base','LIVE_BASE'),('report','REPORT')]
observation=read(HOME/'evidence/RUNTIME_ENVIRONMENT_OBSERVATION.json')
parent=read(HOME/'BASE_EVALUATION.STARTED.json')['time']
native=read(HOME/'solver/base/STARTED.json')['started_at_utc']
observed=observation['observed_at_utc']
assert datetime.fromisoformat(parent)<=datetime.fromisoformat(observed)<=datetime.fromisoformat(read(HOME/'BASE_EVALUATION.COMPLETE.json')['time'])
write(HOME/'evidence/EVALUATION_RUNTIME_RECEIPT.json',{
 'created_at_utc':now(),'working_directory':str(HOME),'python':python,'python_flags':['-B'],
 'additional_environment':{},'removed_environment':[],
 'commands':[{'action':action,'command':python+' -B batch.py '+action,
   'stdout_stderr_log':bind(HOME/'evidence'/(name+'.log'))} for action,name in actions],
 'preparation':bind(HOME/'EVALUATION_PREPARATION.COMPLETE.json'),
 'trained_complete':bind(HOME/'TRAINED_EVALUATION.COMPLETE.json'),
 'base_complete':bind(HOME/'BASE_EVALUATION.COMPLETE.json'),
 'report':bind(HOME/'MIXED_EVALUATION_REPORT.json'),
 'environment_observation':bind(HOME/'evidence/RUNTIME_ENVIRONMENT_OBSERVATION.json'),
 'environment_observation_limitation':observation['limitation'],
 'environment_observation_stage':{'parent_started_at_utc':parent,'native_started_at_utc':native,
    'observed_at_utc':observed,'before_native_start_record':datetime.fromisoformat(observed)<datetime.fromisoformat(native)},
 'recoveries':[],'solver_automatic_replays':0,'frozen_code_edits':[],'training_performed':False})
