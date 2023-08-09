import os
from tensorboard.backend.event_processing import event_accumulator
from lib.util import merge_dicts

def load_tb_events(filep, step=None, tag_filter=None, tag_type='scalars'):
    if step is not None and not isinstance(step, list):
        step = [step]
    
    ea = event_accumulator.EventAccumulator(filep,
       size_guidance={ # see below regarding this argument
       event_accumulator.COMPRESSED_HISTOGRAMS: 500,
       event_accumulator.IMAGES: 4,
       event_accumulator.AUDIO: 4,
       event_accumulator.SCALARS: 0,
       event_accumulator.HISTOGRAMS: 1,
       })

    _=ea.Reload() # loads events from file
    
    event_reader = ea.Scalars
    if tag_type == 'histograms':
        event_reader = ea.Histograms

    result = dict()
    for tag in ea.Tags()[tag_type]:
        if tag_filter is not None and not tag_filter(tag):
            continue
        tag_entries = []
        value = None
        event_step = None
        if step is None:
            for event in event_reader(tag):
                if tag_type == 'scalars':
                    value = event.value
                else:
                    value = event.histogram_value
                event_step = event.step
                tag_entries.append((value, event_step))
        else:
            for event in event_reader(tag):
                if event.step in step:
                    if tag_type == 'scalars':
                        value = event.value
                    else:
                        value = event.histogram_value
                    event_step = event.step
                    tag_entries.append((value, event_step))
            
        if len(tag_entries) > 0:
            result[tag] = tag_entries
        
    return result


def merge_tb_events(tb_events):
    result = tb_events[0]

    for i in range(1, len(tb_events)):
        result = merge_dicts(result, tb_events[i])

    return result


def list_events_in_dir(exp_dirp, filter=None):
    tb_files = dict()
    if not os.path.exists(os.path.join(exp_dirp, 'model')):
        exp_names = os.listdir(exp_dirp)
    else:
        exp_names = [os.path.basename(exp_dirp.strip('/'))]
        exp_dirp = os.path.dirname(exp_dirp.strip('/'))

    for exp_name in exp_names:
        if filter is None or filter(exp_name):
            exp_tb_dir = os.path.join(exp_dirp, exp_name, 'model')
            exp_tb_files = [event_file for event_file in os.listdir(exp_tb_dir) if event_file.startswith('events.out')]
            if len(exp_tb_files) > 1:
                # merge events
                exp_tb_files = [os.path.join(exp_tb_dir, exp_tb_file) for exp_tb_file in exp_tb_files]
                # # sort by filesize, take the largest
                exp_tb_files = sorted(exp_tb_files, key=lambda x: os.path.getsize(x))
                # exp_tb_files = [exp_tb_files[-1]]
                tb_files[exp_name] = exp_tb_files
            else:
                tb_files[exp_name] = os.path.join(exp_tb_dir, exp_tb_files[-1])

    return tb_files


def argmax(event_data, tag):
    maxi = 0
    assert tag in event_data
    tag_data = event_data[tag]
    for i in range(1, len(tag_data)):
        if tag_data[i][0] > tag_data[maxi][0]:
            maxi = i
    return maxi