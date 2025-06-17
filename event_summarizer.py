import tensorflow as tf
import json
import glob
import csv

from tensorflow.core.util.event_pb2 import Event


def extract_training_args(event_file, args2get):
    for raw in tf.data.TFRecordDataset(event_file):
        event = Event.FromString(raw.numpy())  # parses the serialized event
        for value in event.summary.value:
            if value.tag.startswith("model_config") or value.tag.startswith("args"):
                text = value.tensor.string_val[0].decode("utf-8")
                data = json.loads(text)
                results = {}
                for arg in args2get:
                    results[arg] = data[arg]
            return results

def extract_best_epoch_and_f1(event_file):
    best_f1 = 0
    best_epoch = None
    current_epoch = None
    for raw in tf.data.TFRecordDataset(event_file):
        event = Event.FromString(raw.numpy())  # parses the serialized event
        for value in event.summary.value:
            if value.tag == 'train/epoch':
                current_epoch = value.simple_value
            elif value.tag == 'eval/f1':
                if float(value.simple_value) >= best_f1:
                    best_f1 = float(value.simple_value)
                    best_epoch = current_epoch           

    return {
        'best_epoch': best_epoch,
        'best_f1': best_f1,
    }

def summarize(logs_dir, training_args, out_dir):
    event_files = glob.glob(logs_dir + "/events.out.tfevents.*")

    result = []
    for event_file in event_files:
        trial_number = event_file.split('.')[-1]
        args = extract_training_args(event_file, training_args)
        f1_epoch = extract_best_epoch_and_f1(event_file)
        
        final_dict = {
            'trial': trial_number,
        }
        
        final_dict |= args | f1_epoch
        result.append(final_dict)

    out_path = f'{out_dir}/{logs_dir.split('/')[-1]}.csv'
    with open(out_path, 'w', newline='') as f:
        fieldnames = [k for k, v in result[0].items()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result)


logs_dir = 'logs/4'
training_args = ['per_device_train_batch_size', 'learning_rate', 'weight_decay', 'warmup_ratio']

summarize(logs_dir, training_args, 'summary')