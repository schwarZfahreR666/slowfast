params = dict()

params['num_classes'] = 174

params['train_dataset'] = 'train_num2label.txt'
params['val_dataset'] = 'val_num2label.txt'
params['test_dataset'] = 'val_num2label.txt'#'test.txt'

params['epoch_num'] = 40
params['batch_size'] = 16
params['step'] = 10
params['num_workers'] = 4
params['learning_rate'] = 1e-2
params['momentum'] = 0.9
params['weight_decay'] = 5e-3
params['display'] = 10
params['pretrained'] = 'model/SLOWFAST_8x8_R50.pkl'
params['gpu'] = [0]
params['log'] = 'log'
params['save_path'] = 'model'
params['sample_len'] = 64
params['clip_len'] = 48
params['frame_sample_rate'] = 1
params['from_caffe'] = True
