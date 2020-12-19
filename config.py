params = dict()

params['num_classes'] = 174

params['train_dataset'] = 'train_num2label.txt'
params['val_dataset'] = 'val_num2label.txt'
params['test_dataset'] = 'test.txt'

params['epoch_num'] = 50
params['batch_size'] = 16
params['step'] = 15
params['num_workers'] = 4
params['learning_rate'] = 1e-3
params['momentum'] = 0.9
params['weight_decay'] = 5e-3
params['display'] = 10
params['pretrained'] = 'model/84_1/clip_len_32frame_sample_rate_1_checkpoint_39.pth.tar'
params['gpu'] = [0]
params['log'] = 'log'
params['save_path'] = 'model'
params['sample_len'] = 64
params['clip_len'] = 32
params['frame_sample_rate'] = 1
params['from_caffe'] = False
