import time
date = '-' + str(time.localtime().tm_mon) +'-'+ str(time.localtime().tm_mday)
models = ['resnet-32', 'resnet-110']
multimodals = [11]
#names = ['resnet-110']
lrates = [[0.1, 0.01, 0.001]]
margins = [0.0]
#margins = [1.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]

#alphas = [1.0, 2.0, 5.0]
#betas = [1.0]

#alphas = [0.0]
#betas = [0.8, 0.5, 0.2]

#alphas = [0.5, 0.0]
#betas = [0.1, 0.2, 0.01, 0.05]

alphas = [0.0]
betas = [0.1]

seeds = [1,2,3,4]
cmd_ = 'python run_cifar_train.py --dataset cifar-100 '
config_ = '{"init_filter": 3, "disp_iter": 100, "div255": true,\
 "relu_leakiness": false, "num_channel": 3, \
 "combine_weights": 0, \
 "lr_decay_steps": [40000, 60000],\
 "max_train_iter": 80000,\
 "whiten": false, "filters": [16, 16, 32, 64],\
 "init_max_pool": false,  \
 "init_stride": 1,  \
  "width": 32, "save_iter": 10000, "valid_iter": 1000, \
  "lr_scheduler_type": "fixed", "momentum": 0.9, \
  "wd": 0.0002, "optimizer": "mom", "batch_size": 100, \
  "activate_before_residual": [true, false, false], \
  "strides": [1, 2, 2], "height": 32, \
  "filter_initialization": "normal", "num_classes": 100, \
  "data_aug": true, "min_lrn_rate": 0.0001, \
  "prefetch": true, "model_class": "resnet", \
  "use_bottleneck": false \
  '
#}"seed": 0, '
#"multimodal": 13, \
# "name": "resnet-110", \
# "num_residual_units": [18, 18, 18],\
# "lr_list": [0.01, 0.001],\
# "base_learn_rate": 0.1, \
# "margin": 25.0\

def gen_cmd(key, value, shortkey=''):
  return ' --' + str(key) + ' ' + str(value) + ' ', str(shortkey) + str(value)
def gen_config(key, value, isstr=True):
  if isstr:
    return ', "' + str(key) + '": ' + '"' + str(value)+ '" '
  else:
    return ', "' + str(key) + '": ' + str(value)+ ' '

cmds,configs, names= [], [], []
for mo in models:
  _mo = gen_config('name', mo, True)
  _mo_cmd, _mo_name = gen_cmd('model', mo, '')

  if mo in ['resnet-32']:
    _mo += gen_config('num_residual_units', '[5, 5, 5]', False)
  elif mo in ['resnet-110']:
    _mo += gen_config('num_residual_units', '[18, 18, 18]', False)
  elif mo in ['resnet-164']:
    _mo += gen_config('num_residual_units', '[18, 18, 18]', False)
    print('not implemented yet -- use_bottleneck')
    exit(-1)
  else:
    exit(-1)  
  for mm in multimodals:
    _mm = gen_config('multimodal', mm, False)
    _, _mm_name = gen_cmd('', mm, 'opt')

    for margin in margins:
      _margin = gen_config('margin', margin, False)
      _, _margin_name = gen_cmd('', margin, 'm')

      for lrs in lrates:
        _lrs = gen_config('base_learn_rate', lrs[0], False)
        _lrs += gen_config('lr_list', '[' + ','.join([str(t) for t in lrs[1:]]) +']', False)
        _, _lrs_name = gen_cmd('', '.'.join([str(t) for t in lrs]), 'lr')

        for al in alphas:
          _al = gen_config('alpha', al, False)
          _, _al_name = gen_cmd('', al, 'A')
          for be in betas:
            _be = gen_config('beta', be, False)
            _, _be_name = gen_cmd('', be, 'B')
            for seed in seeds:
              _seed = gen_config('seed', seed, False)
              _, _seed_name = gen_cmd('', seed, 'rs')

              # configname = _mo_name + _mm_name + _margin_name + _lrs_name + date
              # configs.append(config_+_mo+_mm+_margin+_lrs + "}")
              configname = _mo_name + _mm_name + _al_name + _be_name + _margin_name + _seed_name
              configs.append(config_+_mo+_mm+_margin+_lrs+_al+_be + _seed + "}")
              names.append(('../configs/conf.' + configname, configname))
              cmds.append(cmd_ + _mo_cmd +'--config configs/conf.' + configname)


N = len(cmds)
print('there are in total {} commands'.format(N))

off = 8
for i in range(N):
  with open('run{}_{}.sh'.format(off + i, date), 'w') as f:
    f.write('export CUDA_VISIBLE_DEVICES={}\ncd ..\n'.format(i))
    c = cmds[i]
    f.write(c+'\necho "done!"')
  conf = configs[i]
  conf_n = names[i][0]
  with open('{}'.format(conf_n), 'w') as f:
    f.write(conf)


