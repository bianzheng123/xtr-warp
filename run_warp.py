import os

if __name__ == '__main__':
    config_l = {
        'dbg': {
            'dataset_l': ['lotte', 'msmacro'],
            'n_thread_l': [1, 32]
        },
        'local': {
            'dataset_l': ['lotte-500-gnd'],
            'n_thread_l': [1, 32]
            # 'n_thread_l': [32]
        }
    }
    host_name = 'local'
    config = config_l[host_name]
    dataset_l = config['dataset_l']
    n_thread_l = config['n_thread_l']
    for dataset in dataset_l:
        for n_thread in n_thread_l:
            cmd = f'python3 run_api.py --dataset {dataset} --n_thread {n_thread} '
            os.system(cmd)
