CFG = {
    'data_path': '../data/OfficeHomeDataset_256/',
    'kwargs': {'num_workers': 1},
    'batch_size': 32,
    'epoch': 100,
    'lr': 1e-3,
    'momentum': .9,
    'log_interval': 10,
    'l2_decay': 0,
    'lambda': None,#1000,
    'backbone': None,
    'n_class': 65,
}