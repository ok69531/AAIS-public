import argparse

def get_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type = int, default = 10, help = 'number of runs')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size for training (default = 32)')
    parser.add_argument('--epochs', type = int, default = 100, help = 'number of epochs to train (default = 100)')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate (default = 0.001)')
    parser.add_argument('--lr_scale', type = float, default = 1, help = 'relative learning rate for the feature extraction layer (default = 1)')
    parser.add_argument('--num_layer', type = int, default = 5, help = 'number of GNN message passing layers (default = 5)')
    parser.add_argument('--emb_dim', type = int, default = 300, help = 'embedding dimensions (default = 300)')
    parser.add_argument('--dropout_ratio', type = float, default = 0.5, help = 'dropout ratio (default = 0.5)')
    parser.add_argument('--graph_pooling', type = str, default = 'mean', help = 'graph level pooling (sum, mean, max)')
    parser.add_argument('--JK', type = str, default = 'last', help = 'how the node features across layers are combined. (last, sum, max or concat)')
    parser.add_argument('--gnn_type', type = str, default = 'gcn', help = 'gcn, gin')
    parser.add_argument('--ratio', type = float, default = 0.5, help = 'top k IF ratio (default = 0.5)')
    parser.add_argument('--m', type = int, default = 3, help = 'number of update for perturbation (default = 3)')
    parser.add_argument('--step_size', type = float, default = 0.001, help = 'gradient ascent learning rate (default = 0.001)')
    parser.add_argument('--max_pert', type = float, default = 0.01, help = 'perturbation budget (default = 0.01)')
    parser.add_argument('--burn', type = int, default = 20, help = 'burn-in period (default = 20)')
    parser.add_argument('--dataset', type = str, default = 'bbbp', help = 'bace, bbbp, clintox, hiv, sider, tox21, toxcast')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of workers for dataset loading')
    parser.add_argument('--virtual', type = bool, default = False)
    parser.add_argument('--residual', type = bool, default = False)
    parser.add_argument('--train_type', type = str, default = 'base', help = 'base, aa, aais')
    parser.add_argument('--optim_method', type = str, default = 'sgd', help = 'sgd, adam')
    
    return parser
