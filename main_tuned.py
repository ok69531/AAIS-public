import torch
import torch.optim as optim

import torch_geometric
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset

from module.set_seed import set_seed
from module.argument import get_parser
from module.model import GNNGraphPred
from module.train import (
    train,
    aa_train,
    aais_sgd_train,
    aais_adam_train,
    evaluation
)

import numpy as np
from copy import deepcopy

parser = get_parser()
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


if args.dataset == "tox21":
    num_task = 12
elif args.dataset == "hiv":
    num_task = 1
elif args.dataset == "bace":
    num_task = 1
elif args.dataset == "bbbp":
    num_task = 1
elif args.dataset == "toxcast":
    num_task = 617
elif args.dataset == "sider":
    num_task = 27
elif args.dataset == "clintox":
    num_task = 2


criterion = torch.nn.BCEWithLogitsLoss(reduction = 'sum')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = PygGraphPropPredDataset(name = f'ogbg-mol{args.dataset}', root = 'dataset')
    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx['train']], batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    val_loader = DataLoader(dataset[split_idx['valid']], batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

    auc_vals, auc_tests = [], []
    f1_vals, f1_tests = [], []

    for seed in range(args.num_runs):
        
        auc_vals_per_ratio, auc_tests_per_ratio = [], []
        f1_vals_per_ratio, f1_tests_per_ratio = [], []
        
        for args.ratio in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]:
            print(args)
            print(f'====================== run: {seed} ======================')
            
            set_seed(seed)
            torch_geometric.seed_everything(seed)
            
            best_val_auc, final_test_auc = 0, 0
            best_val_f1, final_test_f1 = 0, 0
            
            model = GNNGraphPred(num_tasks = num_task, num_layer = args.num_layer, emb_dim = args.emb_dim, 
                                gnn_type = args.gnn_type,
                                graph_pooling = args.graph_pooling, drop_ratio = args.dropout_ratio, JK = args.JK, 
                                virtual_node = args.virtual, residual = args.residual)
            model = model.to(device)
            
            if args.optim_method == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr = args.lr)
            elif args.optim_method == 'adam':
                iteration = 0
                first_moment = 0.
                second_moment = 0.
                optimizer = optim.Adam(model.parameters(), lr = args.lr)
            
            for epoch in range(1, args.epochs + 1):
                print(f'=== epoch {epoch}')
                
                if args.ratio == 0:
                    train(model, device, train_loader, criterion, optimizer)
                elif args.ratio == 1:
                    aa_train(model, device, train_loader, criterion, optimizer, args)
                else:
                    if epoch <= args.burn:
                        train(model, device, train_loader, criterion, optimizer)
                    else:
                        if args.optim_method == 'sgd':
                            aais_sgd_train(model, device, train_loader, criterion, optimizer, args)
                        elif args.optim_method == 'adam':
                            iteration, first_moment, second_moment = aais_adam_train(model, device, train_loader, criterion, optimizer, args, iteration, first_moment, second_moment)
                
                train_loss, train_auc, train_f1 = evaluation(model, device, train_loader, criterion, args)
                val_loss, val_auc, val_f1 = evaluation(model, device, val_loader, criterion, args)
                test_loss, test_auc, test_f1 = evaluation(model, device, test_loader, criterion, args)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    final_test_auc = test_auc
                    best_val_f1 = val_f1
                    final_test_f1 = test_f1
                    
                    best_epoch = epoch
                    model_params = deepcopy(model.state_dict())

                print(f'train loss: {train_loss:.4f}, train auc: {train_auc*100:.2f}')
                print(f'val loss: {val_loss:.4f}, val auc: {val_auc*100:.2f}')
                print(f'test loss: {test_loss:.4f}, test auc: {test_auc*100:.2f}')
            
            auc_vals_per_ratio.append(best_val_auc)
            auc_tests_per_ratio.append(final_test_auc)
            f1_vals_per_ratio.append(best_val_f1)
            f1_tests_per_ratio.append(final_test_f1)
        
        best_val_idx = auc_vals_per_ratio.index(max(auc_vals_per_ratio))
        auc_vals.append(auc_vals_per_ratio[best_val_idx])
        auc_tests.append(auc_tests_per_ratio[best_val_idx])
        f1_vals.append(f1_vals_per_ratio[best_val_idx])
        f1_tests.append(f1_tests_per_ratio[best_val_idx])
            
    print('')
    print(f'Validation auc: {np.mean(auc_vals)*100:.2f}({np.std(auc_vals)*100:.2f})')
    print(f'Test auc: {np.mean(auc_tests)*100:.2f}({np.std(auc_tests)*100:.2f})')
    print(f'Validation f1-score: {np.mean(f1_vals)*100:.2f}({np.std(f1_vals)*100:.2f})')
    print(f'Test f1-score: {np.mean(f1_tests)*100:.2f}({np.std(f1_tests)*100:.2f})')
    

if __name__ == '__main__':
    main()
