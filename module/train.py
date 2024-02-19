import sys
sys.path.append('../')

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
)
from module.sgd_influence import sgd_lin_if
from module.adam_influence import (
    cum_first_moment,
    cum_second_moment,
    compute_lin_batch_grad,
    adam_lin_if
)


def train(model, device, loader, criterion, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            
            pred = model(batch)
            is_labeled = batch.y == batch.y
            loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss = loss / torch.sum(is_labeled)
            
            loss.backward()
            optimizer.step()


def aa_train(model, device, loader, criterion, optimizer, args):
    model.train()
    
    m = args.m
    max_pert = args.max_pert
    step_size = args.step_size

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            
            graph_embedding = model.pool(model.gnn(batch), batch.batch)
            perturb = torch.FloatTensor(graph_embedding.shape[0], graph_embedding.shape[1]).uniform_(-max_pert, max_pert).to(device)
            perturb.requires_grad_()
            
            pred = model.graph_pred_linear(graph_embedding + perturb)
            is_labeled = batch.y == batch.y
            
            loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss = loss / (torch.sum(is_labeled) * m)
            
            for _ in range(m-1):
                loss.backward()
                perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
                perturb.data = perturb_data.data
                perturb.grad[:] = 0
                
                tmp_graph_embedding = model.pool(model.gnn(batch), batch.batch) + perturb
                tmp_pred = model.graph_pred_linear(tmp_graph_embedding)
                
                loss = 0
                loss = criterion(tmp_pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss = loss / (torch.sum(is_labeled) * m)
                        
            loss.backward()
            optimizer.step()


def aais_sgd_train(model, device, loader, criterion, optimizer, args):
    model.train()
    
    m = args.m
    emb_dim = args.emb_dim
    ratio = args.ratio
    max_pert = args.max_pert
    step_size = args.step_size
    
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            
            k = int(len(batch.y) * ratio)
            
            loo_influence = sgd_lin_if(model, batch, criterion, args)
            
            _, topk_idx = torch.topk(loo_influence, k = k, axis = -1)
            low_idx = [i for i in range(len(batch.y)) if i not in topk_idx]
            
            perturb = torch.FloatTensor(k, emb_dim).uniform_(-max_pert, max_pert).to(device)
            perturb.requires_grad_()
            
            y = batch.y
            is_labeled = y == y

            for _ in range(m-1):
                graph_embedding = model.pool(model.gnn(batch), batch.batch)[topk_idx, :]
                graph_embedding = graph_embedding + perturb
                
                pred = model.graph_pred_linear(graph_embedding)
                
                topk_y = y[topk_idx]
                topk_is_labeled = topk_y == topk_y
                
                loss = 0
                loss = criterion(pred.to(torch.float32)[topk_is_labeled], topk_y.to(torch.float32)[topk_is_labeled])
                loss = loss / (torch.sum(is_labeled) * m)
                loss.backward()
                
                perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
                perturb.data = perturb_data.data
                perturb.grad[:] = 0
            
            graph_embedding = model.pool(model.gnn(batch), batch.batch)
            
            topk_graph_embedding = graph_embedding[topk_idx, :] + perturb
            low_graph_embedding = graph_embedding[low_idx, :]
            
            low_y = y[low_idx]
            low_is_labeled = low_y == low_y
            
            topk_pred = model.graph_pred_linear(topk_graph_embedding)
            low_pred = model.graph_pred_linear(low_graph_embedding)
            
            loss = 0
            topk_loss = criterion(topk_pred.to(torch.float32)[topk_is_labeled], topk_y.to(torch.float32)[topk_is_labeled])
            low_loss = criterion(low_pred.to(torch.float32)[low_is_labeled], low_y.to(torch.float32)[low_is_labeled])
            
            topk_loss = topk_loss / (torch.sum(is_labeled) * m)
            low_loss = topk_loss / (torch.sum(is_labeled))
            
            loss = topk_loss + low_loss
            
            loss.backward()
            optimizer.step()


def aais_adam_train(model, device, loader, criterion, optimizer, args, iteration, first_moment, second_moment):
    model.train()
    
    m = args.m
    emb_dim = args.emb_dim
    ratio = args.ratio
    max_pert = args.max_pert
    step_size = args.step_size
    
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            
            iteration += 1
            
            k = int(len(batch.y) * ratio)
            
            curr_batch_grad = compute_lin_batch_grad(model ,batch, criterion)
            first_moment = cum_first_moment(curr_batch_grad, first_moment)
            second_moment = cum_second_moment(curr_batch_grad, second_moment)
            loo_influence = adam_lin_if(model, batch, curr_batch_grad, first_moment, second_moment, iteration, criterion, args)
            
            _, topk_idx = torch.topk(loo_influence, k = k, axis = -1)
            low_idx = [i for i in range(len(batch.y)) if i not in topk_idx]
            
            perturb = torch.FloatTensor(k, emb_dim).uniform_(-max_pert, max_pert).to(device)
            perturb.requires_grad_()
            
            y = batch.y
            is_labeled = y == y

            for _ in range(m-1):
                graph_embedding = model.pool(model.gnn(batch), batch.batch)[topk_idx, :]
                graph_embedding = graph_embedding + perturb
                
                pred = model.graph_pred_linear(graph_embedding)
                
                topk_y = y[topk_idx]
                topk_is_labeled = topk_y == topk_y
                
                loss = 0
                loss = criterion(pred.to(torch.float32)[topk_is_labeled], topk_y.to(torch.float32)[topk_is_labeled])
                loss = loss / (torch.sum(is_labeled) * m)
                loss.backward()
                
                perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
                perturb.data = perturb_data.data
                perturb.grad[:] = 0
            
            graph_embedding = model.pool(model.gnn(batch), batch.batch)
            
            topk_graph_embedding = graph_embedding[topk_idx, :] + perturb
            low_graph_embedding = graph_embedding[low_idx, :]
            
            low_y = y[low_idx]
            low_is_labeled = low_y == low_y
            
            topk_pred = model.graph_pred_linear(topk_graph_embedding)
            low_pred = model.graph_pred_linear(low_graph_embedding)
            
            loss = 0
            topk_loss = criterion(topk_pred.to(torch.float32)[topk_is_labeled], topk_y.to(torch.float32)[topk_is_labeled])
            low_loss = criterion(low_pred.to(torch.float32)[low_is_labeled], low_y.to(torch.float32)[low_is_labeled])
            
            topk_loss = topk_loss / (torch.sum(is_labeled) * m)
            low_loss = topk_loss / (torch.sum(is_labeled))
            
            loss = topk_loss + low_loss
            
            loss.backward()
            optimizer.step()

    return iteration, first_moment, second_moment


@torch.no_grad()
def evaluation(model, device, loader, criterion, args):
    model.eval()
    
    y_true = []
    y_pred_prob = []
    loss_list = []
    
    for _, batch in enumerate(loader):
        batch = batch.to(device)
        
        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = torch.sigmoid(pred).to(torch.float32)
            y = batch.y.view(pred.shape).to(torch.float32)
            
            is_labeled = batch.y == batch.y
            
            y_true.append(y.detach().cpu())
            y_pred_prob.append(pred.detach().cpu())
            
            loss = criterion(pred[is_labeled], y[is_labeled])
            loss = loss / torch.sum(is_labeled)
            
            loss_list.append(loss)
        
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred_prob = torch.cat(y_pred_prob, dim = 0).numpy()
    y_pred = np.where(y_pred_prob > 0.5, 1, 0)
    loss_list = torch.stack(loss_list)
    
    rocauc_list = []
    f1_list = []
    
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred_prob[is_labeled,i]))
        
        is_labeled = y_true[:,i] == y_true[:,i]
        f1_list.append(f1_score(y_true[is_labeled, i], y_pred[is_labeled, i]))
    
    return sum(loss_list)/len(loss_list), sum(rocauc_list)/len(rocauc_list),  sum(f1_list)/len(f1_list)
