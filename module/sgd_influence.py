import torch


def flat_grad(grad_list):
    gradient = torch.cat([x.view(-1) for x in grad_list])
    return gradient
    

def sample_in_batch(batch, i):
    one_sample = batch[i]
    
    x = one_sample.x
    edge_attr = one_sample.edge_attr
    edge_index = one_sample.edge_index
    y = one_sample.y
    b = torch.zeros(one_sample.num_nodes).to(torch.int64).to(x.device)
    
    return (x, edge_attr, edge_index, b, y)


def compute_lin_sample_grad(model, embedding, target, criterion):
    model.eval()
    
    embedding = embedding.unsqueeze(0)
    pred = model(embedding).to(torch.float32)
    
    y = target.view(pred.shape).to(torch.float32)
    is_labeled = y==y
    loss = criterion(pred[is_labeled], y[is_labeled])
    
    return torch.autograd.grad(loss, model.parameters())


def sgd_lin_if(model, batch, criterion, args):
    model.eval()
    
    eta = args.lr
    
    embeddings = model.pool(model.gnn(batch), batch.batch)
    targets = batch.y
    
    sample_grads = torch.stack([flat_grad(compute_lin_sample_grad(model.graph_pred_linear, embeddings[i], targets[i], criterion)) for i in range(len(embeddings))])
    
    sample_influence = []
    for i in range(sample_grads.size(0)):
        if_tmp = torch.abs(eta * sample_grads[i].dot(sample_grads[i]))
        sample_influence.append(if_tmp)
    sample_influence = torch.stack(sample_influence)
    
    return sample_influence
