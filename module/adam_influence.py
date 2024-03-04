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
    
    return torch.autograd.grad(loss, list(model.parameters()))


def compute_lin_batch_grad(model, batch, criterion):
    model.eval()

    pred = model(batch)
    y = batch.y.view(pred.shape)
    is_labeled = y==y
    loss = criterion(pred.to(torch.float32)[is_labeled], y.to(torch.float32)[is_labeled])
    loss = loss / torch.sum(is_labeled)
    
    return torch.autograd.grad(loss, list(model.graph_pred_linear.parameters()))


def cum_first_moment(curr_batch_grad, cum1mom, beta1 = 0.9):
    curr_batch_grad = flat_grad(curr_batch_grad)
    
    try:
        cum1mom = flat_grad(cum1mom)
    except:
        pass
    
    cumulated = beta1 * cum1mom + (1 - beta1) * curr_batch_grad
    
    return cumulated


def cum_second_moment(curr_batch_grad, cum2mom, beta2 = 0.999):
    curr_batch_grad = flat_grad(curr_batch_grad)
    
    try:
        cum2mom = flat_grad(cum2mom)
    except:
        pass
    
    cumulated = beta2 * cum2mom + (1 - beta2) * (curr_batch_grad * curr_batch_grad)
    
    return cumulated


def adam_lin_if(model, batch, curr_batch_grad, first_moment, second_moment, iteration, criterion, args, beta1 = 0.9, beta2 = 0.999, eps = 1e-08):
    model.eval()
    eta = args.lr
    
    m_hat = first_moment / (1 - beta1 ** iteration)
    v_hat = second_moment / (1 - beta2 ** iteration)
    
    curr_batch_grad = flat_grad(curr_batch_grad)
    
    embeddings = model.pool(model.gnn(batch), batch.batch)
    targets = batch.y
    sample_grads = [flat_grad(compute_lin_sample_grad(model.graph_pred_linear, embeddings[i], targets[i], criterion)) for i in range(len(batch))]
    
    sample_influence = []
    
    denom = v_hat ** (3/2)
    for i in range(len(sample_grads)):
        num1 = ((1 - beta1) * v_hat * sample_grads[i]) / (1 - beta1 ** iteration)
        num2 = ((1 - beta2) * sample_grads[i] * curr_batch_grad * m_hat) / (1 - beta2 ** iteration)
        frac = (num1 - num2) / (denom + eps)
        if_tmp = torch.abs(eta * sample_grads[i].dot(frac))
        sample_influence.append(if_tmp)
    sample_influence = torch.stack(sample_influence)
    
    return sample_influence
