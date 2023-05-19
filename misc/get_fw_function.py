def get_forward_function(model, data, Z_ONE_HOT_DIM, **kwargs):
    model_name = model.name
    if model_name in ['TransformerNet', 'FractalNet', 'FractalNetShared']:
        out = model(data.x[:, :Z_ONE_HOT_DIM], data.edge_index, data.subgraph_edge_index,
              data.node_subnode_index, data.subnode_node_index,
              data.ground_node, data.subgraph_batch_index, data.batch, **kwargs)
        return out
    
    elif model_name in ['Net']:
        out = model(data.x[:, :Z_ONE_HOT_DIM], data.edge_index, data.batch, **kwargs)
        return out

    elif model_name in ['GNN', 'GNN_no_rel']:
        out = model(data.x[:, :Z_ONE_HOT_DIM], data.edge_index, data.edge_attr, data.batch, **kwargs)
        return out
    elif model_name in ['EGNN', 'Fractal_EGNN', 'FractalEGNNShared']:
        out = model(data)
        return out
    else:
        raise ValueError("Model name not recognized")