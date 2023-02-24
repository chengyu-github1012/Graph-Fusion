import numpy as np

num_nodes_of_interest = 2
head_to_visualize = 0
# 选择要绘制的目标节点（随机节点）
nodes_of_interest_ids = np.random.randint(low=0, high=data.num_nodes, size=num_nodes_of_interest)  # 根据参数中所指定的范围生成随机整数

put_x1 = my_net.conv1.forward(x=data.x, edge_index=data.edge_index, edge_value=edge,
                              return_attention_weights=True)[1]
print(f'put_x1 = {put_x1}')
edge_index = put_x1[0]
edge_index = np.squeeze(edge_index)
edge_index = edge_index.cpu()
print(f'edge_index = {edge_index}')
all_attention_weights = put_x1[1]
all_attention_weights = all_attention_weights.cpu().detach()
all_attention_weights = np.squeeze(all_attention_weights)
print(f'all_attention_weights = {all_attention_weights}')
print(all_attention_weights.shape)

target_node_ids = edge_index[0]
source_nodes = edge_index[1]

for target_node_id in nodes_of_interest_ids:
    # Step 1: Find the neighboring nodes to the target node
    # Note: self edges are included so the target node is it's own neighbor (Alexandro yo soy tu madre)
    src_nodes_indices = torch.eq(target_node_ids, target_node_id)
    source_node_ids = source_nodes[src_nodes_indices].cpu().numpy()
    size_of_neighborhood = len(source_node_ids)

    # Step 2: Fetch their labels
    labels = data.y[source_node_ids].cpu().numpy()

    # Step3
    attention_weights = all_attention_weights[src_nodes_indices].cpu().numpy()

    print(f'Max attention weight = {np.max(attention_weights)} and min = {np.min(attention_weights)}')
    attention_weights /= np.max(attention_weights)

    # Build up the neighborhood graph whose attention we want to visualize
    # igraph constraint - it works with contiguous range of ids so we map e.g. node 497 to 0, 12 to 1, etc.
    id_to_igraph_id = dict(zip(source_node_ids, range(len(source_node_ids))))
    ig_graph = ig.Graph()
    ig_graph.add_vertices(size_of_neighborhood)
    ig_graph.add_edges(
        [(id_to_igraph_id[neighbor], id_to_igraph_id[target_node_id]) for neighbor in source_node_ids])

    # Prepare the visualization settings dictionary and plot
    visual_style = {
        "edge_width": attention_weights,
        "layout": ig_graph.layout_reingold_tilford_circular()
    }

    color_map = ['red', 'blue']
    visual_style["vertex_color"] = [color_map[label] for label in labels]

    ig.plot(ig_graph, **visual_style)
