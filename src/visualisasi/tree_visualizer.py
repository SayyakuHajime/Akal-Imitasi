import numpy as np
from graphviz import Digraph

def export_tree_graphviz(model, feature_names=None, class_names=None, top_n=3):
    tree = model.tree_
    dot = Digraph(comment='Decision Tree')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
    
    node_counter = 0
    
    def format_leaf(leaf_data):
        cls_idx = leaf_data
        
        # Handle format C4.5 yang leaf-nya berupa dictionary {'value': ..., 'proba': ...}
        if isinstance(leaf_data, dict) and 'value' in leaf_data:
            cls_idx = leaf_data['value']
            
        class_label = class_names[cls_idx] if class_names else f"Class {cls_idx}"
        return f"Prediction:\n{class_label}"

    def recurse(node, parent_id, depth, decision_label=None):
        nonlocal node_counter
        my_id = str(node_counter)
        node_counter += 1
        
        is_internal = isinstance(node, dict) and 'feature' in node
        
        if not is_internal:
            label = format_leaf(node)
            dot.node(my_id, label=label, shape='ellipse', fillcolor='lightgreen')
            if parent_id is not None:
                dot.edge(parent_id, my_id, label=decision_label)
            return

        if depth >= top_n:
            dot.node(my_id, label="... (Tree Truncated) ...", shape='box', style='dashed', fillcolor='lightgrey')
            if parent_id is not None:
                dot.edge(parent_id, my_id, label=decision_label)
            return

        feat_idx = node['feature']
        threshold = node['threshold']
        
        feat_name = feature_names[feat_idx] if feature_names is not None else f"Feature[{feat_idx}]"
        label = f"{feat_name}\n<= {threshold:.3f}"
        
        dot.node(my_id, label=label)
        
        if parent_id is not None:
            dot.edge(parent_id, my_id, label=decision_label)
            
        recurse(node['left'], my_id, depth + 1, decision_label="True")
        recurse(node['right'], my_id, depth + 1, decision_label="False")

    recurse(tree, None, 0)
    
    return dot