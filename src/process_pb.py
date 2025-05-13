import os
import argparse

import numpy as np
import jax.numpy as jnp

import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt

from color import hsv2rgb
from cppn import CPPN, FlattenCPPNParameters, activation_fn_map
import util
import picbreeder_util

def load_pbcppn(zip_file_path):
    """
    Load the raw picbreeder genome from a zip file and convert it to a dictionary of nodes and links (the NEAT graph).
    """
    root = picbreeder_util.load_zip_xml_as_dict(zip_file_path)

    ns, ls = [], []
    nodes_ = root['genome']['nodes']['node']
    links_ = root['genome']['links']['link']
    nodes, links = [], []
    for node in nodes_:
        # node = dict(label=node['@label'] if '@label' in node else "", id=int(node['marking']['@id']), activation=node['activation']['#text'][:-3])
        id_ = node['marking']['@branch']+"_"+node['marking']['@id']
        node = dict(label=node['@label'] if '@label' in node else "", id=id_, activation=node['activation']['#text'][:-3])
        nodes.append(node)
    for link in links_:
        # link = dict(id=int(link['marking']['@id']), source=int(link['source']['@id']), target=int(link['target']['@id']), weight=float(link['weight']['#text']))

        source_id = link['source']['@branch']+"_"+link['source']['@id']
        target_id = link['target']['@branch']+"_"+link['target']['@id']
        link = dict(id=int(link['marking']['@id']), source=source_id, target=target_id, weight=float(link['weight']['#text']))
        links.append(link)

    if 'ink' in [node['label'] for node in nodes]: # convert ink output to hsv standard
        node_v = [node for node in nodes if node['label'] == 'ink'][0]
        node_v['label'] = 'brightness'
        nodes.append(dict(label='hue', id=1000000, activation='identity'))
        nodes.append(dict(label='saturation', id=1000001, activation='identity'))
        links.append(dict(id=1000002, source=node_v['id'], target=1000000, weight=0.))
        links.append(dict(id=1000003, source=node_v['id'], target=1000001, weight=0.))

    special_nodes = {}
    special_nodes['x'] = [node['id'] for node in nodes if node['label'] == 'x'][0]
    special_nodes['y'] = [node['id'] for node in nodes if node['label'] == 'y'][0]
    special_nodes['d'] = [node['id'] for node in nodes if node['label'] == 'd'][0]
    special_nodes['bias'] = [node['id'] for node in nodes if node['label'] == 'bias'][0]
    special_nodes['h'] = [node['id'] for node in nodes if node['label'] == 'hue'][0]
    special_nodes['s'] = [node['id'] for node in nodes if node['label'] == 'saturation'][0]
    special_nodes['v'] = [node['id'] for node in nodes if node['label'] == 'brightness'][0]
    # links = [link for link in links if link['weight'] != 0.]
    return dict(nodes=nodes, links=links, special_nodes=special_nodes)

def do_forward_pass(nn):
    """
    Do a forward pass through the NEAT graph given by the above function.
    """
    res = 256
    x = y = jnp.linspace(-1., 1., res)
    x, y = jnp.meshgrid(x, y)
    d = jnp.sqrt(x**2 + y**2)*1.4
    b = jnp.ones_like(x)

    node2activation = {n['id']: n['activation'] for n in nn['nodes']}
    node2out_links = {n['id']: [(l['target'], l['weight']) for l in nn['links'] if l['source'] == n['id']] for n in nn['nodes']}
    node2in_links = {n['id']: [(l['source'], l['weight']) for l in nn['links'] if l['target'] == n['id']] for n in nn['nodes']}
    node_x = nn['special_nodes']['x']
    node_y = nn['special_nodes']['y']
    node_d = nn['special_nodes']['d']
    node_b = nn['special_nodes']['bias']
    node_h = nn['special_nodes']['h']
    node_s = nn['special_nodes']['s']
    node_v = nn['special_nodes']['v']

    node2val = {} #{node['id']: np.zeros_like(x) for node in nn['nodes']}
    node2val[node_x], node2val[node_y], node2val[node_d], node2val[node_b] = x, y, d, b

    def get_value_recur(node, path=[]):
        if node in node2val:
            return node2val[node]
        if node in path:
            print(f'CYCLE: {path}')
            return jnp.zeros_like(x)
        val = jnp.zeros_like(x)
        for node_src, weight in node2in_links[node]:
            val = val + weight * get_value_recur(node_src, path=path+[node])
        node2val[node] = activation_fn_map[node2activation[node]](val)
        return node2val[node]

    for node in [node_h, node_s, node_v]: # actual forward pass
        get_value_recur(node, path=[])
    
    h, s, v = node2val[node_h], node2val[node_s], node2val[node_v]
    r, g, b = hsv2rgb((h+1)%1, s.clip(0,1), jnp.abs(v).clip(0, 1))
    rgb = jnp.stack([r, g, b], axis=-1)
    return dict(rgb=rgb, node2val=node2val)

def get_weight_matrix(net, i_layer, nodes_cache, activation_neurons=""):
    """
    Helper function to get a layeried weight matrix for a specific layer.
    Don't need to call this directly.
    """
    node2activation = {n['id']: n['activation'] for n in net['nodes']}

    activations = [i.split(":")[0] for i in activation_neurons.split(",")]
    d_hidden = np.array([int(i.split(":")[-1]) for i in activation_neurons.split(",")])

    position_offset = {act: p for act, p in zip(activations, np.cumsum(np.array([0]+list(d_hidden)[:-1])))}

    def get_position_within_layer(nodes_caches_layer, node): # node_id
        i_node = [n for n, _ in nodes_caches_layer].index(node)

        node, act = nodes_caches_layer[i_node]
        p = 0
        for n, a in nodes_caches_layer:
            if n==node:
                break
            elif a==act:
                p+=1
        p = p + position_offset[act]
        return p

    prev_layer = nodes_cache[i_layer]
    this_layer = nodes_cache[i_layer+1]
    prev_layer = [(a, 'cache' if not b else node2activation[a]) for a, b in prev_layer]
    this_layer = [(a, 'cache' if not b else node2activation[a]) for a, b in this_layer]

    if i_layer==0:
        prev_layer = [(a, 'cache') for a, _ in prev_layer]

    weight_mat = np.zeros((sum(d_hidden), sum(d_hidden)))
    for n, act in this_layer:
        p = get_position_within_layer(this_layer, n)
        if act=='cache':
            p2 = get_position_within_layer(prev_layer, n)
            # print(n, act, p, p2)
            weight_mat[p, p2] = 1.
        else:
            for src, w in [(l['source'], l['weight']) for l in net['links'] if l['target']==n]:
                p2 = get_position_within_layer(prev_layer, src)
                weight_mat[p, p2] = w
    return weight_mat.T


def layerize_nn(nn):
    """
    Converts the picbreeder NEAT graph to a dense network by layerizing it as described in the paper.
    Turns the connections which connect to neurons in a few layers into copy operations through multiple layers.
    This will automatically find the minimal MLP architecture to represent the NEAT graph. 
    """
    node2activation = {n['id']: n['activation'] for n in nn['nodes']}
    node2out_links = {n['id']: [(l['target'], l['weight']) for l in nn['links'] if l['source'] == n['id']] for n in nn['nodes']}
    node2in_links = {n['id']: [(l['source'], l['weight']) for l in nn['links'] if l['target'] == n['id']] for n in nn['nodes']}
    node_x = nn['special_nodes']['x']
    node_y = nn['special_nodes']['y']
    node_d = nn['special_nodes']['d']
    node_b = nn['special_nodes']['bias']
    node_h = nn['special_nodes']['h']
    node_s = nn['special_nodes']['s']
    node_v = nn['special_nodes']['v']

    outputs = do_forward_pass(nn)
    rgb, node2val = outputs['rgb'], outputs['node2val']

    node2layer = {node_x: 0, node_y: 0, node_d: 0, node_b: 0}
    i_layer = 1
    while True: # get the layer numbers of each node
        nodes_in_layer = []
        for node in node2val:
            if node in node2layer:
                continue
            if all(src in node2layer for src, _ in node2in_links[node]):
                nodes_in_layer.append(node)
        for node in nodes_in_layer:
            node2layer[node] = i_layer
        i_layer += 1
        if not nodes_in_layer:
            break
    
    n_layers = max(node2layer.values()) + 1
    nodes_cache = [[] for i in range(n_layers)] # what nodes are at each layer (layerized)
    for node in node2val: # calculate the cache of each node
        layer_start = node2layer[node]
        if node in [node_h, node_s, node_v]:
            layer_end = n_layers
        elif len(node2out_links[node])>0:
            layer_end = max([node2layer[target] for target, _ in node2out_links[node]])
        for i in range(layer_start, layer_end):
            nodes_cache[i].append((node, i==layer_start))
    
    nodes_cache[0] = [(node, False) for node, _ in nodes_cache[0]]
    
    from collections import defaultdict
    arch = defaultdict(int)
    for i_layer, nodes_layer in enumerate(nodes_cache):
        a = defaultdict(int)
        for node, novel in nodes_layer:
            a[node2activation[node] if novel else "cache"] += 1
        # print(dict(a))
        for k in a:
            arch[k] = max(arch[k], a[k])
    arch = dict(sorted(dict(arch).items()))
    width = sum(arch.values())
    # print("n_nodes=", len(node2val))
    # print(arch)
    # print("n_layers=", n_layers)
    # print("width=", width)
    # print("parameters=", width*width*n_layers)
    features = [jnp.stack([node2val[node] for node, _ in nodes_cache[i]], axis=-1) for i in range(n_layers)]
    def get_novelty(node, nodes_cache_layer):
        a = [novelty for n, novelty in nodes_cache_layer if n==node]
        assert len(a) == 1
        return a[0]
    nodes_cache[0] = [(node, get_novelty(node, nodes_cache[0])) for node in [node_x, node_y, node_d, node_b]]
    # nodes_cache[-1] = [(node, get_novelty(node, nodes_cache[-1])) for node in [node_h, node_s, node_v]]
    nodes_cache.append([(n, False) for n in [node_h, node_s, node_v]])

    n_layers = len(nodes_cache)-2
    arch = f'{n_layers};' + ','.join([f"{k}:{v}" for k, v in arch.items()])


    pbcppn = nn
    activation_neurons = arch.split(";")[1]
    cppn = CPPN(arch)
    cppn = FlattenCPPNParameters(cppn)
    params = jnp.zeros(cppn.n_params)
    params = cppn.param_reshaper.reshape_single(params)
    weight_mats = np.stack([get_weight_matrix(pbcppn, i_layer, nodes_cache, activation_neurons=activation_neurons) for i_layer in range(n_layers+1)])
    for i_layer in range(n_layers+1):
        a, b = params['params'][f'Dense_{i_layer}']['kernel'].shape
        w = weight_mats[i_layer][:a, :b]
        params['params'][f'Dense_{i_layer}']['kernel'] = jnp.array(w)[:a, :b]
    # a, b = params['params'][f'Dense_{n_layers-1}']['kernel'].shape
    # params['params'][f'Dense_{n_layers-1}']['kernel'] = jnp.eye(a, b)
    params = cppn.param_reshaper.flatten_single(params)
    return dict(rgb=rgb, node2val=node2val, node2layer=node2layer, features=features, nodes_cache=nodes_cache, arch=arch, cppn=cppn, params=params)

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
# group.add_argument("--seed", type=int, default=0, help="the random seed")

group = parser.add_argument_group("data")
group.add_argument("--zip_path", type=str, default=None, help="path to rep.zip")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    """
    Layerize the given raw picbreeder genome and save the cppn parameters and image to a directory.
    """
    pbcppn = load_pbcppn(args.zip_path)
    layerize_outputs = layerize_nn(pbcppn)
    arch, params = layerize_outputs['arch'], layerize_outputs['params']
    cppn = FlattenCPPNParameters(CPPN(arch))
    img, features = cppn.generate_image(params, return_features=True, img_size=256)
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        plt.imsave(f"{args.save_dir}/img.png", np.array(img))
        util.save_pkl(args.save_dir, "pbcppn", pbcppn)
        util.save_pkl(args.save_dir, "arch", arch)
        util.save_pkl(args.save_dir, "params", params)

if __name__=="__main__":
    main(parse_args())
