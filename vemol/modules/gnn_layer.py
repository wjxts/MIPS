import dgl
from dgl.nn.functional import edge_softmax
import torch 
import torch.nn as nn

GNN_LAYERS = {}
def register_gnn_layer(name):
    def decorator(gnn_layer):
        GNN_LAYERS[name] = gnn_layer
        return gnn_layer
    return decorator 


# 所有graph data里所有local的命名都要加下划线！ 来和输入的data区分
@register_gnn_layer('gcn')
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_feats))
        self.act = nn.ReLU()
        # self.observer = Observer()
        
    
    def forward(self, g, feature):
        # g应该加self-loop
        # feature = self.observer(feature)
        with g.local_scope():
            g = dgl.remove_self_loop(g) # 避免重复加self-loop
            g = dgl.add_self_loop(g)
            degrees = g.in_degrees().float().clamp(min=1).unsqueeze(-1)
            g.ndata['_pool_h'] = self.linear(feature/torch.sqrt(degrees))
            g.update_all(dgl.function.copy_u('_pool_h', '_m'), dgl.function.sum('_m', '_pool_h'))
            h = g.ndata['_pool_h']/torch.sqrt(degrees)
            h = h + self.bias
            h = self.act(h)
            return h

@register_gnn_layer('gin')
class GINLayer(nn.Module):
    def __init__(self, in_feats, out_feats, learn_eps=False, init_eps=0.) -> None:
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        if learn_eps:
            # self.eps = nn.Parameter(torch.Tensor([init_eps]))
            self.register_parameter('eps', nn.Parameter(torch.Tensor([init_eps])))
        else:
            self.register_buffer('eps', torch.Tensor([init_eps]))
        self.act = nn.ReLU()
    
    def forward(self, g, feature):
        # g不应该包含self-loop
        with g.local_scope():
            # g = dgl.remove_self_loop(g)
            g.ndata['_pool_h'] = feature
            g.update_all(dgl.function.copy_u('_pool_h', '_m'), dgl.function.sum('_m', '_pool_h'))
            # h = self.linear((1+self.eps)*g.ndata['h'])
            h = (1+self.eps)*feature + g.ndata['_pool_h']
            h = self.linear(h)
            h = self.act(h)
            return h

@register_gnn_layer('gat')
class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=2):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, out_feats*2//num_heads)) # 每个head不同
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)
        self.attn_linear = nn.Linear(in_feats, out_feats, bias=False)
        self.num_heads = num_heads
        self.out_feats = out_feats
        
    def forward(self, g, feature):
        with g.local_scope():
            g = dgl.remove_self_loop(g) # 避免重复加self-loop
            g = dgl.add_self_loop(g)
            g.ndata['_attn_h'] = feature
            def cal_score_func(edges):
                src = edges.src 
                dst = edges.dst
                # print(src['x'].shape, dst['x'].shape)
                src_key = self.attn_linear(src['_attn_h']).reshape(-1, self.num_heads, self.out_feats//self.num_heads)
                dst_key = self.attn_linear(dst['_attn_h']).reshape(-1, self.num_heads, self.out_feats//self.num_heads)
                key = torch.cat([src_key, dst_key], dim=-1)
                score = (key * self.a.unsqueeze(0)).sum(-1)
                score = self.attn_act(score)
                return {'_score': score}

            g.apply_edges(cal_score_func)
            g.edata['_attn'] = edge_softmax(g, g.edata['_score'])
            def message_func(edges):
                src = edges.src 
                dst = edges.dst
                message = g.edata['_attn'].unsqueeze(-1) * self.linear(src['_attn_h']).reshape(-1, self.num_heads, self.out_feats//self.num_heads)
                message = message.reshape(-1, self.out_feats) # 拼接multi-head的输出
                return {'_m': message}

            def reduce_func(nodes):
                # print(nodes.mailbox['m'].shape) # 会把相同度的节点的消息放在一起
                return {'_newh': nodes.mailbox['_m'].sum(1)}
            g.update_all(message_func, reduce_func)
            h = g.ndata['_newh']
            return h

@register_gnn_layer('gat_v2')
class GATLayerV2(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=2):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, out_feats//num_heads)) # 每个head不同
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)
        self.attn_linear_src = nn.Linear(in_feats, out_feats, bias=False)
        self.attn_linear_dst = nn.Linear(in_feats, out_feats, bias=False)
        self.num_heads = num_heads
        self.out_feats = out_feats
        
    def forward(self, g, feature):
        with g.local_scope():
            g = dgl.remove_self_loop(g) # 避免重复加self-loop
            g = dgl.add_self_loop(g)
            g.ndata['_attn_h'] = feature
            def cal_score_func(edges):
                src = edges.src 
                dst = edges.dst
                # print(src['x'].shape, dst['x'].shape)
                src_key = self.attn_linear_src(src['_attn_h']).reshape(-1, self.num_heads, self.out_feats//self.num_heads)
                dst_key = self.attn_linear_dst(dst['_attn_h']).reshape(-1, self.num_heads, self.out_feats//self.num_heads)
                score = self.attn_act(score)
                score = (src_key * self.a.unsqueeze(0)).sum(-1) + (dst_key * self.a.unsqueeze(0)).sum(-1)
                return {'_score': score}

            g.apply_edges(cal_score_func)
            g.edata['_attn'] = edge_softmax(g, g.edata['_score'])
            def message_func(edges):
                src = edges.src 
                dst = edges.dst
                message = g.edata['_attn'].unsqueeze(-1) * self.linear(src['_attn_h']).reshape(-1, self.num_heads, self.out_feats//self.num_heads)
                message = message.reshape(-1, self.out_feats) # 拼接multi-head的输出
                return {'_m': message}

            def reduce_func(nodes):
                # print(nodes.mailbox['m'].shape) # 会把相同度的节点的消息放在一起
                return {'_newh': nodes.mailbox['_m'].sum(1)}
            g.update_all(message_func, reduce_func)
            h = g.ndata['_newh']
            return h
        
def test_gin():
    from dgl.nn.pytorch.conv import GINConv
    g = dgl.graph(([0, 1, 2], [1, 2, 0]))
    g = dgl.add_reverse_edges(g)
    input_dim = 3
    output_dim = 4
    print(g.num_nodes())
    x = torch.randn(g.num_nodes(), input_dim) 
    g.ndata['h'] = x
    print(g.ndata['h'])
    
    m1 = GINLayer(input_dim, output_dim)
    m2 = GINConv(m1.linear, 'sum', activation=nn.ReLU())
    y1 = m1(g, g.ndata['h'])
    y2 = m2(g, g.ndata['h'])
    y3 = m1.linear(x.sum(dim=0, keepdim=True))
    print(y1, y2, y3, sep='\n')

def test_gat():
    g = dgl.graph(([0, 1, 2], [1, 2, 0]))
    g = dgl.add_reverse_edges(g)
    input_dim = 3
    output_dim = 4
    x = torch.randn(g.num_nodes(), input_dim) 
    m1 = GATLayer(input_dim, output_dim, num_heads=2)
    m2 = GATLayerV2(input_dim, output_dim, num_heads=2)
    y1 = m1(g, x)
    y2 = m2(g, x)
    print(y1, y2, sep='\n')

if __name__ == "__main__":
    # test_gin()
    test_gat()
    