import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module. '''

    def __init__(self, n_heads, dk, dv, lq, lk, lv, reduce_heads, reduce_entities, transform_last, residual_conn, norm):
        '''
        Args:
            n_heads: number of heads.
            dk: number of dimensions you would like each key vector to be projected to.
            dv: number of dimensions you would like each value vector to be projected to.
            lq: number of dimensions of each query vector fed in. This is before linearly projecting each
                query vector to have dv dimensions.
            lk: number of dimensions of each key vector fed in. This is before linearly projecting each
                key vector to have dk dimensions.
            lv: number of dimensions of each value vector fed in. This is before linearly projecting each
                value vector to have dv dimensions.
            reduce_heads: method for reducing the output for the many heads. Options "concat", "sum".
            reduce_entities: method for reducing over the entities. Options "off", "sum".
            transform_last: apply the last linear layer as is common in such architectures if True. False otherwise.
            residual_conn: residual connection to q if True. False otherwise.
            norm: apply normalization at end. Options "default", "off".
        Example:
            The "Attention is all you need" paper sets n_heads=8, dk=dv=64, lq=lk=lv=128.
        '''
        super().__init__()

        self.n_heads = n_heads
        self.dk = dk
        self.dv = dv
        self.lq = lq
        self.lk = lk
        self.lv = lv
        self.reduce_heads = reduce_heads
        self.reduce_entities = reduce_entities
        self.transform_last = transform_last
        self.residual_conn = residual_conn
        self.norm = norm
        
        self.wq = torch.nn.Linear(lq, n_heads*dk)
        self.wk = torch.nn.Linear(lk, n_heads*dk)
        self.wv = torch.nn.Linear(lv, n_heads*dv)

        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k))) # TODO necessary?
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.softmax = nn.Softmax(dim=-1)

        if reduce_heads == 'concat':
            layernorm_shape = n_heads*dv
        elif self.reduce_heads == 'sum':
            layernorm_shape = dv

        if transform_last:
            if self.reduce_heads == 'concat':
                self.wa = torch.nn.Linear(n_heads*dv, lq)
            elif self.reduce_heads == 'sum':
                self.wa = torch.nn.Linear(dv, lq)
            layernorm_shape = self.wa.out_features


        if norm == 'default':
            self.layer_norm = nn.LayerNorm(normalized_shape=[layernorm_shape], eps=1e-5)

        # self.fc = nn.Linear(n_head * d_v, d_model)
        # nn.init.xavier_normal_(self.fc.weight) # TODO necessary?

        # self.dropout = nn.Dropout(dropout) # TODO necessary?


    def forward(self, q, k, v):
        '''
        Args:
            q: queries. torch tensor shaped bs x Nq x lq. Example: Nq = 64 for a 8x8 gridworld where each square may contain an entity - so 64 entities in total. lq is no. of dimensions each entity is afforded. Suppose each entity is described with a vector with 3 elements (x,y,entitytype), then lq=3. bs is batch size.
            k: keys. torch tensor shaped bs x Nk x lk. Nk is number of keys. lk is no. of dimensions of each key vector. bs is batch size.
            v: values. torch tensor shaped bs x Nv x lv. Nv is number of values. lv is no. of dimensions of each value vector. Note that Nk == Nv because each key must be paired with a value. bs is batch size.
        Returns:
            output: torch tensor shaped bs x Nq x lq.
            attention: scaled attention. Torch tensor shaped bs x n_heads x Nq x Nk.
        '''
        dk, dv, H = self.dk, self.dv, self.n_heads
        residual = q
        bs, Nq, _ = q.shape
        bs, Nk, _ = k.shape
        bs, Nv, _ = v.shape
        Q = self.wq(q) # shape bs x Nq x H*dk
        K = self.wk(k) # shape bs x Nk x H*dk
        V = self.wv(v)
        Q = Q.view(bs, Nq, H, dk)
        K = K.view(bs, Nk, H, dk)
        V = V.view(bs, Nv, H, dv)
        V = V.permute(0,2,1,3) # shape bs x H x Nv x dv
        Q = Q.permute(0,2,1,3) # shape bs x H x Nq x dk
        K_t = K.permute(0,2,3,1) # shape bs x H x dk x Nk
        attention = self.softmax( torch.matmul(Q, K_t) * (1 / (dk**0.5)) ) # shape bs x H x Nq x Nk
        A = torch.matmul(attention, V) # shape bs x H x Nq x dv
        if self.reduce_heads == 'concat':
            A.permute(0,2,1,3) # bs x Nq x H x dv
            A = A.contiguous().view(bs, Nq, H*dv) # shape bs x Nq x H*dv
        elif self.reduce_heads == 'sum':
            A.permute(0,2,1,3) # bs x Nq x H x dv
            A = torch.sum(A, dim=-2, keepdim=False) # bs x Nq x dv
        else:
            NotImplementedError
        if self.reduce_entities == 'sum':
            A = torch.sum(A, dim=-2, keepdim=True) # bs x 1 x dv  or  bs x 1 x H*dv
        elif self.reduce_entities == 'off':
            pass
        else:
            NotImplementedError
        if self.transform_last:
            Y = self.wa(A)
        else:
            Y = A
        if self.residual_conn:
            Y = Y + residual
        # output = self.dropout(self.fc(Y))  # TODO necessary?
        if self.norm == 'default':
            output = self.layer_norm(Y)
        elif self.norm == 'off':
            output = Y
        else:
            NotImplementedError
        return output, attention


if __name__ == "__main__":
    
    # Example of stacking 2 relational blocks together:
    Nq = 64; lq = 5
    Nk = 60; lk = 4
    Nv = 60; lv = 5
    H = 3
    batchsize = 11
    q = torch.randn(batchsize,Nq,lq)
    k = torch.randn(batchsize,Nk,lk)
    v = torch.randn(batchsize,Nv,lv)
    dk = 7; dv = 9
    relational_block = MultiHeadAttention(n_heads=H, dk=dk, dv=dv, lq=lq, lk=lk, lv=lv)
    output, attention = relational_block(q=q,k=k,v=v)
    bs, No, lo = output.shape
    relational_block2 = MultiHeadAttention(n_heads=H, dk=dk, dv=dv, lq=lo, lk=lo, lv=lo)
    output2, attention2 = relational_block2(output,output,output)

    # Example of [1,3,5] relational block config:
    Ne=64; le=5; batchsize=1
    E = torch.randn(batchsize, Ne,le)
    relational_block_1 = MultiHeadAttention(n_heads=3, dk=7, dv=9, lq=le, lk=le, lv=le)
    relational_block_2_1 = MultiHeadAttention(n_heads=4, dk=7, dv=10, lq=le, lk=le, lv=le)
    relational_block_2_2 = MultiHeadAttention(n_heads=3, dk=6, dv=9, lq=le, lk=le, lv=le)
    relational_block_2_3 = MultiHeadAttention(n_heads=2, dk=5, dv=8, lq=le, lk=le, lv=le)
    relational_block_3_1 = MultiHeadAttention(n_heads=5, dk=7, dv=9, lq=3*le, lk=3*le, lv=3*le)
    relational_block_3_2 = MultiHeadAttention(n_heads=3, dk=6, dv=8, lq=3*le, lk=3*le, lv=3*le)
    relational_block_3_3 = MultiHeadAttention(n_heads=2, dk=5, dv=6, lq=3*le, lk=3*le, lv=3*le)
    O1, A1 = relational_block_1(E,E,E)
    O21, A21 = relational_block_2_1(O1,O1,O1)
    O22, A22 = relational_block_2_2(O1,O1,O1)
    O23, A23 = relational_block_2_3(O1,O1,O1)
    O2 = torch.cat([O21,O22,O23], dim=-1)
    O31, A31 = relational_block_3_1(O2,O2,O2)
    O32, A32 = relational_block_3_2(O2,O2,O2)
    O33, A33 = relational_block_3_3(O2,O2,O2)
    O3 = torch.cat([O31,O32,O33], dim=-1)
    print('output shape', O3.shape)
