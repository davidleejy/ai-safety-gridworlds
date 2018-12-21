import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module. '''

    def __init__(self, n_heads, dk, dv, lq, lk, lv):
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

        self.wq = torch.nn.Linear(lq, n_heads*dk)
        self.wk = torch.nn.Linear(lk, n_heads*dk)
        self.wv = torch.nn.Linear(lv, n_heads*dv)

        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k))) # TODO necessary?
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.softmax = nn.Softmax(dim=-1)

        self.layer_norm = nn.LayerNorm(normalized_shape=lq, eps=1e-5)
        self.wa = torch.nn.Linear(n_heads*dv, lq)

        # self.fc = nn.Linear(n_head * d_v, d_model)
        # nn.init.xavier_normal_(self.fc.weight) # TODO necessary?

        # self.dropout = nn.Dropout(dropout) # TODO necessary?


    def forward(self, q, k, v):
        '''
        Args:
            q: queries. torch tensor shaped Nq x lq. Example: Nq = 64 for a 8x8 gridworld where each square may contain an entity - so 64 entities in total. lq is no. of dimensions each entity is afforded. Suppose each entity is described with a vector with 3 elements (x,y,entitytype), then lq=3.
            k: keys. torch tensor shaped Nk x lk. Nk is number of keys. lk is no. of dimensions of each key vector.
            v: values. torch tensor shaped Nv x lv. Nv is number of values. lv is no. of dimensions of each value vector. Note that Nk == Nv because each key must be paired with a value.
        Returns:
            output: torch tensor shaped Nq x lq.
            attention: scaled attention. Torch tensor shaped n_heads x Nq x Nk.
        '''
        dk, dv, H = self.dk, self.dv, self.n_heads
        residual = q
        Nq, _ = q.shape
        Nk, _ = k.shape
        Nv, _ = v.shape
        Q = self.wq(q) # shape Nq x H*dk
        K = self.wk(k)
        V = self.wv(v)
        Q = Q.view(Nq, H, dk)
        K = K.view(Nk, H, dk)
        V = V.view(Nv, H, dv)
        V = V.permute(1,0,2) # shape H x Nv x dv
        Q = Q.permute(1,0,2) # shape H x Nq x dk
        K_t = K.permute(1,2,0) # shape H x dk x Nk

        attention = self.softmax( torch.bmm(Q, K_t) * (1 / (dk**0.5)) )
        A = torch.bmm(attention, V) # shape H x Nq x Nk
        A = A.permute(1,0,2) # shape Nq x H x dv
        A = A.contiguous().view(Nq, H*dv) # shape Nq x H*dv
        Y = self.wa(A)
        Y = Y + residual
        # output = self.dropout(self.fc(Y))  # TODO necessary?
        output = self.layer_norm(Y)
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
    No, lo = output.shape
    relational_block2 = MultiHeadAttention(n_heads=H, dk=dk, dv=dv, lq=lo, lk=lo, lv=lo)
    output2, attention2 = relational_block2(output,output,output)

    # Example of [1,3,5] relational block config:
    Ne=64; le=5
    E = torch.randn(Ne,le)
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
    O2 = torch.cat([O21,O22,O23], dim=1)
    O31, A31 = relational_block_3_1(O2,O2,O2)
    O32, A32 = relational_block_3_2(O2,O2,O2)
    O33, A33 = relational_block_3_3(O2,O2,O2)
    O3 = torch.cat([O31,O32,O33], dim=1)
    print('output shape', O3.shape)
