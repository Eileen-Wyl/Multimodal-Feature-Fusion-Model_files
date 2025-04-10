# QSCP_files
The repository contains an archive of scripts and data files that can be used to reproduce the development and validation of QSCR models that predict explosion diameters based on chemical structures. The library was established at the request of the Journal of Chemical Informatics.
Make sure you have installed the Python libraries you rely on in your code, such as pandas, numpy, torch, scikit-learn, bayesian optimization, and so on. Next, place the CSV file in the directory where the code file is located (or modify the file path in the code to point to the correct file location) and run the code file smiles.py directly. The code will automatically load data for pre-processing, extract features by the SMILESTransformer model defined, search for the optimal model parameters using Bayesian optimization method, and finally train the final SVR model and evaluate it on the test set, and output evaluation indicators such as MSE, R², RMSE and MAE. Ablation experiments were conducted to compare the full model, no cross-attention and no Transformer

======================================================================================================================================
Example code snippets for model predicted
class SMILESTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3,
                 num_feat_dim=17, dropout=0.1, ablation=None):
        super().__init__()
        self.ablation = ablation

        # 动态调整nhead
        while d_model % nhead != 0:
            nhead -= 1
            if nhead <= 0:
                raise ValueError(f"Unable to find valid nhead for d_model={d_model}")

        # 基本组件
        self.embedding = nn.Embedding(vocab_size, d_model)
        if ablation != "no_transformer":
            encoder_layers = TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=0.5,batch_first=True)
            self.transformer = TransformerEncoder(encoder_layers, num_layers)
        else:
            self.transformer = None

        self.num_proj = nn.Linear(num_feat_dim, d_model)
        if ablation != "no_cross_attn":
                self.cross_attn = CrossAttention(d_model)
        else:
            self.num_proj = None

        self.d_model = d_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, num_feat):
        src = src.to(self.device)

        # SMILES特征提取
        if self.transformer is not None:
            src = src.permute(1, 0)
            src = self.embedding(src) * math.sqrt(self.d_model)
            trans_output = self.transformer(src)
            smiles_feat = torch.mean(trans_output, dim=0)
        else:
          
            embedded = self.embedding(src)  # (B, S, d_model)
            noise = torch.randn_like(embedded) * 0.5
            smiles_feat = torch.mean(embedded + noise, dim=1)
        smiles_feat = self.dropout(smiles_feat)

        if self.num_proj is None:
            return smiles_feat  

        num_feat = num_feat.to(self.device)
        num_feat = self.num_proj(num_feat)

        if hasattr(self, 'cross_attn') and self.cross_attn is not None:
            attn_output = self.cross_attn(smiles_feat, num_feat)
            return torch.cat([smiles_feat, attn_output], dim=1)
        else:
            return torch.cat([smiles_feat, num_feat], dim=1)
======================================================================================================================================
Expected output formats
|   iter    |  target   |     C     |  d_model  |  epsilon  |   gamma   |   nhead   | num_la... |
-------------------------------------------------------------------------------------------------
| 1         | -0.09592  | 374.5     | 249.7     | 0.1491    | 5.987     | 1.312     | 1.468     |
| 2         | -0.09101  | 58.08     | 238.9     | 0.1242    | 7.081     | 1.041     | 3.91      |
| 3         | -0.03429  | 832.4     | 155.2     | 0.04455   | 1.834     | 1.608     | 2.574     |

Ablation Study Results:
                Model        R2       MSE      RMSE       MAE
0          Full Model  0.947247  0.008697  0.093260  0.064001
1  No Cross-Attention  0.273161  0.119834  0.346171  0.283702
2      No Transformer  0.927225  0.011998  0.109537  0.079796

Final Model Performance:
{'MSE': 0.00876492692587948, 'R2': 0.9468374294574723, 'RMSE': 0.093621188445135, 'MAE': 0.06354326390656213}
