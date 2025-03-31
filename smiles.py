import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,root_mean_squared_error
from bayes_opt import BayesianOptimization
import math

# 数据加载函数（修正版）
def load_data(file_path):
    """加载CSV数据并正确分割各列"""
    df = pd.read_csv(file_path, header=None)

    # 列索引定义（根据问题描述）
    smiles_col = 0  # 第1列：SMILES
    feature_start = 1  # 第2列开始的特征
    feature_end = -1  # 倒数第二列截止
    target_col = -1  # 最后一列：目标值

    # 数据分割
    X_smiles = df.iloc[:, smiles_col].values.astype(str)
    X_numerical = df.iloc[:, feature_start:feature_end].values.astype(float)
    y = df.iloc[:, target_col].values.astype(float)

    print(f"数据加载完成：共{len(X_smiles)}条样本")
    print(f"特征维度：SMILES序列 x1, 数值特征 {X_numerical.shape[1]}列")
    return X_smiles, X_numerical, y


# 数据预处理函数（带维度验证）
def preprocess_data(X_smiles, X_numerical, max_length=100):
    """预处理流程包含数据验证"""
    # 验证数值特征维度
    assert X_numerical.shape[1] == 4, "必须包含17个数值特征（温度、压力、体积、泄漏尺寸）"
     # SMILES编码处理
    all_chars = set(char for sm in X_smiles for char in sm)
    vocab = {char: i + 1 for i, char in enumerate(all_chars)}  # 0为填充字符
    vocab_size = len(vocab) + 1

    # SMILES序列编码和填充
    X_encoded = []
    for sm in X_smiles:
        encoded = [vocab.get(c, 0) for c in sm[:max_length]]
        padded = encoded + [0] * (max_length - len(encoded))
        X_encoded.append(padded)

    # 数值特征标准化
    num_scaler = StandardScaler()
    X_num = num_scaler.fit_transform(X_numerical)

    return np.array(X_encoded), X_num, vocab_size, num_scaler


# 先定义CrossAttention类
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, smiles_feat, num_feat):
        Q = self.query(smiles_feat).unsqueeze(1)  # (B, 1, d_model)
        K = self.key(num_feat).unsqueeze(1)  # (B, 1, d_model)
        V = self.value(num_feat).unsqueeze(1)  # (B, 1, d_model)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, 1, 1)
        attn_weights = torch.softmax(attn_scores / math.sqrt(Q.size(-1)), dim=-1)
        output = torch.matmul(attn_weights, V).squeeze(1)  # (B, d_model)
        return output

# Transformer模型（保持结构）
class SMILESTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3,num_feat_dim=4):
        super().__init__()
        # 参数验证
        if nhead <= 0:
            raise ValueError(f"num_heads ({nhead}) must be > 0")
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
            # 设置 batch_first=True
        encoder_layers = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True  # 关键参数
            )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.num_proj = nn.Linear(num_feat_dim, d_model)  # 数值特征投影
        self.cross_attn = CrossAttention(d_model)
        self.d_model = d_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, src, num_feat):
        # 设备映射
        src = src.to(self.device)
        num_feat = num_feat.to(self.device)

        # SMILES处理
        src = src.permute(1, 0)  # (S, B)
        src = self.embedding(src) * math.sqrt(self.d_model)
        trans_output = self.transformer(src)  # (S, B, d_model)
        smiles_feat = torch.mean(trans_output, dim=0)  # (B, d_model)

        # 数值特征投影
        num_feat = self.num_proj(num_feat)  # (B, d_model)

        # 交叉注意力
        attn_output = self.cross_attn(smiles_feat, num_feat)  # (B, d_model)

        # 特征拼接
        return torch.cat([smiles_feat, attn_output], dim=1)  # (B, 2*d_model)


# 特征提取流程（带维度校验）
def extract_features(model, smiles_data, numerical_data, batch_size=32):
    model.eval()
    features = []

    # 创建数据集并验证维度匹配
    assert len(smiles_data) == len(numerical_data), "SMILES与数值特征数量不匹配"
    dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(smiles_data),
        torch.FloatTensor(numerical_data)
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for smiles, nums in loader:
            smiles = smiles.to(model.device)
            nums = nums.to(model.device)
            # 调整维度：batch_size x seq_len
            if len(smiles.shape) == 1:
                smiles = smiles.unsqueeze(1)
            # 前向传播
            combined_feat = model(smiles, nums)
            # 移动到CPU并转为numpy
            features.append(combined_feat.cpu().numpy())

    return np.vstack(features)


# 贝叶斯优化流程（作用域修正）
def svm_eval(d_model, nhead, num_layers, C, epsilon, gamma):

    """访问全局数据变量"""
    global X_smiles, X_numerical, y  # 明确声明使用全局变量

    # 强制参数约束
    valid_nheads = [2, 4, 8]  # 128的有效因数

    # 将贝叶斯优化的浮点数映射到离散索引
    index = int(nhead) % len(valid_nheads)  # 确保索引在0-2之间
    selected_nhead = valid_nheads[index]  # 映射后的合法值

    # 数据预处理
    X_enc, X_num, vocab_size, _ = preprocess_data(X_smiles, X_numerical)

    # 数据分割
    X_train_enc, X_test_enc, X_train_num, X_test_num, y_train, y_test = \
        train_test_split(X_enc, X_num, y, test_size=0.2, random_state=42)

    # 模型初始化
    model = SMILESTransformer(
        vocab_size,
        d_model=128,              # 固定为128
        nhead=selected_nhead,      # 使用映射后的合法值
        num_layers=int(num_layers)
    )

    # 特征提取
    train_features = extract_features(model, X_train_enc, X_train_num)

    # 训练SVR
    svr = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel='rbf')
    svr.fit(train_features, y_train)

    # 验证评估
    test_features = extract_features(model, X_test_enc, X_test_num)
    return -np.mean((svr.predict(test_features) - y_test) ** 2)  # 负MSE


# 主程序流程（完整作用域管理）
if __name__ == "__main__":
    # 1. 数据加载（必须在最外层作用域）
    data_path = "S5.csv"
    X_smiles, X_numerical, y = load_data(data_path)

    # 2. 参数空间定义

    pbounds = {
        'd_model': (128, 256),
        'nhead': (0, 2.99),
        'num_layers': (1, 4),
        'C': (1e-3, 1e3),
        'epsilon': (0.01, 0.2),
        'gamma': (1e-5, 1e1)
    }

    # 3. 贝叶斯优化
    optimizer = BayesianOptimization(
        f=svm_eval,
        pbounds=pbounds,
        random_state=42
    )
    optimizer.maximize(init_points=8, n_iter=25)

    # 4. 最终模型训练
    best_params = optimizer.max['params']
    X_enc, X_num, vocab_size, num_scaler = preprocess_data(X_smiles, X_numerical)

    final_model = SMILESTransformer(
        vocab_size,
        d_model=int(best_params['d_model']),
        nhead=int(best_params['nhead']),
        num_layers=int(best_params['num_layers'])
    )

    # 提取全部特征
    features = extract_features(final_model, X_enc, X_num)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

    # 训练最终模型
    svr = SVR(
        C=best_params['C'],
        epsilon=best_params['epsilon'],
        gamma=best_params['gamma'],
        kernel='rbf'
    )
    svr.fit(X_train, y_train)

    # 最终评估

    y_pred2 = svr.predict(X_test)
    print(f"测试集MSE: {mean_squared_error(y_test, y_pred2):.4f}")
    print(f"测试集R²: {r2_score(y_test, y_pred2):.4f}")
    print(f"rmse:{root_mean_squared_error(y_test, y_pred2):.4f}")
    print(f"MAE:{mean_absolute_error(y_test, y_pred2):.4f}")
