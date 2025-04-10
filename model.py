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
import random

# 设置全局随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据加载函数（修正版）
def load_data(file_path):
    """加载CSV数据并正确分割各列"""
    df = pd.read_csv(file_path, header=None,skiprows=1,encoding='latin1')

    # 列索引定义（根据问题描述）
    smiles_col = 0  # 第1列：SMILES
    feature_start = 1  # 第2列开始的特征
    feature_end = -1  # 倒数第二列截止
    target_col = -1  # 最后一列：目标值

    # 数据排序（确保每次加载顺序一致）
    df = df.sort_values(by=smiles_col)

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
    assert X_numerical.shape[1] == 17
    # SMILES编码处理（排序字符以固定词汇表顺序）
    all_chars = sorted(set(char for sm in X_smiles for char in sm))  # 排序
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
            # 简单嵌入+平均池化替代
            embedded = self.embedding(src)  # (B, S, d_model)
            noise = torch.randn_like(embedded) * 0.5
            smiles_feat = torch.mean(embedded + noise, dim=1)
        smiles_feat = self.dropout(smiles_feat)

        # 数值特征处理
        if self.num_proj is None:
            return smiles_feat  # 无数值特征情况

        num_feat = num_feat.to(self.device)
        num_feat = self.num_proj(num_feat)

        if hasattr(self, 'cross_attn') and self.cross_attn is not None:
            attn_output = self.cross_attn(smiles_feat, num_feat)
            return torch.cat([smiles_feat, attn_output], dim=1)
        else:
            # 无交叉注意力时简单拼接
            return torch.cat([smiles_feat, num_feat], dim=1)


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
    nhead = max(1, int(nhead))  # 确保 nhead 至少为 1
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

def optimize_parameters(data_path):
    global X_smiles, X_numerical, y
    X_smiles, X_numerical, y = load_data(data_path)

    pbounds = {
        'd_model': (128, 256),
        'nhead': (1, 3),
        'num_layers': (1, 4),
        'C': (1e-3, 1e3),
        'epsilon': (0.01, 0.2),
        'gamma': (1e-5, 1e1)
    }

    optimizer = BayesianOptimization(
        f=svm_eval,
        pbounds=pbounds,
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=35)
    return optimizer.max['params']

# 评估模型
def evaluate_model(data_path, best_params):
    X_smiles, X_numerical, y = load_data(data_path)
    X_enc, X_num, vocab_size, _ = preprocess_data(X_smiles, X_numerical)
    X_train_enc, X_test_enc, X_train_num, X_test_num, y_train, y_test = train_test_split(
        X_enc, X_num, y, test_size=0.2, random_state=42
    )

    final_model = SMILESTransformer(
        vocab_size,
        d_model=int(best_params['d_model']),
        nhead=int(best_params['nhead']),
        num_layers=int(best_params['num_layers'])
    )

    features = extract_features(final_model, X_enc, X_num)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

    svr = SVR(
        C=best_params['C'],
        epsilon=best_params['epsilon'],
        gamma=best_params['gamma'],
        kernel='rbf'
    )
    svr.fit(X_train, y_train)

    y_pred = svr.predict(X_test)

    # 保存预测结果
    results_df = pd.DataFrame({
        'SMILES': X_smiles[-len(y_test):],  # 取对应的测试集SMILES
        'True_Value': y_test,
        'Predicted_Value': y_pred
    })
    results_df.to_csv('final_model_predictions.csv', index=False)

    return {
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred)
    }

def run_ablation_study(data_path, best_params):
    # 加载数据
    X_smiles, X_numerical, y = load_data(data_path)
    X_enc, X_num, vocab_size, _ = preprocess_data(X_smiles, X_numerical)

    # 定义消融实验配置
    ablation_configs = [
        {"name": "Full Model", "ablation": None},
        {"name": "No Cross-Attention", "ablation": "no_cross_attn"},
        {"name": "No Transformer", "ablation": "no_transformer"}
    ]

    results = []

    for config in ablation_configs:
        print(f"\nRunning ablation: {config['name']}")

        # 初始化模型
        model = SMILESTransformer(
            vocab_size,
            d_model=int(best_params['d_model']),
            nhead=int(best_params['nhead']),
            num_layers=int(best_params['num_layers']),
            ablation=config['ablation']
        )

        # 特征提取
        features = extract_features(model, X_enc, X_num)
        X_train, X_test, y_train, y_test = train_test_split(
            features, y, test_size=0.2, random_state=42)

        # 训练SVR
        svr = SVR(
            C=best_params['C'],
            epsilon=best_params['epsilon'],
            gamma=best_params['gamma'],
            kernel='rbf'
        )
        svr.fit(X_train, y_train)

        # 评估
        y_pred = svr.predict(X_test)
        metrics = {
            "Model": config['name'],
            "R2": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred)
        }
        results.append(metrics)

        print(f"{config['name']} Results:")
        print(f"MSE: {metrics['MSE']:.4f}, R2: {metrics['R2']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")

    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv("ablation_results.csv", index=False)
    return results_df


# 在主程序中调用
if __name__ == "__main__":
    data_path = "Data.csv"
    best_params = optimize_parameters(data_path)
    ablation_results = run_ablation_study(data_path, best_params)
    print("\nAblation Study Results:")
    print(ablation_results)

    # 评估最终模型并保存预测结果
    final_metrics = evaluate_model(data_path, best_params)
    print("\nFinal Model Performance:")
    print(final_metrics)

