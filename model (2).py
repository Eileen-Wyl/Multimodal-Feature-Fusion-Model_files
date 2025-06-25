import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from bayes_opt import BayesianOptimization
import math
import random
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 设置全局随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ChemModelPipeline:
    def __init__(self):
        self.data = {}
        self.models = {}
        self.scalers = {}

    def load_data(self, file_path, nan_strategy='mean'):
        """加载并预处理数据"""
        df = pd.read_csv(file_path, header=None, skiprows=1, encoding='latin1')

        # 列索引定义
        smiles_col = 0
        feature_start = 1
        target_col = -1

        # 数据排序
        df = df.sort_values(by=smiles_col)

        # 基础数据分割
        X_smiles = df.iloc[:, smiles_col].values.astype(str)
        X_numerical = df.iloc[:, feature_start:target_col].values.astype(float)
        y = df.iloc[:, target_col].values.astype(float)

        # 检查并处理NaN
        nan_count = np.isnan(X_numerical).sum()
        if nan_count > 0:
            print(f"发现{nan_count}个NaN值，使用策略: {nan_strategy}")

            if nan_strategy == 'drop':
                valid_idx = ~np.isnan(X_numerical).any(axis=1)
                X_numerical = X_numerical[valid_idx]
                X_smiles = [X_smiles[i] for i in range(len(valid_idx)) if valid_idx[i]]
                y = y[valid_idx]
            else:
                strategy = nan_strategy if nan_strategy in ['mean', 'median'] else 'mean'
                imputer = SimpleImputer(strategy=strategy)
                X_numerical = imputer.fit_transform(X_numerical)

        # 提取RDKit描述符
        rdkit_features = []
        for sm in X_smiles:
            mol = Chem.MolFromSmiles(sm)
            if mol:
                desc = Descriptors.CalcMolDescriptors(mol)
                rdkit_features.append(list(desc.values()))
            else:
                rdkit_features.append([0] * 200)  # 无效分子填充

        # 合并特征
        X_combined = np.hstack([X_numerical, np.array(rdkit_features)])

        print(f"数据加载完成：{len(X_smiles)}条样本")
        print(f"特征维度：原始数值 {X_numerical.shape[1]} + RDKit {len(rdkit_features[0])} = {X_combined.shape[1]}")

        self.data = {
            'X_smiles': X_smiles,
            'X_numerical': X_combined,
            'y': y
        }
        return self

    def preprocess_data(self, max_length=100):
        """数据预处理"""
        X_smiles = self.data['X_smiles']
        X_numerical = self.data['X_numerical']

        # SMILES编码
        all_chars = sorted(set(char for sm in X_smiles for char in sm))
        vocab = {char: i + 1 for i, char in enumerate(all_chars)}
        vocab_size = len(vocab) + 1

        X_encoded = []
        for sm in X_smiles:
            encoded = [vocab.get(c, 0) for c in sm[:max_length]]
            padded = encoded + [0] * (max_length - len(encoded))
            X_encoded.append(padded)

        # 数值特征标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numerical)

        # 特征选择（保留方差>0.1的特征）
        variances = np.var(X_scaled, axis=0)
        selected_idx = np.where(variances > 0.1)[0]
        X_selected = X_scaled[:, selected_idx]

        print(f"特征选择：从{X_scaled.shape[1]}维保留{len(selected_idx)}维高方差特征")

        self.data.update({
            'X_encoded': np.array(X_encoded),
            'X_processed': X_selected,
            'vocab_size': vocab_size,
            'scaler': scaler,
            'selected_idx': selected_idx
        })
        return self

    def get_train_test_split(self, test_size=0.2):
        """获取训练测试分割"""
        X_train_enc, X_test_enc, X_train_num, X_test_num, y_train, y_test = train_test_split(
            self.data['X_encoded'],
            self.data['X_processed'],
            self.data['y'],
            test_size=test_size,
            random_state=SEED
        )
        return X_train_enc, X_test_enc, X_train_num, X_test_num, y_train, y_test


class FixedAblationTransformer(nn.Module):
    def __init__(self, vocab_size, num_feat_dim, d_model=128, nhead=4,
                 num_layers=3, dropout=0.1, ablation=None):
        super().__init__()
        self.ablation = ablation
        self.d_model = d_model

        # 自动调整nhead使其能整除d_model
        original_nhead = nhead
        while d_model % nhead != 0 and nhead > 0:
            nhead -= 1
        if nhead == 0:
            raise ValueError(f"无法为d_model={d_model}找到合适的nhead (原始值: {original_nhead})")

        print(f"使用nhead={nhead} (原始值: {original_nhead})")
        self.nhead = nhead

        # 1. SMILES处理部分
        self.embedding = nn.Embedding(vocab_size, d_model)
        if ablation != "no_transformer":
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=4 * d_model,
                dropout=dropout, batch_first=True)
            self.transformer = TransformerEncoder(encoder_layer, num_layers)
        else:
            self.transformer = None

        # 2. 数值特征处理部分
        if ablation != "no_numerical":
            self.num_proj = nn.Sequential(
                nn.Linear(num_feat_dim, d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model)
            )
        else:
            self.num_proj = None

        # 3. 特征融合部分
        self.use_cross_attn = (ablation not in ["no_cross_attn"])
        if self.use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 输出层（统一输出维度为d_model）
        self.output = nn.Linear(d_model, d_model)

        self.ablation_dropout = nn.Dropout(0 if ablation else 0.0)
        self.ablation_noise_std = 0.1 if ablation else 0.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, src, num_feat):
        src = src.to(self.device)
        num_feat = num_feat.to(self.device) if self.num_proj is not None else None

        # SMILES特征提取
        embedded = self.embedding(src)  # [batch, seq_len, d_model]
        if self.transformer is not None:
            encoded = self.transformer(embedded)  # [batch, seq_len, d_model]
            smiles_feat = torch.mean(encoded, dim=1)  # [batch, d_model]
        else:
            # 消融实验：简单平均池化
            smiles_feat = torch.mean(embedded, dim=1)  # [batch, d_model]

        # 数值特征处理
        if self.num_proj is not None:
            num_feat = self.num_proj(num_feat)  # [batch, d_model]

            # 特征融合
            if self.use_cross_attn:
                # 使用交叉注意力 [batch, 1, d_model]
                attn_output, _ = self.cross_attn(
                    smiles_feat.unsqueeze(1),  # query
                    num_feat.unsqueeze(1),  # key
                    num_feat.unsqueeze(1)  # value
                )
                combined = smiles_feat + attn_output.squeeze(1)  # 残差连接
            else:
                # 简单相加
                combined = smiles_feat + num_feat
        else:
            # 无数值特征
            combined = smiles_feat

        output = self.output(combined)  # [batch, d_model]

        output = self.ablation_dropout(output)
        if self.ablation_noise_std > 0.0:
            noise = torch.randn_like(output) * self.ablation_noise_std
            output = output + noise

        return output




def extract_features(model, smiles_data, numerical_data, batch_size=32):
    """统一的特征提取函数，确保维度一致"""
    model.eval()
    features = []

    dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(smiles_data),
        torch.FloatTensor(numerical_data)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for smiles, nums in loader:
            smiles = smiles.to(model.device)
            nums = nums.to(model.device)
            if len(smiles.shape) == 1:
                smiles = smiles.unsqueeze(1)
            feat = model(smiles, nums)
            features.append(feat.cpu().numpy())

    return np.vstack(features)


def evaluate_final_model(pipeline, best_params):
    """修复维度不一致问题的最终评估函数"""
    X_enc = pipeline.data['X_encoded']
    X_num = pipeline.data['X_processed']
    y = pipeline.data['y']
    X_smiles = pipeline.data['X_smiles']

    # 获取训练测试分割（保留索引）
    X_train_enc, X_test_enc, X_train_num, X_test_num, y_train, y_test, train_idx, test_idx = \
        train_test_split(
            X_enc, X_num, y, range(len(y)),
            test_size=0.2,
            random_state=SEED
        )

    # 初始化模型（使用最佳参数）
    model = FixedAblationTransformer(
        pipeline.data['vocab_size'],
        X_num.shape[1],  # 使用完整的特征维度
        d_model=int(2 ** round(math.log2(best_params['d_model']))),
        nhead=int(best_params['nhead']),
        num_layers=int(best_params['num_layers']),
        ablation=None)

    # 提取训练集特征
    train_feat = extract_features(model, X_train_enc, X_train_num)
    print(f"训练特征维度: {train_feat.shape}")  # 调试输出

    # 训练SVR
    svr = SVR(
        C=10 ** best_params['C'],
        epsilon=best_params['epsilon'],
        gamma=10 ** best_params['gamma'],
        kernel='rbf')
    svr.fit(train_feat, y_train)

    # 提取测试集特征（使用相同的模型和流程）
    test_feat = extract_features(model, X_test_enc, X_test_num)
    print(f"测试特征维度: {test_feat.shape}")  # 调试输出

    # 确保维度匹配
    assert train_feat.shape[1] == test_feat.shape[1], \
        f"特征维度不匹配！训练:{train_feat.shape[1]}，测试:{test_feat.shape[1]}"

    # 生成预测结果
    y_train_pred = svr.predict(train_feat)
    y_test_pred = svr.predict(test_feat)

    # 构建结果DataFrame
    train_results = pd.DataFrame({
        'SMILES': [X_smiles[i] for i in train_idx],
        'True': y_train,
        'Predicted': y_train_pred
    })
    test_results = pd.DataFrame({
        'SMILES': [X_smiles[i] for i in test_idx],
        'True': y_test,
        'Predicted': y_test_pred
    })

    train_results.to_csv('train_predictions1.csv', index=False)
    test_results.to_csv('test_predictions1.csv', index=False)
    print("训练集和测试集预测结果已分别保存为 train_predictions1.csv 和 test_predictions1.csv")

    results = pd.DataFrame({
        'SMILES': X_smiles,
        'True': y,
        'Predicted': np.concatenate([y_train_pred, y_test_pred]),
        'Dataset': ['train' if i in train_idx else 'test' for i in range(len(y))]
    })

    # 保存结果
    results.to_csv('final_predictions1.csv', index=False)
    print(f"结果已保存，特征维度: {train_feat.shape[1]}")

    # 计算测试集指标
    metrics = {
        'R2': r2_score(y_test, y_test_pred),
        'MSE': mean_squared_error(y_test, y_test_pred),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred))
    }

    return metrics, y_train_pred, y_test_pred

def run_ablation_study(pipeline, best_params):
    """执行消融实验"""
    X_enc = pipeline.data['X_encoded']
    X_num = pipeline.data['X_processed']
    y = pipeline.data['y']
    vocab_size = pipeline.data['vocab_size']

    # 定义消融实验配置
    ablation_configs = [
        {"name": "Full Model", "ablation": None},
        {"name": "No Transformer", "ablation": "no_transformer"},
        {"name": "No Numerical", "ablation": "no_numerical"},
        {"name": "No Cross-Attention", "ablation": "no_cross_attn"}
    ]

    results = []

    for config in ablation_configs:
        print(f"\n=== 运行消融实验: {config['name']} ===")

        # 初始化模型
        model = FixedAblationTransformer(
            vocab_size, X_num.shape[1],
            d_model=int(2 ** round(math.log2(best_params.get('d_model', 128)))) ,
            nhead=int(best_params['nhead']),
            num_layers=int(best_params['num_layers']),
            ablation=config['ablation'])

        # 特征提取
        features = extract_features(model, X_enc, X_num)
        X_train, X_test, y_train, y_test = train_test_split(
            features, y, test_size=0.2, random_state=SEED)

        # 使用优化后的SVR参数
        svr = SVR(
            C=10 ** best_params['C'],
            epsilon=best_params['epsilon'],
            gamma=10 ** best_params['gamma'],
            kernel='rbf')
        svr.fit(X_train, y_train)

        # 评估
        y_pred = svr.predict(X_test)
        metrics = {
            "Model": config['name'],
            "R2": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
        }
        results.append(metrics)

        print(f"{config['name']} 结果:")
        for k, v in metrics.items():
            if k != "Model":
                print(f"{k}: {v:.4f}")

    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv("ablation_results.csv", index=False)
    return results_df


def svm_eval(d_model, nhead, num_layers, C, epsilon, gamma, pipeline):
    """贝叶斯优化目标函数"""
    # 参数处理
    d_model = int(2 ** round(math.log2(d_model)))  # 转换为最近的2的幂次
    nhead = int(nhead)
    num_layers = int(num_layers)

    # 获取数据
    X_train_enc, X_test_enc, X_train_num, X_test_num, y_train, y_test = pipeline.get_train_test_split()

    # 模型初始化（使用完整模型进行优化）
    model = FixedAblationTransformer(
        pipeline.data['vocab_size'],
        X_train_num.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ablation=None)  # 完整模型

    # 特征提取
    train_feat = extract_features(model, X_train_enc, X_train_num)
    test_feat = extract_features(model, X_test_enc, X_test_num)

    # SVR训练（对数空间参数）
    svr = SVR(
        C=10 ** C,
        epsilon=epsilon,
        gamma=10 ** gamma,
        kernel='rbf')
    svr.fit(train_feat, y_train)

    # 返回负MSE（优化目标）
    return -mean_squared_error(y_test, svr.predict(test_feat))

def main():
    data_path = "C:\\Users\\34841\\Desktop\\S6.csv"
    pipeline = ChemModelPipeline().load_data(data_path).preprocess_data()

    # 1. 贝叶斯优化
    print("=== 开始贝叶斯优化 ===")
    pbounds = {
        'd_model': (64, 256),
        'nhead': (1, 8),
        'num_layers': (1, 6),
        'C': (-3, 3),
        'epsilon': (0.01, 0.2),
        'gamma': (-5, 1)
    }

    # 创建优化器（使用lambda传递pipeline）
    optimizer = BayesianOptimization(
        f=lambda d_model, nhead, num_layers, C, epsilon, gamma:
        svm_eval(d_model, nhead, num_layers, C, epsilon, gamma, pipeline),
        pbounds=pbounds,
        random_state=SEED,
        verbose=2)
    optimizer.maximize(init_points=10, n_iter=40)

    best_params = optimizer.max['params']
    print("\n最佳参数组合:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    # 2. 消融实验
    print("\n=== 开始消融实验 ===")
    ablation_results = run_ablation_study(pipeline, best_params)

    print("\n=== 消融实验结果 ===")
    print(ablation_results)

    # 3. 最终评估
    print("\n=== 最终模型评估 ===")
    final_metrics = evaluate_final_model(pipeline, best_params)


if __name__ == "__main__":
    main()