from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 示例 SMILES 列表
smiles_list = [
    'CC', 'CCC', 'C', 'CCCCCCC',
    'CCCCC', 'C=C', 'C=CC=C',
    'C=CC', 'CO', 'CC(O)C',
    'CCN(CC)CC', 'C=CC=O',
    'CC(=O)C', 'C1=CC=CC=C1',
    'C1CO1', 'CSC', '[H][H]',
    'S'
]


def tanimoto_matrix(smiles_list, radius=2, nBits=1024):
    """计算 Tanimoto 相似性矩阵"""
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits) for mol in mols]

    n = len(fps)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            matrix[i][j] = sim
            matrix[j][i] = sim  # 对称矩阵

    return matrix


# 计算相似性矩阵
sim_matrix = tanimoto_matrix(smiles_list)


def plot_similarity(matrix, smiles_list, figsize=(12, 10)):
    """绘制相似性热力图"""
    # 截取 SMILES 字符串以避免过长
    labels = [s[:10] + '...' if len(s) > 10 else s for s in smiles_list]

    # 创建 DataFrame
    df = pd.DataFrame(matrix, index=labels, columns=labels)

    # 绘制热力图
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlOrRd",
                cbar_kws={'label': 'Tanimoto Similarity'},
                annot_kws={"size": 8})
    plt.title("Tanimoto Similarity Matrix (Morgan Fingerprints)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图像
    plt.savefig('tanimoto_matrix.png', dpi=300)



# 绘制热力图
plot_similarity(sim_matrix, smiles_list)

# 计算统计量
mean_sim = np.mean(sim_matrix)
min_sim = np.min(sim_matrix)
max_sim = np.max(sim_matrix)

print(f"平均相似度: {mean_sim:.3f}")
print(f"最小相似度: {min_sim:.3f}")
print(f"最大相似度: {max_sim:.3f}")

# 识别最相似/最不相似对
# 排除对角线元素（自身相似度为 1）
off_diag_matrix = sim_matrix - np.eye(len(sim_matrix))

max_idx = np.unravel_index(np.argmax(off_diag_matrix), off_diag_matrix.shape)
min_idx = np.unravel_index(np.argmin(sim_matrix + np.eye(len(sim_matrix))), sim_matrix.shape)
