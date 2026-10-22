import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pennylane as qml
import os
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    precision_score, recall_score, cohen_kappa_score,
    confusion_matrix
)
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import csv
from scipy.stats import wilcoxon
# 不再需要单独 from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ==================== 参数设置 ====================
n_epochs = 30
n_layers = 1
n_train = 50
n_test = 30
n_experiments = 10
N_EXPERTS = 3

data = np.load('TibetanMNIST.npz')
images = data['image']          # (17768, 28, 28), float32, 值域 [0, 225]
labels = data['label']          # (17768,), int32, 0‑9


# 先打乱，避免单类训练
np.random.seed(42)
idx = np.random.permutation(len(images))
images = images[idx]
labels = labels[idx]

# 简单切片（无分层抽样）
train_images = images[:n_train] / 255.0
train_labels = labels[:n_train]
test_images = images[n_train:n_train + n_test] / 255.0
test_labels = labels[n_train:n_train + n_test]


# 添加通道维度
train_images = np.array(train_images[..., np.newaxis], dtype=np.float32)
test_images  = np.array(test_images[..., np.newaxis], dtype=np.float32)

# 计算类别权重（仅用于最终阴影分类器和经典/量子模型，以提升小类表现）
# 计算类别权重（包含所有可能缺失的类别）
unique_classes = np.unique(train_labels)
all_classes = np.arange(10)          # 0‑9
class_weights_array = class_weight.compute_class_weight(
    'balanced',
    classes=unique_classes,
    y=train_labels
)
# 构建完整的权重字典（缺失的类别给默认权重 1.0）
class_weights_dict = {}
for cls in all_classes:
    if cls in unique_classes:
        idx = np.where(unique_classes == cls)[0][0]
        class_weights_dict[cls] = class_weights_array[idx]
    else:
        class_weights_dict[cls] = 1.0   # 或设为 0.0，1.0 不会影响梯度

print("class_weights_dict:", class_weights_dict)   # 确认包含 0‑9

# ==================== 基础模型 ====================
def MyModel():
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_c_model():
    c_model = keras.models.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax")
    ])
    c_model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = c_model.fit(train_images, train_labels, validation_data=(test_images, test_labels),
                          batch_size=4, epochs=n_epochs, verbose=0, shuffle=False,
                          class_weight=class_weights_dict)          # 加入类别权重
    train_c_model.last_pred = np.argmax(c_model.predict(test_images), axis=1)
    return history

def train_q_model():
    dev = qml.device("lightning.qubit", wires=9)
    rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 9))

    @qml.qnode(dev, interface='autograd')
    def circuit(phi):
        for j in range(9): qml.RY(np.pi * phi[j], wires=j)
        qml.templates.RandomLayers(rand_params, wires=list(range(9)))
        return [qml.expval(qml.PauliZ(j)) for j in range(9)]

    def quanv(image):
        out = np.zeros((9, 9, 4))
        for j in range(0, 27, 3):
            for k in range(0, 27, 3):
                q_results = circuit(image[j:j + 3, k:k + 3, 0].flatten())
                for c in range(4): out[j // 3, k // 3, c] = q_results[c]
        return out

    q_train = np.array([quanv(img) for img in train_images])
    q_test = np.array([quanv(img) for img in test_images])
    q_model = MyModel()
    history = q_model.fit(q_train, train_labels, validation_data=(q_test, test_labels),
                          batch_size=4, epochs=n_epochs, verbose=0, shuffle=False,
                          class_weight=class_weights_dict)          # 加入类别权重
    train_q_model.last_pred = np.argmax(q_model.predict(q_test), axis=1)
    return history

def train_s_model(experiment_seed):
    np.random.seed(experiment_seed)
    tf.random.set_seed(experiment_seed)

    devsh = qml.device("lightning.qubit", wires=27)          # 改回 27 比特
    topologies = ['none', 'chain', 'ring']
    gate_sets_close = ['RY', 'RX']
    gate_sets_full = ['RY', 'RX', 'RZ']

    def get_entangled_pairs(topo, base_wires):
        w = list(base_wires)
        if topo == 'none': return []
        if topo == 'chain': return [[w[i], w[i + 1]] for i in range(len(w) - 1)]
        if topo == 'ring': return [[w[i], (i + 1) % len(w)] for i in range(len(w))]

    # 恢复三层电路
    def build_shadow_circuit(topo1, topo2, topo3, gate1, gate2, gate3):
        @qml.qnode(devsh, interface='autograd')
        def circuit(params1, params2, params3):
            for l in range(n_layers):
                for i in range(9):
                    getattr(qml, gate1)(params1[l, i], wires=i)
                for p in get_entangled_pairs(topo1, range(9)): qml.CNOT(wires=p)
            for l in range(n_layers):
                for i in range(9):
                    getattr(qml, gate2)(params2[l, i], wires=9 + i)
                for p in get_entangled_pairs(topo2, range(9, 18)): qml.CNOT(wires=p)
            for l in range(n_layers):
                for i in range(9):
                    getattr(qml, gate3)(params3[l, i], wires=18 + i)
                for p in get_entangled_pairs(topo3, range(18, 27)): qml.CNOT(wires=p)
            for j in range(9):
                qml.CNOT(wires=[j, 9 + j])
                qml.CNOT(wires=[9 + j, 18 + j])
            return [qml.expval(qml.PauliZ(j)) for j in range(27)]
        return circuit

    def calculate_dot_product_results(images, means):
        patch_27x27 = images[:, :27, :27, 0]
        patches = (patch_27x27.reshape(images.shape[0], 9, 3, 9, 3)
                   .transpose(0, 1, 3, 2, 4)
                   .reshape(images.shape[0], 9, 9, 9))
        patches = np.ascontiguousarray(patches)
        return patches * means

    # evaluate 恢复为 3 层参数
    def evaluate(topo1, topo2, topo3, gate1, gate2, gate3, p1, p2, p3):
        qnode = build_shadow_circuit(topo1, topo2, topo3, gate1, gate2, gate3)
        output = np.array(qnode(p1, p2, p3), dtype=np.float64)
        matrix = output.reshape(3, 9)
        means = matrix[1, :]
        means = means - np.mean(means)
        means = means / (np.std(means) + 1e-8)

        dot_train = calculate_dot_product_results(train_images, means)
        dot_test = calculate_dot_product_results(test_images, means)

        s_model = MyModel()
        s_history = s_model.fit(np.float32(dot_train), train_labels,
                                validation_data=(np.float32(dot_test), test_labels),
                                batch_size=4, epochs=n_epochs, verbose=0)
        return max(s_history.history["val_accuracy"]), means

    def init_experts(n):
        return [np.random.uniform(0, 2 * np.pi, (n_layers, 9)) for _ in range(n)]

    def perturb_expert(expert, std):
        new_p = expert + np.random.normal(0, std, (n_layers, 9))
        return np.clip(new_p, 0, 2 * np.pi)

    best_accuracy = 0
    best_means = None
    print("[多专家TD-QAS] 阶段一：拓扑粗筛...")          # 恢复日志前缀

    FIXED_GATE = 'RY'
    topo_dict = {}
    for t1 in topologies:
        for t2 in topologies:
            for t3 in topologies:                         # 三重循环
                experts_p1 = init_experts(N_EXPERTS)
                experts_p2 = init_experts(N_EXPERTS)
                experts_p3 = init_experts(N_EXPERTS)

                best_acc = -1
                best_exp_idx = 0

                for e_idx in range(N_EXPERTS):
                    acc, _ = evaluate(t1, t2, t3, FIXED_GATE, FIXED_GATE, FIXED_GATE,
                                      experts_p1[e_idx], experts_p2[e_idx], experts_p3[e_idx])
                    if acc > best_acc:
                        best_acc = acc
                        best_exp_idx = e_idx

                topo_dict[(t1, t2, t3)] = [
                    best_acc,
                    experts_p1[best_exp_idx].copy(),
                    experts_p2[best_exp_idx].copy(),
                    experts_p3[best_exp_idx].copy(),
                    None
                ]

    topo_sorted = sorted(topo_dict.items(), key=lambda item: item[1][0], reverse=True)
    top_3_topos = topo_sorted[:3]

    print("[多专家TD-QAS] 阶段二：分层门搜索 (RY, RX -> RZ)...")
    gate_scores = []

    for (t1, t2, t3), (base_acc, base_p1, base_p2, base_p3, _) in top_3_topos:
        cur_best_acc = base_acc
        cur_best_p1, cur_best_p2, cur_best_p3 = base_p1.copy(), base_p2.copy(), base_p3.copy()

        # 相近门
        for g1 in gate_sets_close:
            for g2 in gate_sets_close:
                for g3 in gate_sets_close:                # 三重门循环
                    expert_accs = []
                    expert_params = []
                    for _ in range(N_EXPERTS):
                        tp1 = perturb_expert(cur_best_p1, 0.8)
                        tp2 = perturb_expert(cur_best_p2, 0.8)
                        tp3 = perturb_expert(cur_best_p3, 0.8)
                        acc, _ = evaluate(t1, t2, t3, g1, g2, g3, tp1, tp2, tp3)
                        expert_accs.append(acc)
                        expert_params.append((tp1, tp2, tp3))

                    best_e_idx = np.argmax(expert_accs)
                    cur_gate_acc = expert_accs[best_e_idx]
                    cur_gate_p1, cur_gate_p2, cur_gate_p3 = [p.copy() for p in expert_params[best_e_idx]]

                    if cur_gate_acc > cur_best_acc:
                        cur_best_acc = cur_gate_acc
                        cur_best_p1, cur_best_p2, cur_best_p3 = cur_gate_p1.copy(), cur_gate_p2.copy(), cur_gate_p3.copy()

                    gate_scores.append(
                        ((t1, t2, t3, g1, g2, g3), cur_gate_acc, cur_gate_p1, cur_gate_p2, cur_gate_p3, None))

        # 扩展门
        for g1 in gate_sets_full:
            for g2 in gate_sets_full:
                for g3 in gate_sets_full:
                    if 'RZ' not in (g1, g2, g3): continue

                    expert_accs = []
                    expert_params = []
                    for _ in range(N_EXPERTS):
                        tp1 = perturb_expert(cur_best_p1, 0.8)
                        tp2 = perturb_expert(cur_best_p2, 0.8)
                        tp3 = perturb_expert(cur_best_p3, 0.8)
                        acc, _ = evaluate(t1, t2, t3, g1, g2, g3, tp1, tp2, tp3)
                        expert_accs.append(acc)
                        expert_params.append((tp1, tp2, tp3))

                    best_e_idx = np.argmax(expert_accs)
                    cur_gate_acc = expert_accs[best_e_idx]
                    cur_gate_p1, cur_gate_p2, cur_gate_p3 = [p.copy() for p in expert_params[best_e_idx]]

                    if cur_gate_acc > cur_best_acc:
                        cur_best_acc = cur_gate_acc
                        cur_best_p1, cur_best_p2, cur_best_p3 = cur_gate_p1.copy(), cur_gate_p2.copy(), cur_gate_p3.copy()

                    gate_scores.append(
                        ((t1, t2, t3, g1, g2, g3), cur_gate_acc, cur_gate_p1, cur_gate_p2, cur_gate_p3, None))

    gate_scores.sort(key=lambda x: x[1], reverse=True)
    final_structure = gate_scores[0]
    print(f"  锁定最强结构: 拓扑{final_structure[0][:3]}, 门{final_structure[0][3:]}")

    print("[多专家TD-QAS] 阶段三：参数微调...")
    t1, t2, t3, g1, g2, g3 = final_structure[0]
    cur_p1, cur_p2, cur_p3 = final_structure[2], final_structure[3], final_structure[4]
    local_best_acc = final_structure[1]

    _, cur_means = evaluate(t1, t2, t3, g1, g2, g3, cur_p1, cur_p2, cur_p3)

    for _ in range(5):
        expert_accs = []
        expert_params = []
        for _ in range(N_EXPERTS * 2):
            tp1 = perturb_expert(cur_p1, 0.2)
            tp2 = perturb_expert(cur_p2, 0.2)
            tp3 = perturb_expert(cur_p3, 0.2)
            acc, means_tmp = evaluate(t1, t2, t3, g1, g2, g3, tp1, tp2, tp3)
            expert_accs.append(acc)
            expert_params.append((tp1, tp2, tp3, means_tmp))

        best_e_idx = np.argmax(expert_accs)
        if expert_accs[best_e_idx] > local_best_acc:
            local_best_acc = expert_accs[best_e_idx]
            cur_p1, cur_p2, cur_p3 = expert_params[best_e_idx][0], expert_params[best_e_idx][1], expert_params[best_e_idx][2]
            cur_means = expert_params[best_e_idx][3]

    if local_best_acc > best_accuracy:
        best_accuracy = local_best_acc
        best_means = cur_means
        print(f"  最终突破: {best_accuracy:.4f}")

    print("阴影模型 means (视为3x3卷积核):")
    np.set_printoptions(precision=8, suppress=True)
    print(np.array(best_means).reshape(3, 3))
    print("-" * 40)

    # 最终分类器：类别权重 + 更多轮次（保持不变）
    final_dot_train = calculate_dot_product_results(train_images, best_means)
    final_dot_test = calculate_dot_product_results(test_images, best_means)
    final_model = MyModel()
    final_model.fit(np.float32(final_dot_train), train_labels,
                    batch_size=4, epochs=n_epochs, verbose=0,
                    class_weight=class_weights_dict)
    train_s_model.last_pred = np.argmax(final_model.predict(np.float32(final_dot_test)), axis=1)

    class DummyHistory:
        def __init__(self, acc): self.history = {"val_accuracy": [acc]}
    return DummyHistory(best_accuracy)

def holm_correction(pvals):
    pvals = np.asarray(pvals, dtype=float)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)
    m = int(len(pvals))
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = float((m - rank) * pvals[idx])
        val = max(val, prev)
        adjusted[idx] = min(val, 1.0)
        prev = float(adjusted[idx])
    return adjusted.tolist()   # 返回 Python float 列表


def cohens_dz(x, y):
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    std = float(np.std(diff, ddof=1))
    mean_diff = float(np.mean(diff))
    if std < 1e-12:
        return 0.0 if abs(mean_diff) < 1e-12 else float('inf')
    return mean_diff / std


def wilcoxon_greater(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.allclose(x, y):
        return None, 1.0
    try:
        stat, p = wilcoxon(x, y, alternative='greater', zero_method='wilcox')
        stat = float(stat)
        p = float(p)
    except ValueError:
        stat, p = None, 1.0
    return stat, p

def main():
    # ---------- 初始化指标列表 ----------
    c_accs, q_accs, s_accs = [], [], []
    c_wf1s, q_wf1s, s_wf1s = [], [], []
    c_bal_accs, q_bal_accs, s_bal_accs = [], [], []
    c_mccs, q_mccs, s_mccs = [], [], []
    c_precisions, q_precisions, s_precisions = [], [], []
    c_recalls,    q_recalls,    s_recalls    = [], [], []
    c_specificities, q_specificities, s_specificities = [], [], []
    c_kappas,     q_kappas,     s_kappas     = [], [], []

    for exp in range(n_experiments):
        np.random.seed(exp)
        tf.random.set_seed(exp)
        print(f"\n========== 实验 {exp + 1}/{n_experiments} ==========")
        # ------------------- 经典模型 -------------------
        c_acc = max(train_c_model().history["val_accuracy"])
        c_pred = train_c_model.last_pred
        c_wf1 = f1_score(test_labels, c_pred, average='weighted', zero_division=0)
        c_bal = balanced_accuracy_score(test_labels, c_pred)
        c_mcc = matthews_corrcoef(test_labels, c_pred)
        # ------------------- 量子模型 -------------------
        q_acc = max(train_q_model().history["val_accuracy"])
        q_pred = train_q_model.last_pred
        q_wf1 = f1_score(test_labels, q_pred, average='weighted', zero_division=0)
        q_bal = balanced_accuracy_score(test_labels, q_pred)
        q_mcc = matthews_corrcoef(test_labels, q_pred)
        # ------------------- 阴影模型 -------------------
        s_acc = max(train_s_model(exp).history["val_accuracy"])
        s_pred = train_s_model.last_pred
        s_wf1 = f1_score(test_labels, s_pred, average='weighted', zero_division=0)
        s_bal = balanced_accuracy_score(test_labels, s_pred)
        s_mcc = matthews_corrcoef(test_labels, s_pred)
        # ===== 新增指标计算（每个模型） =====
        # 计算所有模型的 Precision, Recall, Kappa
        c_precision = precision_score(test_labels, c_pred, average='macro', zero_division=0)
        c_recall    = recall_score(test_labels, c_pred, average='macro', zero_division=0)
        c_kappa     = cohen_kappa_score(test_labels, c_pred)
        q_precision = precision_score(test_labels, q_pred, average='macro', zero_division=0)
        q_recall    = recall_score(test_labels, q_pred, average='macro', zero_division=0)
        q_kappa     = cohen_kappa_score(test_labels, q_pred)
        s_precision = precision_score(test_labels, s_pred, average='macro', zero_division=0)
        s_recall    = recall_score(test_labels, s_pred, average='macro', zero_division=0)
        s_kappa     = cohen_kappa_score(test_labels, s_pred)
        # 自定义宏平均 Specificity
        def macro_specificity(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            specificities = []
            for i in range(cm.shape[0]):
                tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
                fp = cm[:, i].sum() - cm[i, i]
                if tn + fp == 0:
                    spec = 0.0
                else:
                    spec = tn / (tn + fp)
                specificities.append(spec)
            return np.mean(specificities)
        c_specificity = macro_specificity(test_labels, c_pred)
        q_specificity = macro_specificity(test_labels, q_pred)
        s_specificity = macro_specificity(test_labels, s_pred)
        # ---------- 保存所有指标 ----------
        c_accs.append(c_acc); c_wf1s.append(c_wf1); c_bal_accs.append(c_bal); c_mccs.append(c_mcc)
        q_accs.append(q_acc); q_wf1s.append(q_wf1); q_bal_accs.append(q_bal); q_mccs.append(q_mcc)
        s_accs.append(s_acc); s_wf1s.append(s_wf1); s_bal_accs.append(s_bal); s_mccs.append(s_mcc)

        c_precisions.append(c_precision); c_recalls.append(c_recall)
        c_specificities.append(c_specificity); c_kappas.append(c_kappa)
        q_precisions.append(q_precision); q_recalls.append(q_recall)
        q_specificities.append(q_specificity); q_kappas.append(q_kappa)
        s_precisions.append(s_precision); s_recalls.append(s_recall)
        s_specificities.append(s_specificity); s_kappas.append(s_kappa)

        # ---------- 扩展后的实时输出 ----------
        print(f"[实验 {exp+1}/{n_experiments}] "
              f"Acc: C={c_acc:.3f} Q={q_acc:.3f} S={s_acc:.3f} | "
              f"W-F1: {c_wf1:.3f} {q_wf1:.3f} {s_wf1:.3f} | "
              f"Bal: {c_bal:.3f} {q_bal:.3f} {s_bal:.3f} | "
              f"MCC: {c_mcc:.3f} {q_mcc:.3f} {s_mcc:.3f} | "
              f"Prec: {c_precision:.3f} {q_precision:.3f} {s_precision:.3f} | "
              f"Rec: {c_recall:.3f} {q_recall:.3f} {s_recall:.3f} | "
              f"Spec: {c_specificity:.3f} {q_specificity:.3f} {s_specificity:.3f} | "
              f"Kap: {c_kappa:.3f} {q_kappa:.3f} {s_kappa:.3f}")

    # ---------- 汇总统计 ----------
    dataset_name = "Tibetan-MNIST"
    model_names = ["Classic CNN", "Quantum Conv", "Shadow"]

    # 原有指标
    acc_means = [np.mean(c_accs), np.mean(q_accs), np.mean(s_accs)]
    acc_stds  = [np.std(c_accs),  np.std(q_accs),  np.std(s_accs)]
    wf1_means = [np.mean(c_wf1s), np.mean(q_wf1s), np.mean(s_wf1s)]
    wf1_stds  = [np.std(c_wf1s),  np.std(q_wf1s),  np.std(s_wf1s)]
    bal_means = [np.mean(c_bal_accs), np.mean(q_bal_accs), np.mean(s_bal_accs)]
    bal_stds  = [np.std(c_bal_accs),  np.std(q_bal_accs),  np.std(s_bal_accs)]
    mcc_means = [np.mean(c_mccs), np.mean(q_mccs), np.mean(s_mccs)]
    mcc_stds  = [np.std(c_mccs),  np.std(q_mccs),  np.std(s_mccs)]

    # 新增指标
    prec_means = [np.mean(c_precisions), np.mean(q_precisions), np.mean(s_precisions)]
    rec_means  = [np.mean(c_recalls),    np.mean(q_recalls),    np.mean(s_recalls)]
    spec_means = [np.mean(c_specificities), np.mean(q_specificities), np.mean(s_specificities)]
    kap_means  = [np.mean(c_kappas),     np.mean(q_kappas),     np.mean(s_kappas)]
    rec_stds = [np.std(c_recalls), np.std(q_recalls), np.std(s_recalls)]
    kap_stds = [np.std(c_kappas), np.std(q_kappas), np.std(s_kappas)]

    # ---------- 写入文本结果文件 ----------
    with open(f"{dataset_name}_results.txt", "w", encoding="utf-8") as f:
        f.write(f"{dataset_name}\n")
        # 扩展表头
        header = (f"{'Model':<25} {'Accuracy':<10} {'W-F1':<10} {'BalAcc':<10} "
                  f"{'MCC':<10} {'Precision':<10} {'Recall':<10} "
                  f"{'Specificity':<12} {'Kappa':<10} {'Acc_Var':<10}\n")
        f.write(header)
        for i in range(3):
            f.write(f"{model_names[i]:<25} {acc_means[i]:<10.4f} {wf1_means[i]:<10.4f} "
                    f"{bal_means[i]:<10.4f} {mcc_means[i]:<10.4f} "
                    f"{prec_means[i]:<10.4f} {rec_means[i]:<10.4f} "
                    f"{spec_means[i]:<12.4f} {kap_means[i]:<10.4f} "
                    f"{acc_stds[i]**2:<10.6f}\n")

    # ---------- 控制台最终输出 ----------
    print(f"\n========== 最终结果 ==========")
    for i, name in enumerate(model_names):
        print(f"{name}: Acc={acc_means[i]:.4f}±{acc_stds[i]:.4f}, "
              f"W-F1={wf1_means[i]:.4f}, BalAcc={bal_means[i]:.4f}, "
              f"MCC={mcc_means[i]:.4f}, Precision={prec_means[i]:.4f}, "
              f"Recall={rec_means[i]:.4f}, Specificity={spec_means[i]:.4f}, "
              f"Kappa={kap_means[i]:.4f}")

    # ---------- 绘制柱状图：Accuracy, Recall, Kappa, MCC ----------
    metrics = ['Accuracy', 'Recall', 'Kappa', 'MCC']
    x = np.arange(len(metrics))
    width = 0.225
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    means_data = [acc_means, rec_means, kap_means, mcc_means]
    stds_data = [acc_stds, rec_stds, kap_stds, mcc_stds]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (model, color) in enumerate(zip(model_names, colors)):
        heights = [means_data[j][i] for j in range(len(metrics))]
        errors = [stds_data[j][i] for j in range(len(metrics))]
        bars = ax.bar(x + i * width, heights, width, yerr=errors, capsize=5,
                      label=model, color=color, edgecolor='black', linewidth=0.5)
        for bar, h, err in zip(bars, heights, errors):
            ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.01,
                    f'{h:.3f}±{err:.3f}', ha='center', va='bottom', fontsize=6)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_title(f'Model Comparison on {dataset_name} ')
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_comparison.png", dpi=150)
    print(f"图表已保存为 {dataset_name}_comparison.png")

    # ==================== 统计显著性后处理（不改变任何已有指标） ====================
    metric_dict = {
        'Accuracy': {'Shadow': s_accs, 'CNN': c_accs, 'QCNN': q_accs},
        'W-F1': {'Shadow': s_wf1s, 'CNN': c_wf1s, 'QCNN': q_wf1s},
        'BalAcc': {'Shadow': s_bal_accs, 'CNN': c_bal_accs, 'QCNN': q_bal_accs},
        'MCC': {'Shadow': s_mccs, 'CNN': c_mccs, 'QCNN': q_mccs},
        'Precision': {'Shadow': s_precisions, 'CNN': c_precisions, 'QCNN': q_precisions},
        'Recall': {'Shadow': s_recalls, 'CNN': c_recalls, 'QCNN': q_recalls},
        'Specificity': {'Shadow': s_specificities, 'CNN': c_specificities, 'QCNN': q_specificities},
        'Kappa': {'Shadow': s_kappas, 'CNN': c_kappas, 'QCNN': q_kappas},
    }

    rows = []
    for metric, models in metric_dict.items():
        shadow = models['Shadow']
        for baseline_name in ['CNN', 'QCNN']:
            baseline = models[baseline_name]
            stat, p = wilcoxon_greater(shadow, baseline)
            rows.append({
                'Metric': str(metric),
                'Comparison': f'Shadow vs {baseline_name}',
                'Shadow_mean': float(np.mean(shadow)),
                'Baseline_mean': float(np.mean(baseline)),
                'Mean_diff': float(np.mean(shadow) - np.mean(baseline)),
                'Cohen_dz': float(cohens_dz(shadow, baseline)),
                'p_raw': float(p),
            })

    p_raw = [r['p_raw'] for r in rows]
    p_holm = holm_correction(p_raw)
    for r, ph in zip(rows, p_holm):
        ph = float(ph)
        r['p_holm'] = ph
        if ph < 0.001:
            r['Sig'] = '***'
        elif ph < 0.01:
            r['Sig'] = '**'
        elif ph < 0.05:
            r['Sig'] = '*'
        else:
            r['Sig'] = 'n.s.'

    # 保存统计检验表 CSV
    with open(f"{dataset_name}_statistical_tests.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Metric', 'Comparison', 'Shadow_mean', 'Baseline_mean',
            'Mean_diff', 'Cohen_dz', 'p_raw', 'p_holm', 'Sig'
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n统计检验表已保存为 {dataset_name}_statistical_tests.csv")

    # 配对箱线图（只画 Accuracy、Recall、Kappa、MCC）
    core_metrics = ['Accuracy', 'Recall', 'Kappa', 'MCC']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric in zip(axes, core_metrics):
        models = metric_dict[metric]
        data = [models['CNN'], models['QCNN'], models['Shadow']]
        ax.boxplot(data, labels=['CNN', 'QCNN', 'Shadow'])
        for i in range(len(models['CNN'])):
            ax.plot([1, 2, 3],
                    [models['CNN'][i], models['QCNN'][i], models['Shadow'][i]],
                    color='gray', alpha=0.35, linewidth=0.8)
        ax.set_title(metric)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_paired_boxplot.png", dpi=150)
    plt.close()
    print(f"配对箱线图已保存为 {dataset_name}_paired_boxplot.png")

if __name__ == "__main__":
    main()