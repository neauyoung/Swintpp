import itertools
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from imblearn.over_sampling import ADASYN, SMOTE
import torch
from skimage.util import random_noise
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import pandas as pd
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ======== 改这里：导入你新的模型 ========
from model_new_sta import swin_tiny_patch4_window7_224 as create_model


# ========================= 可视化函数 =========================
def plot_metrics(epoch, train_loss_all, train_accur_all, i, save_path, test_loss_all=None, test_accur_all=None):
    label_name1 = f'Training Accuracy-{i}-fold'
    label_name2 = f'Training Loss-{i}-fold'

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(epoch), train_accur_all, label=label_name1)
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(range(epoch), train_loss_all, label=label_name2)
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.close()


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Greens,
                          normalize=True, matrix_save_path=None):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if matrix_save_path is not None:
        plt.savefig(matrix_save_path)
    plt.close()


def plot_conf(y_pre, y_val, labels, matrix_save_path=None):
    conf_mat = confusion_matrix(y_true=y_val, y_pred=y_pre)
    print(conf_mat)
    plot_confusion_matrix(conf_mat, normalize=False, target_names=labels,
                          title='Confusion Matrix', matrix_save_path=matrix_save_path)


# ========================= 数据处理函数 =========================
def add_noise_transform(img):
    img_np = np.array(img)
    noisy_img_np = random_noise(img_np, mode='gaussian', seed=int(time.time()), clip=True) * 255
    noisy_img_np = np.uint8(noisy_img_np)
    return Image.fromarray(noisy_img_np)


def data_load(data_dir, img_height, img_width):
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes
    return dataset, class_names


def data_load_add_noise(data_dir, img_height, img_width):
    original_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    noisy_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.Lambda(lambda img: add_noise_transform(img)),
        transforms.ToTensor()
    ])

    original_dataset = datasets.ImageFolder(root=data_dir, transform=original_transform)
    noisy_dataset = datasets.ImageFolder(root=data_dir, transform=noisy_transform)
    combined_dataset = ConcatDataset([original_dataset, noisy_dataset])
    class_names = original_dataset.classes
    return combined_dataset, class_names


def ADASYN_argument(dataset):
    class_counts = defaultdict(int)
    for _, label in dataset:
        class_counts[label] += 1

    class_counts_dict = dict(class_counts)
    X, y = [], []
    for img, label in dataset:
        X.append(img.numpy().flatten())
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    sampling_strategy = {
        1: max(int(class_counts_dict[0] * 0.9), class_counts_dict[1]),
        2: max(int(class_counts_dict[0] * 0.8), class_counts_dict[2]),
    }
    if 3 in class_counts_dict:
        sampling_strategy[3] = max(int(class_counts_dict[0] * 0.8), class_counts_dict[3])

    adasyn = ADASYN(sampling_strategy=sampling_strategy, n_neighbors=5, random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    # 当前 train 使用 3 通道 224x224
    X_resampled_images = X_resampled.reshape(-1, 3, 224, 224)
    X_resampled_tensors = torch.tensor(X_resampled_images, dtype=torch.float32)
    y_resampled_tensors = torch.tensor(y_resampled, dtype=torch.long)

    class CustomDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return self.images[index], self.labels[index]

    return CustomDataset(X_resampled_tensors, y_resampled_tensors)


def SMOTE_argument(dataset):
    class_counts = defaultdict(int)
    for _, label in dataset:
        class_counts[label] += 1

    class_counts_dict = dict(class_counts)
    X, y = [], []
    for img, label in dataset:
        X.append(img.numpy().flatten())
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    sampling_strategy = {
        1: max(int(class_counts_dict[0] * 0.9), class_counts_dict[1]),
        2: max(int(class_counts_dict[0] * 0.8), class_counts_dict[2]),
    }
    if 3 in class_counts_dict:
        sampling_strategy[3] = max(int(class_counts_dict[0] * 0.8), class_counts_dict[3])

    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_resampled_images = X_resampled.reshape(-1, 3, 224, 224)
    X_resampled_tensors = torch.tensor(X_resampled_images, dtype=torch.float32)
    y_resampled_tensors = torch.tensor(y_resampled, dtype=torch.long)

    class CustomDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return self.images[index], self.labels[index]

    return CustomDataset(X_resampled_tensors, y_resampled_tensors)


def data_rebuild(dataset):
    X, y = [], []
    for img, label in dataset:
        X.append(img.numpy().flatten())
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X_resampled_images = X.reshape(-1, 3, 224, 224)
    X_resampled_tensors = torch.tensor(X_resampled_images, dtype=torch.float32)
    y_resampled_tensors = torch.tensor(y, dtype=torch.long)

    class CustomDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return self.images[index], self.labels[index]

    return CustomDataset(X_resampled_tensors, y_resampled_tensors)


def calculate_metrics(conf_matrix):
    Nn = conf_matrix[0, 0]
    Rr = conf_matrix[1, 1]
    Tt = conf_matrix[2, 2]

    N = np.sum(conf_matrix[0, :])
    R = np.sum(conf_matrix[1, :])
    T = np.sum(conf_matrix[2, :])

    specificity = Nn / N if N > 0 else 0.0
    sensitivity = (Rr + Tt) / (R + T) if (R + T) > 0 else 0.0
    return specificity, sensitivity


# ========================= 训练 / 测试函数 =========================
loss_1 = nn.CrossEntropyLoss()


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    train_loss = 0.0
    train_num = 0
    train_accuracy = 0

    train_bar = tqdm(dataloader, desc="Training")
    for img, target in train_bar:
        img, target = img.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(img)
        loss = loss_1(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * img.size(0)
        pred = torch.argmax(outputs, dim=1)
        train_accuracy += torch.sum(pred == target).item()
        train_num += img.size(0)

    return train_loss / train_num, train_accuracy / train_num


def test(model, testdata, device):
    test_loss = 0.0
    total_accuracy = 0
    y_true = []
    y_pred = []
    test_num = 0

    model.eval()
    with torch.no_grad():
        test_bar = tqdm(testdata, desc="Testing")
        for img, target in test_bar:
            img, target = img.to(device), target.to(device)

            outputs = model(img)
            loss = loss_1(outputs, target)
            test_loss += loss.item() * img.size(0)

            predicted = torch.argmax(outputs, dim=1)
            total_accuracy += torch.sum(predicted == target).item()

            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            test_num += img.size(0)

    test_loss = test_loss / test_num
    avg_accuracy = total_accuracy / test_num
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return test_loss, avg_accuracy, precision, recall, f1, conf_matrix, y_pred, y_true


# ========================= 微调辅助函数 =========================
def load_pretrained_weights(model, weight_path, num_classes=3, device='cpu'):
    """
    加载预训练权重：
    - 自动兼容 state_dict / checkpoint / 整模型中的 state_dict
    - 删除 head 权重，避免类别数不一致
    """
    if weight_path is None or weight_path == "" or (not os.path.exists(weight_path)):
        print("未提供有效预训练权重路径，将从头训练。")
        return

    checkpoint_data = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint_data, dict):
        if 'model' in checkpoint_data:
            weights_dict = checkpoint_data['model']
        elif 'state_dict' in checkpoint_data:
            weights_dict = checkpoint_data['state_dict']
        else:
            weights_dict = checkpoint_data
    else:
        # 若误存成整模型对象
        weights_dict = checkpoint_data.state_dict()

    # 处理 DataParallel 前缀
    clean_weights = {}
    for k, v in weights_dict.items():
        if k.startswith("module."):
            clean_weights[k[7:]] = v
        else:
            clean_weights[k] = v
    weights_dict = clean_weights

    # 删除分类头
    keys_to_delete = [k for k in weights_dict.keys() if k.startswith("head.")]
    for k in keys_to_delete:
        del weights_dict[k]

    incompatible = model.load_state_dict(weights_dict, strict=False)
    print("预训练权重加载完成")
    print("missing_keys:", incompatible.missing_keys)
    print("unexpected_keys:", incompatible.unexpected_keys)


def set_trainable_params_phase1(model):
    """
    第一阶段：冻结主干，只训练新增/后端模块
    """
    for _, param in model.named_parameters():
        param.requires_grad = False

    train_keywords = [
        "st_attention",
        "local_conv",
        "global_gate",
        "fuse_conv",
        "norm",
        "head"
    ]

    for name, param in model.named_parameters():
        if any(key in name for key in train_keywords):
            param.requires_grad = True


def set_trainable_params_phase2(model):
    """
    第二阶段：解冻全部参数
    """
    for _, param in model.named_parameters():
        param.requires_grad = True


def build_optimizer(model, lr, weight_decay=5e-2):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def print_trainable_params(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(f"总参数量: {total:,}")
    print(f"可训练参数量: {trainable:,}")
    print(f"可训练比例: {100.0 * trainable / total:.2f}%")


# ========================= 主程序 =========================
if __name__ == '__main__':
    torch.manual_seed(42)

    # ===== 路径与配置：尽量保持你原来的风格 =====
    data_dir = r"/root/autodl-tmp/aug_feature_gamma"
    save_dir = "./model_weight"
    os.makedirs(save_dir, exist_ok=True)

    # 如果你有 Swin Tiny 预训练权重，把路径填这里；没有就设为 None
    pretrained_path = r"/root/autodl-tmp/Swin-Transformer-master/swin_tiny_patch4_window7_224.pth"
    # 例如：
    # pretrained_path = r"./pretrained/swin_tiny_patch4_window7_224.pth"

    num_classes = 3
    batch_size = 32

    # 两阶段微调
    phase1_epochs = 5     # 先训新增模块
    phase2_epochs = 15    # 再全局微调
    total_epochs = phase1_epochs + phase2_epochs

    phase1_lr = 1e-4
    phase2_lr = 5e-5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    matrix_list = []
    test_results = []
    rpts = []

    for i in range(1, 6):
        print("=" * 80)
        print("Fold:", i)

        # ===== 创建模型 =====
        mymodel = create_model(num_classes=num_classes, in_chans=3)
        load_pretrained_weights(mymodel, pretrained_path, num_classes=num_classes, device=device)
        mymodel.to(device)

        # ===== 五折数据划分（保留你原来的逻辑）=====
        train_all = None
        result_pred_true = {}

        for j in range(1, 6):
            if j == i:
                test_dir = os.path.join(data_dir, str(j))
                test_ds_raw, class_names = data_load(test_dir, 224, 224)

                file_name = []
                for root, dirs, files in os.walk(test_dir):
                    if len(files) != 0:
                        file_name += files
                print('file_name', len(file_name))
                result_pred_true['文件名'] = file_name
            else:
                train_dir = os.path.join(data_dir, str(j))
                train_ds, class_names = data_load(train_dir, 224, 224)
                if train_all is None:
                    train_all = train_ds
                else:
                    train_all = ConcatDataset([train_all, train_ds])

        # 可选采样增强：默认不启用
        # train_all = ADASYN_argument(train_all)
        # train_all = SMOTE_argument(train_all)
        # train_all = data_rebuild(train_all)

        train_loader = DataLoader(train_all, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_ds_raw, batch_size=batch_size, shuffle=False, num_workers=8)

        train_loss_all = []
        train_accur_all = []

        # ===== Phase 1：冻结主干，训练新增模块 =====
        print("\n[Phase 1] 冻结 backbone + 训练新增模块")
        set_trainable_params_phase1(mymodel)
        print_trainable_params(mymodel)
        optimizer = build_optimizer(mymodel, lr=phase1_lr)

        for epoch in range(phase1_epochs):
            train_loss, train_acc = train_one_epoch(mymodel, train_loader, optimizer, device)
            print(f"Phase1 Epoch [{epoch + 1}/{phase1_epochs}] "
                  f"train-Loss: {train_loss:.6f}, train-accuracy: {train_acc:.6f}")
            train_loss_all.append(train_loss)
            train_accur_all.append(train_acc)

        # ===== Phase 2：解冻全模型微调 =====
        print("\n[Phase 2] 解冻全部参数，进行全局微调")
        set_trainable_params_phase2(mymodel)
        print_trainable_params(mymodel)
        optimizer = build_optimizer(mymodel, lr=phase2_lr)

        for epoch in range(phase2_epochs):
            train_loss, train_acc = train_one_epoch(mymodel, train_loader, optimizer, device)
            print(f"Phase2 Epoch [{epoch + 1}/{phase2_epochs}] "
                  f"train-Loss: {train_loss:.6f}, train-accuracy: {train_acc:.6f}")
            train_loss_all.append(train_loss)
            train_accur_all.append(train_acc)

        # ===== 保存当前 fold 的 state_dict =====
        model_name = f'swin_sta_fold{i}.pth'
        save_path_model = os.path.join(save_dir, model_name)
        torch.save({
            "model": mymodel.state_dict(),
            "num_classes": num_classes,
            "fold": i
        }, save_path_model)
        print("模型参数已保存:", save_path_model)

        # ===== 训练曲线 =====
        save_path_curve = os.path.join(save_dir, f'result-swin-sta-fold{i}.png')
        plot_metrics(
            epoch=total_epochs,
            train_loss_all=train_loss_all,
            train_accur_all=train_accur_all,
            i=i,
            save_path=save_path_curve
        )

        # ===== 测试：重新建模再载入，更稳妥 =====
        test_model_instance = create_model(num_classes=num_classes, in_chans=3).to(device)
        ckpt = torch.load(save_path_model, map_location=device)
        test_model_instance.load_state_dict(ckpt["model"], strict=True)

        test_loss, avg_accuracy, precision, recall, f1, conf_matrix, y_pred, y_true = test(
            test_model_instance, test_loader, device
        )
        specificity, sensitivity = calculate_metrics(conf_matrix)

        matrix_list.append(conf_matrix)
        test_results.append({
            'fold': i,
            'avg_accuracy': avg_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'se': sensitivity,
            'sp': specificity,
        })

        label = ['normal', 'luoyin', 'guosu']
        matrix_save_path = os.path.join(save_dir, f'matrix-swin-sta-fold{i}.png')

        result_pred_true['真实标签'] = y_true
        result_pred_true['预测标签'] = y_pred
        df_r = pd.DataFrame(result_pred_true)
        rpts.append(df_r)

        plot_conf(y_pred, y_true, label, matrix_save_path)

        print(f'Test Loss: {test_loss}')
        print(f'Average Accuracy: {avg_accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'Confusion Matrix:\n{conf_matrix}')
        print(f'Se: {sensitivity}')
        print(f'Sp: {specificity}')

    # ===== 五折平均结果 =====
    average_results = {
        'avg_accuracy': sum(result['avg_accuracy'] for result in test_results) / len(test_results),
        'precision': sum(result['precision'] for result in test_results) / len(test_results),
        'recall': sum(result['recall'] for result in test_results) / len(test_results),
        'f1': sum(result['f1'] for result in test_results) / len(test_results),
        'se': sum(result['se'] for result in test_results) / len(test_results),
        'sp': sum(result['sp'] for result in test_results) / len(test_results),
    }

    output_path = os.path.join(save_dir, 'true_pred_compare.xlsx')
    sheet_names = [f"Fold_{i+1}" for i in range(len(rpts))]
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for df, name in zip(rpts, sheet_names):
            df.to_excel(writer, sheet_name=name, index=False)

    print("5折平均结果：")
    print(f'平均准确率: {round(average_results["avg_accuracy"] * 100, 2)}%')
    print(f'平均精确率: {round(average_results["precision"] * 100, 2)}%')
    print(f'平均召回率: {round(average_results["recall"] * 100, 2)}%')
    print(f'平均F1分数: {round(average_results["f1"] * 100, 2)}%')
    print(f'特异性SP: {round(average_results["sp"] * 100, 2)}%')
    print(f'敏感性Se: {round(average_results["se"] * 100, 2)}%')

    avg_cm = np.mean(matrix_list, axis=0)
    np.set_printoptions(suppress=True)
    print(avg_cm)

    specificity, sensitivity = calculate_metrics(avg_cm)
    score = (specificity + sensitivity) / 2
    print(f'特异性SP: {round(specificity * 100, 2)}%')
    print(f'敏感性Se: {round(sensitivity * 100, 2)}%')
    print(f'score: {round(score * 100, 2)}%')

    test_results.append(average_results)
    df_results = pd.DataFrame(test_results)
    excel_path = os.path.join(save_dir, 'cross_validation_results.xlsx')
    df_summary = pd.DataFrame([{
        "specificity": round(specificity, 4),
        "sensitivity": round(sensitivity, 4),
        "score": round(score, 4)
    }])

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="CrossValidation", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    row_sums = avg_cm.sum(axis=1, keepdims=True)
    percentage_matrix = avg_cm / row_sums
    formatted_matrix = np.round(percentage_matrix, 3)
    print(formatted_matrix)
