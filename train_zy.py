import itertools
import os
import time
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from imblearn.over_sampling import ADASYN, SMOTE
from matplotlib import rcParams
from skimage.util import random_noise
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from model_zy import swin_tiny_patch4_window7_224 as create_model

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

batch_size = 64
num_workers = 8
num_classes = 3
img_height = 224
img_width = 224
seed = 42


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_metrics(epoch, train_loss_all, train_accur_all, val_loss_all, val_accur_all, i, save_path):
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(epoch), train_accur_all, label=f'Training Accuracy-{i}-fold')
    plt.plot(range(epoch), val_accur_all, label=f'Validation Accuracy-{i}-fold')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(range(epoch), train_loss_all, label=f'Training Loss-{i}-fold')
    plt.plot(range(epoch), val_loss_all, label=f'Validation Loss-{i}-fold')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.close()


def plot_confusion_matrix(cm, target_names, title='Confusion matrix',
                          cmap=plt.cm.Greens, normalize=True, matrix_save_path=None):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

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

    if matrix_save_path:
        plt.savefig(matrix_save_path)
    plt.close()


def plot_conf(y_pre, y_val, labels, matrix_save_path=None):
    conf_mat = confusion_matrix(y_true=y_val, y_pred=y_pre)
    print(conf_mat)
    plot_confusion_matrix(
        conf_mat,
        normalize=False,
        target_names=labels,
        title='Confusion Matrix',
        matrix_save_path=matrix_save_path
    )


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
    _ = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True)
    return dataset, class_names


def calculate_metrics(conf_matrix):
    Nn = conf_matrix[0, 0]
    Rr = conf_matrix[1, 1]
    Tt = conf_matrix[2, 2]

    N = np.sum(conf_matrix[0, :])
    R = np.sum(conf_matrix[1, :])
    T = np.sum(conf_matrix[2, :])

    specificity = Nn / N if N != 0 else 0
    sensitivity = (Rr + Tt) / (R + T) if (R + T) != 0 else 0
    return specificity, sensitivity


loss_1 = nn.CrossEntropyLoss()


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    train_loss = 0.0
    train_num = 0
    train_accuracy = 0.0

    train_bar = tqdm(dataloader, desc="Training")

    for img, target in train_bar:
        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

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


def evaluate_one_epoch(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    val_num = 0
    val_accuracy = 0.0

    with torch.no_grad():
        val_bar = tqdm(dataloader, desc="Validation")

        for img, target in val_bar:
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            outputs = model(img)
            loss = loss_1(outputs, target)

            val_loss += loss.item() * img.size(0)
            pred = torch.argmax(outputs, dim=1)
            val_accuracy += torch.sum(pred == target).item()
            val_num += img.size(0)

    return val_loss / val_num, val_accuracy / val_num


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
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

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

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return test_loss, avg_accuracy, precision, recall, f1, conf_matrix, y_pred, y_true


def load_pretrained_weights(model, weight_path, device='cpu'):
    if weight_path is None or weight_path == "" or not os.path.exists(weight_path):
        print("未提供有效预训练权重路径，将从头训练。")
        return

    checkpoint_data = torch.load(weight_path, map_location=device, weights_only=False)

    if isinstance(checkpoint_data, dict):
        if 'model' in checkpoint_data:
            weights_dict = checkpoint_data['model']
        elif 'state_dict' in checkpoint_data:
            weights_dict = checkpoint_data['state_dict']
        else:
            weights_dict = checkpoint_data
    else:
        weights_dict = checkpoint_data.state_dict()

    clean_weights = {}
    for k, v in weights_dict.items():
        k = k.replace("module.", "")
        if k.startswith("head."):
            continue
        clean_weights[k] = v

    model_dict = model.state_dict()
    filtered_dict = {}
    skipped = []

    for k, v in clean_weights.items():
        if k in model_dict and hasattr(v, "shape") and v.shape == model_dict[k].shape:
            filtered_dict[k] = v
        else:
            skipped.append(k)

    incompatible = model.load_state_dict(filtered_dict, strict=False)

    print("预训练权重加载完成")
    print("加载参数数目:", len(filtered_dict))
    print("跳过参数数目:", len(skipped))
    print("missing_keys:", len(incompatible.missing_keys))
    print("unexpected_keys:", len(incompatible.unexpected_keys))


def set_trainable_params_phase1(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if (
            name.startswith("head")
            or "high_proj" in name
            or "local_conv" in name
            or "global_gate" in name
        ):
            param.requires_grad = True


def set_trainable_params_phase2(model):
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


if __name__ == '__main__':
    set_seed(seed)

    data_dir = r"/root/autodl-tmp/aug_feature_gamma"
    save_dir = "./model_weight_exp3"
    os.makedirs(save_dir, exist_ok=True)

    pretrained_path = r"/root/autodl-tmp/Swin-Transformer-master/swin_tiny_patch4_window7_224.pth"

    phase1_epochs = 5
    phase2_epochs = 15
    total_epochs = phase1_epochs + phase2_epochs
    phase1_lr = 1e-4
    phase2_lr = 5e-5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(f"预训练权重不存在: {pretrained_path}")

    matrix_list = []
    test_results = []
    rpts = []

    for i in range(1, 6):
        print("=" * 80)
        print("Fold:", i)

        mymodel = create_model(num_classes=num_classes, use_foldback=True)
        load_pretrained_weights(mymodel, pretrained_path, device=device)
        mymodel.to(device)

        train_all = None
        result_pred_true = {}

        for j in range(1, 6):
            if j == i:
                test_dir = os.path.join(data_dir, str(j))
                test_ds_raw, class_names = data_load(test_dir, img_height, img_width)

                file_name = []
                for root, dirs, files in os.walk(test_dir):
                    if len(files) != 0:
                        file_name += files

                print('file_name', len(file_name))
                result_pred_true['文件名'] = file_name

            else:
                train_dir = os.path.join(data_dir, str(j))
                train_ds, class_names = data_load(train_dir, img_height, img_width)

                if train_all is None:
                    train_all = train_ds
                else:
                    train_all = ConcatDataset([train_all, train_ds])

        train_loader = DataLoader(
            train_all,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False
        )

        test_loader = DataLoader(
            test_ds_raw,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False
        )

        train_loss_all = []
        train_accur_all = []
        val_loss_all = []
        val_accur_all = []

        print("\n[Phase 1] 冻结 backbone + 训练新增模块和分类头")
        set_trainable_params_phase1(mymodel)
        print_trainable_params(mymodel)
        optimizer = build_optimizer(mymodel, lr=phase1_lr)

        for epoch_idx in range(phase1_epochs):
            train_loss, train_acc = train_one_epoch(mymodel, train_loader, optimizer, device)
            val_loss, val_acc = evaluate_one_epoch(mymodel, test_loader, device)

            print(f"[Phase1] Epoch {epoch_idx + 1}/{phase1_epochs} | "
                  f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                  f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

            train_loss_all.append(train_loss)
            train_accur_all.append(train_acc)
            val_loss_all.append(val_loss)
            val_accur_all.append(val_acc)

        print("\n[Phase 2] 解冻全部参数微调")
        set_trainable_params_phase2(mymodel)
        print_trainable_params(mymodel)
        optimizer = build_optimizer(mymodel, lr=phase2_lr)

        for epoch_idx in range(phase2_epochs):
            train_loss, train_acc = train_one_epoch(mymodel, train_loader, optimizer, device)
            val_loss, val_acc = evaluate_one_epoch(mymodel, test_loader, device)

            print(f"[Phase2] Epoch {epoch_idx + 1}/{phase2_epochs} | "
                  f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                  f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

            train_loss_all.append(train_loss)
            train_accur_all.append(train_acc)
            val_loss_all.append(val_loss)
            val_accur_all.append(val_acc)

        model_name = f'exp3_fold{i}.pth'
        model_path = os.path.join(save_dir, model_name)
        torch.save(mymodel.state_dict(), model_path)
        print("模型已保存:", model_path)

        save_path = os.path.join(save_dir, f'result-exp3-fold{i}.png')
        plot_metrics(
            epoch=total_epochs,
            train_loss_all=train_loss_all,
            train_accur_all=train_accur_all,
            val_loss_all=val_loss_all,
            val_accur_all=val_accur_all,
            i=i,
            save_path=save_path
        )

        eval_model = create_model(num_classes=num_classes, use_foldback=True)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        eval_model.load_state_dict(state_dict, strict=True)
        eval_model.to(device)

        test_loss, avg_accuracy, precision, recall, f1, conf_matrix, y_pred, y_true = test(
            eval_model, test_loader, device
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
        matrix_save_path = os.path.join(save_dir, f'matrix-exp3-fold{i}.png')

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
        print(f'se: {sensitivity}')
        print(f'sp: {specificity}')

        del mymodel, eval_model, optimizer, train_loader, test_loader
        torch.cuda.empty_cache()

    average_results = {
        'avg_accuracy': sum(result['avg_accuracy'] for result in test_results) / len(test_results),
        'precision': sum(result['precision'] for result in test_results) / len(test_results),
        'recall': sum(result['recall'] for result in test_results) / len(test_results),
        'f1': sum(result['f1'] for result in test_results) / len(test_results),
        'se': sum(result['se'] for result in test_results) / len(test_results),
        'sp': sum(result['sp'] for result in test_results) / len(test_results),
    }

    output_path = os.path.join(save_dir, 'true_pred_compare.xlsx')
    sheet_names = [f"Fold_{i + 1}" for i in range(len(rpts))]

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
    df_summary = pd.DataFrame([{
        "specificity": round(specificity, 4),
        "sensitivity": round(sensitivity, 4),
        "score": round(score, 4)
    }])

    excel_path = os.path.join(save_dir, 'cross_validation_results.xlsx')

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="CrossValidation", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    row_sums = avg_cm.sum(axis=1, keepdims=True)
    percentage_matrix = avg_cm / row_sums
    formatted_matrix = np.round(percentage_matrix, 3)

    print(formatted_matrix)
