import torch
import matplotlib.pyplot as plt


def random_loader_sampling(test_loader, num_samples=10):
    images, labels = [], []
    # random seed
    torch.manual_seed(0)

    for i, (image, label) in enumerate(test_loader):
        if i == num_samples:
            break
        images.append(image)
        labels.append(label)

    # list to tensor
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)

    return images, labels


def evaluate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images_batch, labels_batch in data_loader:
            outputs = model(images_batch)
            _, predicted_labels = torch.max(outputs, 1)
            correct += (predicted_labels == labels_batch).sum().item()
            total += labels_batch.size(0)
    accuracy = correct / total
    return accuracy


def save_grad_cam_heatmap(heatmap, filename):
    num_samples = min(heatmap.size(0), 36)
    heatmap = heatmap[:num_samples]
    fig, axs = plt.subplots(6, 6, figsize=(15, 15))

    for i, ax in enumerate(axs.flat):
        if i >= num_samples:
            break
        hm = heatmap[i].squeeze().detach().cpu().numpy()
        ax.imshow(hm, cmap='viridis')
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(filename)
    plt.close()


def save_adversarial_example(adv_example, classidx, predidx, filename):
    num_samples = min(adv_example.size(0), 36)
    adv_example = adv_example[:num_samples]
    fig, axs = plt.subplots(6, 6, figsize=(15, 15))

    for i, ax in enumerate(axs.flat):
        if i >= num_samples:
            break
        adv_ex = adv_example[i].squeeze().cpu().numpy()
        ax.imshow(adv_ex, cmap='gray')
        ax.axis('off')

        # 添加小标题，显示真实标签和预测标签
        if classidx is not None and predidx is not None:
            true_label = classidx[i]  # 真实标签
            pred_label = predidx[i]  # 预测标签
            
            # 设置标题并设置字体颜色
            title_color = 'green' if true_label == pred_label else 'red'  # 正确预测显示绿色，错误显示红色
            ax.set_title(f"{true_label}\n {pred_label}", fontsize=8, color=title_color)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(filename)
    plt.close()
