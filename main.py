import os
import numpy as np
import jax.numpy as jnp

# optax for optimizer
import optax

# The Flax NNX API.
from flax import nnx

# grain to load data
from loaderx import dataloader

# orbax to save model
import orbax.checkpoint as ocp

# model
from net import RubbishClassifier

# load data
from feeder import Dataset

from tensorboardX import SummaryWriter


def load_class_weights(weight_path="./data/class_weights.npy"):
    """加载类别权重"""
    try:
        class_weights = np.load(weight_path, allow_pickle=True).item()
        # 转换为JAX数组，按类别ID顺序排列
        weights_array = jnp.array([class_weights[i] for i in range(len(class_weights))])
        print(f"已加载类别权重: {class_weights}")
        return weights_array
    except FileNotFoundError:
        print("未找到类别权重文件，使用均匀权重")
        return None


def weighted_loss_fn(model: RubbishClassifier, batch, class_weights=None):
    """带权重的损失函数"""
    logits = model(batch["data"])

    if class_weights is not None:
        # 获取每个样本的权重
        sample_weights = class_weights[batch["label"]]
        # 计算加权交叉熵损失
        losses = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch["label"])
        weighted_losses = losses * sample_weights
        loss = weighted_losses.mean()
    else:
        # 标准交叉熵损失
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch["label"]).mean()

    return loss, logits


def loss_fn(model: RubbishClassifier, batch):
    """标准损失函数（向后兼容）"""
    logits = model(batch["data"])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch["label"]).mean()
    return loss, logits


@nnx.jit
def train_step(model: RubbishClassifier, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch, class_weights=None):
    """Train for a single step."""
    if class_weights is not None:
        grad_fn = nnx.value_and_grad(lambda m, b: weighted_loss_fn(m, b, class_weights), has_aux=True)
    else:
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # In-place updates.
    optimizer.update(grads)  # In-place updates.


@nnx.jit
def eval_step(model: RubbishClassifier, metrics: nnx.MultiMetric, batch, class_weights=None):
    if class_weights is not None:
        loss, logits = weighted_loss_fn(model, batch, class_weights)
    else:
        loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # In-place updates.


@nnx.jit
def pred_step(model: RubbishClassifier, batch):
    logits = model(batch["data"])
    return logits.argmax(axis=1)


if __name__ == "__main__":
    # hyperparameters
    batch_size = 64
    use_class_weights = True  # 是否使用类别权重

    writer = SummaryWriter(log_dir="./logs")

    # 加载类别权重
    class_weights = None
    if use_class_weights:
        class_weights = load_class_weights("./data/class_weights.npy")

    # Load the data.
    train_dataset = Dataset(root_dir="./data", mode="train")
    val_dataset = Dataset(root_dir="./data", mode="val")

    # Instantiate the model.
    # 注意：这里应该使用12个类别而不是4个
    model = RubbishClassifier(num_classes=12, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-6, b1=0.9))
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    best_acc = 0
    train_loader = dataloader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_epochs=50)

    with ocp.CheckpointManager(
        os.path.join(os.getcwd(), "checkpoints/"),
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    ) as mngr:
        for step, batch in enumerate(train_loader):
            train_step(model, optimizer, metrics, batch, class_weights)
            train_metrics = metrics.compute()
            writer.add_scalar("train/loss", train_metrics["loss"], step)
            writer.add_scalar("train/accuracy", train_metrics["accuracy"], step)

            if step > 0 and step % 500 == 0:
                print("Step:{}_Train Acc@1: {:.4f} loss: {:.4f}".format(step, train_metrics["accuracy"], train_metrics["loss"]))
            metrics.reset()  # Reset the metrics for the train set.

            if step > 0 and step % 500 == 0:
                # Compute the metrics on the test set after each training epoch.
                val_loader = dataloader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_epochs=1)
                for val_batch in val_loader:
                    eval_step(model, metrics, val_batch, class_weights)
                val_metrics = metrics.compute()
                print("Step:{}_Val Acc@1: {:.4f} loss: {:.4f}".format(step, val_metrics["accuracy"], val_metrics["loss"]))

                writer.add_scalar("val/loss", val_metrics["loss"], step)
                writer.add_scalar("val/accuracy", val_metrics["accuracy"], step)

                if val_metrics["accuracy"] > best_acc:
                    best_acc = val_metrics["accuracy"]
                    _, state = nnx.split(model)
                    mngr.save(step, args=ocp.args.StandardSave(state))
                    print(f"新的最佳验证准确率: {best_acc:.4f}, 模型已保存")

                metrics.reset()  # Reset the metrics for the val set.

    writer.close()
    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")
