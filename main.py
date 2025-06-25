import os

# optax for optimizer
import optax

# The Flax NNX API.
from flax import nnx

# grain to load data
from loaderx import dataloader

# orbax to save model
import orbax.checkpoint as ocp
from pathlib import Path

ckpt_dir = Path(Path.cwd() / "./checkpoints")
# model
from net import RubbishClassifier

# load data
from feeder import Dataset


def loss_fn(model: RubbishClassifier, batch):
    logits = model(batch["data"])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch["label"]).mean()
    return loss, logits


@nnx.jit
def train_step(model: RubbishClassifier, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # In-place updates.
    optimizer.update(grads)  # In-place updates.


@nnx.jit
def eval_step(model: RubbishClassifier, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # In-place updates.


@nnx.jit
def pred_step(model: RubbishClassifier, batch):
    logits = model(batch["data"])
    return logits.argmax(axis=1)


if __name__ == "__main__":
    # hyperparameters
    batch_size = 16

    # Load the data.
    train_dataset = Dataset(root_dir="./data", mode="train")
    val_dataset = Dataset(root_dir="./data", mode="val")
    train_loader = dataloader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_epochs=50)
    val_loader = dataloader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_epochs=50)

    # Instantiate the model.
    model = RubbishClassifier(num_classes=40, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-6, b1=0.9))
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    best_acc = 0
    with ocp.CheckpointManager(
    os.path.join(os.getcwd(), 'checkpoints/'),
    options = ocp.CheckpointManagerOptions(max_to_keep=1),
    ) as mngr:
        for step, batch in enumerate(train_loader):
            # batch = {"data": jnp.asarray(batch["data"], dtype=jnp.float32), "label": jnp.asarray(batch["label"], dtype=jnp.int32)}
            train_step(model, optimizer, metrics, batch)
            if step > 0 and step % 500 == 0:
                train_metrics = metrics.compute()
                print("Step:{}_Train Acc@1: {} loss: {} ".format(step, train_metrics["accuracy"], train_metrics["loss"]))
                metrics.reset()  # Reset the metrics for the train set.

                # Compute the metrics on the test set after each training epoch.
                for val_batch in val_loader:
                    eval_step(model, metrics, val_batch)
                val_metrics = metrics.compute()
                print("Step:{}_Val Acc@1: {} loss: {} ".format(step, val_metrics["accuracy"], val_metrics["loss"]))
                if metrics.compute()["accuracy"] > best_acc:
                    best_acc = metrics.compute()["accuracy"]
                    _, state = nnx.split(model)
                    mngr.save(step, args=ocp.args.StandardSave(state))
                metrics.reset()  # Reset the metrics for the val set.
