from flax import nnx
import jax.numpy as jnp
from functools import partial
from typing import Optional


class CNN(nnx.Module):
    """A simple CNN model"""

    def __init__(self, rngs: nnx.Rngs, return_latent=True, num_classes=3):
        self.return_latent = return_latent

        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        self.map = nnx.Linear(9984, 256, rngs=rngs)
        self.embed = nnx.Linear(256, 2, rngs=rngs)
        self.logits = nnx.Linear(2, num_classes, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.map(x))
        latent = self.embed(x)
        logits = self.logits(latent)
        if self.return_latent:
            return latent, logits
        else:
            return logits


class RubbishClassifier(nnx.Module):
    """改进的垃圾分类CNN模型"""

    def __init__(self, num_classes: int, rngs: nnx.Rngs):
        self.num_classes = num_classes

        # 卷积层
        self.conv1 = nnx.Conv(3, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(64, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv4 = nnx.Conv(128, 256, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv5 = nnx.Conv(256, 512, kernel_size=(3, 3), padding="SAME", rngs=rngs)

        # 批量归一化
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)
        self.bn2 = nnx.BatchNorm(64, rngs=rngs)
        self.bn3 = nnx.BatchNorm(128, rngs=rngs)
        self.bn4 = nnx.BatchNorm(256, rngs=rngs)
        self.bn5 = nnx.BatchNorm(512, rngs=rngs)

        # 池化层
        self.max_pool = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
        self.global_avg_pool = partial(nnx.avg_pool, window_shape=(1, 1), strides=(1, 1))

        # 全连接层
        self.fc1 = nnx.Linear(512, 256, rngs=rngs)
        self.fc2 = nnx.Linear(256, 128, rngs=rngs)
        self.fc3 = nnx.Linear(128, num_classes, rngs=rngs)

    def __call__(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = nnx.relu(x)
        x = self.max_pool(x)

        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = nnx.relu(x)
        x = self.max_pool(x)

        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = nnx.relu(x)
        x = self.max_pool(x)

        # 第四个卷积块
        x = self.conv4(x)
        x = self.bn4(x)
        x = nnx.relu(x)
        x = self.max_pool(x)

        # 第五个卷积块
        x = self.conv5(x)
        x = self.bn5(x)
        x = nnx.relu(x)

        # 全局平均池化
        x = jnp.mean(x, axis=(1, 2))  # 替代全局平均池化

        # 全连接层
        x = self.fc1(x)
        x = nnx.relu(x)

        x = self.fc2(x)
        x = nnx.relu(x)

        x = self.fc3(x)

        return x