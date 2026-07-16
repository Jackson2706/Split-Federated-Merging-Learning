import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ehsfp.prototype_space import CosineClassifier, validate_prototype_space, COSINE_SPACES


def set_requires_grad(module, freeze=False):
    for param in module.parameters():
        param.requires_grad = not freeze


class ClientModel(nn.Module):
    """Client tier: full ResNet-18 backbone -> GAP -> [B, 512, 1, 1].

    The client emits a *pooled semantic vector* (kept in 4D [B,512,1,1] form so
    the downstream pooling/flatten plumbing is unchanged). Transmitting a
    class-mean of a post-GAP semantic vector is a faithful prototype; the old
    split transmitted a class-mean of a low-level [64,32,32] layer1 map, whose
    within-class variance dwarfs the between-class signal — that near-meaningless
    prototype (resampled as an isotropic Gaussian) was the root cause of the
    ~8% collapse. All the resnet capacity now sits before the prototype, so the
    per-class centroid is class-clustered and separable.
    """

    def __init__(self, freeze_backbone=False):
        super(ClientModel, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify for CIFAR input (32x32)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()

        # Client stops at layer2 (128-d) then GAP -> [B, 128, 1, 1]. This is a
        # deliberate memory/semantics trade-off: the codebase deep-copies one
        # client model per user (num_users=200), so a full backbone would need
        # ~27 GB just for client copies+Adam and OOMs. layer2 (0.68M params) is
        # ~1.6 GB for 200 copies, while its GAP-pooled 128-d feature is a *far*
        # more faithful prototype than a raw [64,32,32] layer1 map.
        self.part = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.avgpool,  # -> [B, 128, 1, 1]
        )
        set_requires_grad(self.part, freeze=freeze_backbone)

        # T1 experiment (env-toggled): protect the *pretrained* semantic blocks
        # from degenerate SupCon. With 200 users there are ~2.5 samples/class per
        # client, so most SupCon batches have no positive pairs -> the contrastive
        # gradient is noise that corrupts the ImageNet-pretrained backbone before
        # prototypes are extracted. Freezing layer1+layer2 keeps clean pretrained
        # features while leaving the (randomly-initialised, CIFAR-adapted) conv1
        # stem trainable so it can adapt to 32x32. Toggle with HSFP_FREEZE_BACKBONE=1.
        if os.environ.get("HSFP_FREEZE_BACKBONE", "0") == "1":
            set_requires_grad(resnet.layer1, freeze=True)
            set_requires_grad(resnet.layer2, freeze=True)
            # Keep frozen BN blocks in eval mode so their running stats (from
            # ImageNet) are used rather than being overwritten by tiny CIFAR batches.
            resnet.layer1.eval()
            resnet.layer2.eval()
            self._freeze_pretrained = True
        else:
            self._freeze_pretrained = False

    def train(self, mode: bool = True):
        """Keep frozen pretrained blocks in eval mode even when the client is
        set to train() during SSL (otherwise BN would recompute stats on the
        degenerate per-client batches)."""
        super().train(mode)
        if getattr(self, "_freeze_pretrained", False):
            # part indices 4 and 5 are layer1 and layer2
            self.part[4].eval()
            self.part[5].eval()
        return self

    def forward(self, x):
        return self.part(x)  # [B, 128, 1, 1]


class EdgeModel(nn.Module):
    """Edge tier: metric-refinement head over the pooled 512-d vector.

    Implemented with 1x1 convolutions so the tensor stays 4D ([B,512,1,1]);
    this keeps every AdaptiveAvgPool2d/flatten call in the pipeline a no-op /
    unchanged while acting as an MLP on the pooled feature vector. The edge and
    cloud aggregation levels (and the E-HSFP memory/reliability at both) are
    fully preserved — only the backbone location moved to the client.
    """

    def __init__(self, freeze_backbone=False):
        super(EdgeModel, self).__init__()

        # Refine the pooled 128-d client vector and project up to 256-d for the
        # cloud. 1x1 convs keep the tensor 4D ([B,C,1,1]).
        # GroupNorm (not BatchNorm): the edge trains on *synthetic* Gaussian
        # resamples but is evaluated on *real* features. BatchNorm's running
        # mean/var would be calibrated on the synthetic distribution and then
        # mis-normalize real features at eval — a direct cause of the "fits
        # synthetic, fails real" gap. GroupNorm has no running stats, so its
        # train and eval behavior are identical.
        self.part = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
        )
        set_requires_grad(self.part, freeze=freeze_backbone)

        # Contrastive projection head over the 256-d refined vector.
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        # Autocast can leave cached/synthetic features in bfloat16 on CPU while
        # Conv2d parameters remain FP32. Keep the tier boundary dtype-safe even
        # when callers supply a mixed-precision tensor.
        conv = self.part[0]
        x = x.to(device=conv.weight.device, dtype=conv.weight.dtype)
        return self.part(x)  # [B, 256, 1, 1]

    def forward_contrastive(self, out):
        out = out.view(out.size(0), -1)  # Flatten
        return self.fc(out)


class CloudModel(nn.Module):
    def __init__(self, args):
        super(CloudModel, self).__init__()
        self.prototype_space = validate_prototype_space(args.get("prototype_space"))
        if self.prototype_space in COSINE_SPACES:
            self.cosine_head = CosineClassifier(256, args["num_classes"])
            return
        # Input dim is 256 (the edge refiner's output vector).
        # Learnable scale for the L2-normalized input (NormFace-style): after
        # normalization the feature norm is 1, so we rescale before the MLP to
        # give the logits usable magnitude.
        self.feat_scale = nn.Parameter(torch.tensor(16.0))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, args["num_classes"]),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.prototype_space in COSINE_SPACES:
            return self.cosine_head(x)
        # L2-normalize so the classifier sees the SAME feature geometry whether
        # the input is a synthetic Gaussian resample (training) or a real
        # feature (eval). This removes the scale/norm mismatch between the two
        # distributions — a primary driver of the synthetic->real gap — and
        # matches the unit-sphere geometry the SupCon encoder was trained on.
        x = F.normalize(x, dim=1) * self.feat_scale
        return self.classifier(x)


# === Optional: test forward and gradient flow ===
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = ClientModel().to(device)
    edge = EdgeModel().to(device)
    cloud = CloudModel({"num_classes": 100}).to(device)

    x = torch.randn(4, 3, 32, 32).to(device)  # CIFAR-100 input
    x = client(x)          # [4, 512, 1, 1]
    x = edge(x)            # [4, 512, 1, 1]
    output = cloud(x)      # flattens internally -> [4, num_classes]

    y = torch.randint(0, 100, (4,)).to(device)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()

    # === Print gradient norms for debugging ===
    def print_gradients(model, name):
        print(f"\nGradient norms in {name}:")
        for pname, param in model.named_parameters():
            if "bn" in pname.lower():
                continue
            if param.grad is not None:
                print(f"{pname}: grad norm = {param.grad.norm().item():.6f}")
            else:
                print(f"{pname}: ❌ No gradient")

    print_gradients(client, "ClientModel")
    print_gradients(edge, "EdgeModel")
    print_gradients(cloud, "CloudModel")


if __name__ == "__main__":
    test()
