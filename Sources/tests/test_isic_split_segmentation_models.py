import gc

import torch

from segmentation.HSFL.models.ISIC_ResNet50 import (
    ISICClientModelResNet50 as HSFLClient,
    ISICCloudModelResNet50 as HSFLCloud,
    ISICEdgeModelResNet50 as HSFLEdge,
)
from segmentation.SplitFL.models.ISIC_ResNet50 import (
    ISICClientModelResNet50 as SplitFLClient,
    ISICMergedModelResNet50 as SplitFLMerged,
    ISICServerModelResNet50 as SplitFLServer,
)


def test_splitfl_isic_resnet50_shapes_without_pretrained_download():
    client = SplitFLClient(weights=None).eval()
    server = SplitFLServer(weights=None).eval()
    image = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        smashed = client(image)
        bottleneck = server.encoder(smashed)
        output = server.decoder(bottleneck)

    assert smashed.shape == (1, 512, 28, 28)
    assert bottleneck.shape == (1, 2048, 7, 7)
    assert output.shape == (1, 1, 224, 224)

    del client, server, smashed, output
    gc.collect()

    merged = SplitFLMerged(weights=None).eval()
    with torch.no_grad():
        assert merged(image).shape == (1, 1, 224, 224)


def test_hsfl_isic_resnet50_shapes_without_pretrained_download():
    client = HSFLClient(weights=None).eval()
    edge = HSFLEdge(weights=None).eval()
    cloud = HSFLCloud(weights=None).eval()
    image = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        client_smashed = client(image)
        edge_smashed = edge(client_smashed)
        bottleneck = cloud.feature_part(edge_smashed)
        output = cloud.decoder(bottleneck)

    assert client_smashed.shape == (1, 256, 56, 56)
    assert edge_smashed.shape == (1, 1024, 14, 14)
    assert bottleneck.shape == (1, 2048, 7, 7)
    assert output.shape == (1, 1, 224, 224)
