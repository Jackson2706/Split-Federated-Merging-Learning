import torch.nn.functional as F
from torch import nn

# class AlexNetClient_SplitFed(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         # 1. Base Feature Extractor (Standard)
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
#         )
        
#         # 2. HetBL Encoder [cite: 257]
#         # This layer compresses/encodes features to 'p' channels.
#         # 'p' varies: High-end uses wide_channels, Low-end uses narrow_channels.
#         # We read this from args["client_out_channels"].
#         self.compression_channels = args.get("client_out_channels", 64) 
#         self.bl_encoder = nn.Conv2d(64, self.compression_channels, kernel_size=3, padding=1)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.bl_encoder(x) # Output shape: [Batch, p, H, W]
#         return x

class AlexNetClient_SplitFed(nn.Module):
    def __init__(self, args):  # <--- Must accept args/config
        super().__init__()
        # ... standard layers ...
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Read the channel width from config (defaults to 64 if missing)
        self.compression_channels = args.get("client_out_channels", 64) 
        
        # Dynamic Output Layer
        self.bl_encoder = nn.Conv2d(64, self.compression_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.features(x)
        x = self.bl_encoder(x) # Returns [Batch, 76, H, W] if args set correctly
        return x
class AlexNetServer_SplitFed(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_classes = args["num_classes"]
        
        # 3. HetBL Decoder [cite: 257]
        # This layer receives 'p' channels and maps them back to 64 
        # so the rest of the server model can process it normally.
        # This fixes your RuntimeError (expected 64, got 76).
        input_channels = args.get("server_in_channels", 64)
        self.bl_decoder = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)

        # 4. Original Server Layers (unchanged internals)
        self.server_part = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # Decode the compressed input first
        x = self.bl_decoder(x) 
        # Then pass to standard layers
        x = self.server_part(x)
        return F.log_softmax(x, dim=1)


class AlexNetMergedModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Ensure client and server args match for the merged test model
        # For the merged model, we usually simulate the 'Wide' path (High-end)
        self.client_side_model = AlexNetClient_SplitFed(args)
        self.server_side_model = AlexNetServer_SplitFed(args)

    def load_weight(self, client_model_weight, server_model_weight):
        self.client_side_model.load_state_dict(client_model_weight)
        self.server_side_model.load_state_dict(server_model_weight)
    
    def forward(self, x):
        out = self.client_side_model(x)
        return self.server_side_model(out)