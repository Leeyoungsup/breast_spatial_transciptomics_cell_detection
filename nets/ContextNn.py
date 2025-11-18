import math

import torch

from utils.util import make_anchors


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, e=0.5):
        super().__init__()
        self.conv1 = Conv(ch, int(ch * e), torch.nn.SiLU(), k=3, p=1)
        self.conv2 = Conv(int(ch * e), ch, torch.nn.SiLU(), k=3, p=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
        self.conv2 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
        self.conv3 = Conv(2 * (out_ch // 2), out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0),
                                         Residual(out_ch // 2, e=1.0))

    def forward(self, x):
        y = self.res_m(self.conv1(x))
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r), torch.nn.SiLU())
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, torch.nn.SiLU())

        if not csp:
            self.res_m = torch.nn.ModuleList(Residual(out_ch // r) for _ in range(n))
        else:
            self.res_m = torch.nn.ModuleList(CSPModule(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2, torch.nn.SiLU())
        self.conv2 = Conv(in_ch * 2, out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat(tensors=[x, y1, y2, self.res_m(y2)], dim=1))


class Attention(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5

        self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, torch.nn.Identity())

        self.conv1 = Conv(ch, ch, torch.nn.Identity(), k=3, p=1, g=ch)
        self.conv2 = Conv(ch, ch, torch.nn.Identity())

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)

        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
        return self.conv2(x)


class PSABlock(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.conv1 = Attention(ch, num_head)
        self.conv2 = torch.nn.Sequential(Conv(ch, ch * 2, torch.nn.SiLU()),
                                         Conv(ch * 2, ch, torch.nn.Identity()))

    def forward(self, x):
        x = x + self.conv1(x)
        return x + self.conv2(x)


class PSA(torch.nn.Module):
    def __init__(self, ch, n):
        super().__init__()
        self.conv1 = Conv(ch, 2 * (ch // 2), torch.nn.SiLU())
        self.conv2 = Conv(2 * (ch // 2), ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(*(PSABlock(ch // 2, ch // 128) for _ in range(n)))

    def forward(self, x):
        x, y = self.conv1(x).chunk(2, 1)
        return self.conv2(torch.cat(tensors=(x, self.res_m(y)), dim=1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], torch.nn.SiLU(), k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p2.append(CSP(width[2], width[3], depth[0], csp[0], r=4))
        # p3/8
        self.p3.append(Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p3.append(CSP(width[3], width[4], depth[1], csp[0], r=4))
        # p4/16
        self.p4.append(Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p4.append(CSP(width[4], width[4], depth[2], csp[1], r=2))
        # p5/32
        self.p5.append(Conv(width[4], width[5], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p5.append(CSP(width[5], width[5], depth[3], csp[1], r=2))
        self.p5.append(SPP(width[5], width[5]))
        self.p5.append(PSA(width[5], depth[4]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[5], csp[0], r=2)
        self.h2 = CSP(width[4] + width[4], width[3], depth[5], csp[0], r=2)
        self.h3 = Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h4 = CSP(width[3] + width[4], width[4], depth[5], csp[0], r=2)
        self.h5 = Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h6 = CSP(width[4] + width[5], width[5], depth[5], csp[1], r=2)

    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
        return p3, p4, p5


class DFL(torch.nn.Module):
    # Generalized Focal Loss
    # https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class TissueContextEncoder(torch.nn.Module):
    """
    Tissue context encoder using ImageNet pre-trained classification backbone
    
    Supports multiple backbone architectures optimized for global tissue understanding
    """
    def __init__(self, context_dim=256, backbone='mobilenet_v3_small', pretrained=True):
        """
        Args:
            context_dim: output dimension of context features
            backbone: 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 
                     'efficientnet_b1', 'resnet18', 'resnet34'
            pretrained: use ImageNet pre-trained weights (recommended for better tissue features)
        """
        super().__init__()
        
        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError("torchvision required. Install with: pip install torchvision")
        
        self.backbone_name = backbone
        
        # Select backbone and get feature dimension (optimized for 512x512 images)
        if backbone == 'mobilenet_v3_small':
            base_model = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
            feat_dim = 576
            self.encoder = base_model.features
            self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        
        elif backbone == 'mobilenet_v3_large':
            base_model = models.mobilenet_v3_large(weights='IMAGENET1K_V2' if pretrained else None)
            feat_dim = 960
            self.encoder = base_model.features
            self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        
        elif backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            feat_dim = 1280
            self.encoder = base_model.features
            self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        
        elif backbone == 'efficientnet_b1':
            base_model = models.efficientnet_b1(weights='IMAGENET1K_V1' if pretrained else None)
            feat_dim = 1280
            self.encoder = base_model.features
            self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        
        elif backbone == 'resnet18':
            base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            feat_dim = 512
            self.encoder = torch.nn.Sequential(*list(base_model.children())[:-1])  # Remove FC
        
        elif backbone == 'resnet34':
            base_model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            feat_dim = 512
            self.encoder = torch.nn.Sequential(*list(base_model.children())[:-1])
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                           f"Choose from: mobilenet_v3_small, mobilenet_v3_large, efficientnet_b0, "
                           f"efficientnet_b1, resnet18, resnet34")
        
        # Context projection to desired dimension
        self.context_proj = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, context_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(context_dim, context_dim)
        )
        
        self.feat_dim = feat_dim
        self.context_dim = context_dim
    
    def forward(self, x):
        """
        Args:
            x: tissue context image [B, 3, H, W]
        Returns:
            context_features: [B, context_dim]
        """
        # Extract features with backbone
        x = self.encoder(x)
        
        # Global pooling (if not already done by backbone)
        if hasattr(self, 'global_pool'):
            x = self.global_pool(x)
        
        # Flatten
        x = x.flatten(1)
        
        # Project to context dimension
        x = self.context_proj(x)
        
        return x


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=(), use_context=False, context_dim=0):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.use_context = use_context

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box,torch.nn.SiLU(), k=3, p=1),
                                                           Conv(box, box,torch.nn.SiLU(), k=3, p=1),
                                                           torch.nn.Conv2d(box, out_channels=4 * self.ch,
                                                                           kernel_size=1)) for x in filters)
        
        # Classification branch with optional context fusion
        if use_context:
            # Late fusion: context features are fused at the classification stage
            self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
                                                               Conv(x, cls, torch.nn.SiLU()),
                                                               Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
                                                               Conv(cls, cls, torch.nn.SiLU())) for x in filters)
            
            # Context fusion layers for each detection scale
            self.context_fusion = torch.nn.ModuleList(
                torch.nn.Sequential(
                    torch.nn.Linear(context_dim, cls),
                    torch.nn.SiLU(),
                    torch.nn.Linear(cls, cls)
                ) for _ in filters
            )
            
            # Final classification layers after fusion
            self.cls_final = torch.nn.ModuleList(
                torch.nn.Conv2d(cls, out_channels=self.nc, kernel_size=1) for _ in filters
            )
        else:
            self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
                                                               Conv(x, cls, torch.nn.SiLU()),
                                                               Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
                                                               Conv(cls, cls, torch.nn.SiLU()),
                                                               torch.nn.Conv2d(cls, out_channels=self.nc,
                                                                               kernel_size=1)) for x in filters)

    def forward(self, x, context_features=None):
        """
        Args:
            x: list of feature maps from FPN
            context_features: tissue context features [B, context_dim] (optional)
        """
        if self.use_context and context_features is not None:
            # Late fusion with context
            for i, (box, cls, ctx_fusion, cls_final) in enumerate(zip(self.box, self.cls, 
                                                                        self.context_fusion, self.cls_final)):
                box_out = box(x[i])
                cls_feat = cls(x[i])  # [B, cls, H, W]
                
                # Fuse context features
                b, c, h, w = cls_feat.shape
                ctx_weight = ctx_fusion(context_features)  # [B, cls]
                ctx_weight = ctx_weight.view(b, c, 1, 1).expand_as(cls_feat)  # [B, cls, H, W]
                
                # Element-wise multiplication for late fusion
                cls_feat = cls_feat * (1 + ctx_weight)  # Modulate features with context
                cls_out = cls_final(cls_feat)
                
                x[i] = torch.cat(tensors=(box_out, cls_out), dim=1)
        else:
            # Standard forward without context
            for i, (box, cls) in enumerate(zip(self.box, self.cls)):
                x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        
        if self.training:
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0
            # cls (.01 objects, 80 classes, 640 image)
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes, use_context=False, 
                 context_backbone='resnet18', context_dim=256, context_pretrained=True):
        """
        Args:
            width, depth, csp: YOLO architecture parameters
            num_classes: number of detection classes
            use_context: whether to use tissue context encoder
            context_backbone: backbone for context encoder ('resnet18', 'resnet34', 'resnet50', 
                            'efficientnet_b0', 'mobilenet_v3_small')
            context_dim: dimension of context features
            context_pretrained: use ImageNet pretrained weights for context encoder
        """
        super().__init__()
        self.use_context = use_context
        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)
        
        # Tissue context encoder (optional)
        if use_context:
            self.context_encoder = TissueContextEncoder(
                context_dim=context_dim,
                backbone=context_backbone,
                pretrained=context_pretrained
            )
        else:
            context_dim = 0

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]), 
                        use_context=use_context, context_dim=context_dim)
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x, tissue_context=None):
        """
        Args:
            x: main input image [B, 3, H, W]
            tissue_context: tissue context image [B, 3, H, W] (optional)
        """
        # Extract context features if provided
        context_features = None
        if self.use_context and tissue_context is not None:
            context_features = self.context_encoder(tissue_context)
        
        # Main detection branch
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x), context_features)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v11_n(num_classes: int = 80, use_context: bool = False, 
               context_backbone: str = 'mobilenet_v3_small', context_dim: int = 256):
    """
    YOLOv11-nano with optional tissue context (optimized for 512x512 images)
    
    Args:
        num_classes: number of detection classes
        use_context: enable tissue context encoder
        context_backbone: 'mobilenet_v3_small' (default), 'mobilenet_v3_large', 
                         'efficientnet_b0', 'efficientnet_b1', 'resnet18'
        context_dim: dimension of context features (default: 256)
    
    Recommended backbones for 512x512:
        - mobilenet_v3_small: Fast, lightweight (576 features)
        - efficientnet_b0: Balanced accuracy/speed (1280 features)
        - resnet18: Higher accuracy (512 features)
    """
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, csp, num_classes, use_context, context_backbone, context_dim)


def yolo_v11_t(num_classes: int = 80, use_context: bool = False,
               context_backbone: str = 'mobilenet_v3_small', context_dim: int = 256):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 24, 48, 96, 192, 384]
    return YOLO(width, depth, csp, num_classes, use_context, context_backbone, context_dim)


def yolo_v11_s(num_classes: int = 80, use_context: bool = False,
               context_backbone: str = 'efficientnet_b0', context_dim: int = 256):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, csp, num_classes, use_context, context_backbone, context_dim)


def yolo_v11_m(num_classes: int = 80, use_context: bool = False,
               context_backbone: str = 'efficientnet_b1', context_dim: int = 384):
    csp = [True, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes, use_context, context_backbone, context_dim)


def yolo_v11_l(num_classes: int = 80, use_context: bool = False,
               context_backbone: str = 'resnet34', context_dim: int = 512):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes, use_context, context_backbone, context_dim)


def yolo_v11_x(num_classes: int = 80, use_context: bool = False,
               context_backbone: str = 'resnet34', context_dim: int = 512):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return YOLO(width, depth, csp, num_classes, use_context, context_backbone, context_dim)
