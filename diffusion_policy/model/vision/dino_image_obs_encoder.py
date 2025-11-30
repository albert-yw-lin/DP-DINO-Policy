from typing import Dict, Tuple, Union, Optional
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


def get_dinov3_convnext_tiny(
    repo_dir: Optional[str] = None,
    weights: Optional[str] = None,
    model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    device: Optional[torch.device] = None,
    use_local_model: bool = False
) -> nn.Module:
    """Load DINOv3 ConvNeXt Tiny model.

    Can load from either local Torch Hub checkpoint or Hugging Face Transformers.
    By default, uses Hugging Face. Set use_local_model=True to use local checkpoint.

    Args:
        repo_dir: Path to locally cloned DINOv3 repository (Torch Hub format)
        weights: Path or URL to the pre-trained weights compatible with Torch Hub
        model_name: Hugging Face model identifier
        device: Optional device to place the model on
        use_local_model: If True, attempt to load from local Torch Hub checkpoint.
                        If False (default), use Hugging Face Transformers.

    Returns:
        An nn.Module producing (B, D) features when given normalized images.
    """
    
    # Primary path: use local Torch Hub checkpoint if requested
    if use_local_model and repo_dir is not None and weights is not None:
        try:
            model = torch.hub.load(
                repo_or_dir=repo_dir,
                model="dinov3_convnext_tiny",
                source="local",
                weights=weights,
            )
            if device is not None:
                model = model.to(device)
            model.eval()
            # Don't freeze parameters here - let the workspace control freezing
            # for param in model.parameters():
            #     param.requires_grad = False
            return model
        except Exception as e:
            if use_local_model:
                raise RuntimeError(
                    f"Failed to load DINOv3 from local Torch Hub ({e}). "
                    f"Please check dinov3_repo_dir and dinov3_weights paths."
                ) from e
            else:
                print(f"Warning: Failed to load DINOv3 from Torch Hub ({e}). Falling back to Hugging Face.")

    # Default/Fallback: use Hugging Face Transformers implementation
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "transformers library is required to load DINOv3 from Hugging Face."
        ) from exc

    try:
        model = AutoModel.from_pretrained(model_name)
        if device is not None:
            model = model.to(device)
        model.eval()
        # Don't freeze parameters here - let the workspace control freezing
        # for param in model.parameters():
        #     param.requires_grad = False

        class DINOv3FeatureExtractor(nn.Module):
            def __init__(self, dinov3_model):
                super().__init__()
                self.model = dinov3_model

            def forward(self, x):
                outputs = self.model(pixel_values=x)
                if not hasattr(outputs, "pooler_output") or outputs.pooler_output is None:
                    raise ValueError(
                        "Hugging Face DINOv3 model did not return pooler_output."
                    )
                return outputs.pooler_output

        return DINOv3FeatureExtractor(model)
    except KeyError as e:
        if 'dinov3_convnext' in str(e):
            raise RuntimeError(
                f"Your transformers library version doesn't support dinov3_convnext model type. "
                f"Please either:\n"
                f"1. Update transformers: pip install git+https://github.com/huggingface/transformers.git\n"
                f"2. Or provide valid dinov3_repo_dir and dinov3_weights paths for Torch Hub loading."
            ) from e
        raise


class DINOImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=True,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=True,
            # DINOv3 specific: use local weights path if provided
            dinov3_repo_dir: Optional[str] = None,
            dinov3_weights: Optional[str] = None,
            use_local_model: bool = False
        ):
        """
        DINOv3-based multi-image observation encoder.
        
        Uses pre-trained DINOv3 ConvNeXt Tiny model for encoding RGB images.
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        dinov3_model = get_dinov3_convnext_tiny(
            repo_dir=dinov3_repo_dir,
            weights=dinov3_weights,
            model_name=model_name,
            device=getattr(self, "device", None),
            use_local_model=use_local_model,
        )

        # handle sharing vision backbone
        if share_rgb_model:
            key_model_map['rgb'] = dinov3_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    # Create a copy of the DINOv3 model for this key
                    this_model = copy.deepcopy(dinov3_model)
                
                if this_model is not None:
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                
                # configure normalizer
                # DINOv3 expects ImageNet normalization
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
        
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        
        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape

