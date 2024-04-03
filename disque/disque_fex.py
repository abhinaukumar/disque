from typing import Any, Dict, Optional
import torch
from torchvision import transforms
import numpy as np
from videolib import Video
from qualitylib.feature_extractor import FeatureExtractor
from qualitylib.result import Result
from PIL import Image

from .disque_module import DisQUEModule


class DisqueFeatureExtractor(FeatureExtractor):
    NAME = 'DisQUE_fex'
    VERSION = '1.0'
    def __init__(self, ckpt_path: str, use_cache: bool = True, sample_rate: Optional[int] = None, batch_size: int = 1) -> None:
        super().__init__(use_cache, sample_rate)
        self.ckpt_path = ckpt_path
        self.batch_size = batch_size
        self.model = DisQUEModule.load_from_checkpoint(self.ckpt_path, match_sizes=True, strict=False).cuda()

    def _run_on_asset(self, asset_dict: Dict[str, Any]) -> Result:
        sample_interval = self._get_sample_interval(asset_dict)
        with Video(
            asset_dict['ref_path'], mode='r',
            standard=asset_dict['ref_standard'],
            width=asset_dict['width'], height=asset_dict['height']
            ) as v_ref:
            with Video(
                asset_dict['dis_path'], mode='r',
                standard=asset_dict['dis_standard'],
                width=asset_dict['width'], height=asset_dict['height']
                ) as v_dis:
                feats = None
                x_ref_batch = []
                x_ref_half_batch = []
                x_dis_batch = []
                x_dis_half_batch = []
                num_frames = min(v_ref.num_frames, v_dis.num_frames)
                for frame_ind, (frame_ref, frame_dis) in enumerate(zip(v_ref, v_dis)):
                    if frame_ind % sample_interval == 0:
                        with torch.no_grad():
                            image_ref = Image.fromarray(frame_ref.rgb.astype(frame_ref.standard.dtype), mode='RGB')
                            image_ref_half = image_ref.resize((image_ref.size[0]//2,image_ref.size[1]//2))
                            image_dis = Image.fromarray(frame_dis.rgb.astype(frame_dis.standard.dtype), mode='RGB')
                            image_dis_half = image_dis.resize((image_dis.size[0]//2,image_dis.size[1]//2))
                            x_ref = transforms.ToTensor()(image_ref).unsqueeze(0)
                            x_ref_batch.append(x_ref)
                            x_ref_half = transforms.ToTensor()(image_ref_half).unsqueeze(0)
                            x_ref_half_batch.append(x_ref_half)
                            x_dis = transforms.ToTensor()(image_dis).unsqueeze(0)
                            x_dis_batch.append(x_dis)
                            x_dis_half = transforms.ToTensor()(image_dis_half).unsqueeze(0)
                            x_dis_half_batch.append(x_dis_half)

                            if (len(x_ref_batch) == self.batch_size) or (frame_ind == num_frames-1):
                                x_ref_batch = torch.cat(x_ref_batch, dim=0)
                                _, a_ref = self.model.forward(x_ref_batch.cuda())
                                x_ref_half_batch = torch.cat(x_ref_half_batch, dim=0)
                                _, a_ref_half = self.model.forward(x_ref_half_batch.cuda())
                                a_ref = torch.cat([a_ref, a_ref_half], dim=1)
                                x_dis_batch = torch.cat(x_dis_batch, dim=0)
                                _, a_dis = self.model.forward(x_dis_batch.cuda())
                                x_dis_half_batch = torch.cat(x_dis_half_batch, dim=0)
                                _, a_dis_half = self.model.forward(x_dis_half_batch.cuda())
                                a_dis = torch.cat([a_dis, a_dis_half], dim=1)
                                a = torch.abs(a_ref - a_dis)

                        a = a.squeeze().cpu().numpy()
                        if feats is None:
                            feats = a
                        else:
                            feats += a

            feats = feats / (num_frames // sample_interval)
            print(f'Processed {asset_dict["dis_path"]}')
            return self._to_result(asset_dict, np.expand_dims(feats, 0))
