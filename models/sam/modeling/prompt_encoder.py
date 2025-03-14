# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Optional, Tuple, Type
from ...common import LayerNorm2d
from typing import Union, List
from open_clip import create_model_from_pretrained, get_tokenizer
from transformers import CLIPTextModel, AutoModel, AutoTokenizer

class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        in_chan: int = 768,
        out_chan: int = 256
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)
        self.chan_transform = nn.Linear(in_chan, out_chan).to('cuda')
        self.attention_layer = SemanticAttention(embedding_dim=768)
        self.semantic_weight = nn.Parameter(torch.tensor(4.0))

        # self.text_encoder = AutoModel.from_pretrained('BiomedCLIP').text_model
        # self.tokenizer = AutoTokenizer.from_pretrained('BiomedCLIP')
        local_model_path = "cached_biomedclip_model.pth"
        if os.path.exists(local_model_path):
            checkpoint = torch.load(local_model_path)
            self.med_model = checkpoint['model']
            self.preprocess = checkpoint['preprocess']
            self.tokenizer = checkpoint['tokenizer']
        else:
            self.med_model, self.preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            torch.save({
                    'model': self.med_model,
                    'preprocess': self.preprocess,
                    'tokenizer': self.tokenizer
                }, local_model_path)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0) # [1, e, h, w]

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: List[Union[str, List[str]]],
        pad: bool,
    ):
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        # if pad:
        #     padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device) # [b, 1, 2]
        #     padding_label = -torch.ones((labels.shape[0], 1), device=labels.device) # [b, 1]
        #     points = torch.cat([points, padding_point], dim=1) # [b, 2, 2]
        #     labels = torch.cat([labels, padding_label], dim=1) # [b, 2]
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.med_model = self.med_model.to(device)
        points = points.to(device)
        class_embeddings = []
        if len(labels[0])>0:
            for label_set in labels:
                tokens = self.tokenizer(label_set).to(device)
                with torch.no_grad():
                    image_features, class_embedding, logit_scale = self.med_model(torch.empty(len(label_set), 3, 224, 224).to(device), tokens)
                    class_embeddings.append(class_embedding)
            class_embeddings = torch.stack(class_embeddings, dim=0)
            point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size) # [b, 2, 128]
            point_embedding = torch.cat([point_embedding, class_embeddings], dim=2)
            point_embedding = self.chan_transform(point_embedding)
            point_embedding[labels == -1] = 0.0
            point_embedding[labels == -1] += self.not_a_point_embed.weight
            point_embedding[labels == 0] += self.point_embeddings[0].weight
            point_embedding += self.point_embeddings[1].weight.unsqueeze(0)
        else:
            point_embedding = None
        return point_embedding

    def _embed_boxes(self, boxes: Tuple[torch.Tensor, List[Union[str, List[str]]]]) -> torch.Tensor:
        """Embeds box prompts."""
        coords = boxes[0] + 0.5  # Shift to center of pixel
        labels = boxes[1]
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.med_model = self.med_model.to(device)
        coords = coords.to(device)
        class_embeddings = []
        for label_set in labels:
                tokens = self.tokenizer(label_set).to(device)
                with torch.no_grad():
                    image_features, class_embedding, logit_scale = self.med_model(torch.empty(len(label_set), 3, 224, 224).to(device), tokens)
                    class_embeddings.append(class_embedding)
        class_embeddings = torch.stack(class_embeddings, dim=0)
        class_embeddings = class_embeddings.repeat_interleave(2, dim=1)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding = torch.cat([corner_embedding, class_embeddings], dim=2)
        corner_embedding = self.chan_transform(corner_embedding)
        b, n, c = corner_embedding.shape
        for i in range(0, n, 2):
            corner_embedding[:, i, :] += self.point_embeddings[2].weight
            corner_embedding[:, i+1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: Tuple[torch.Tensor, List[Union[str, List[str]]]]) -> torch.Tensor:
        """Embeds mask inputs."""
        coords = masks[0]
        labels = masks[1]
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.med_model = self.med_model.to(device)
        coords = coords.to(device)
        class_embeddings = []
        for label_set in labels:
                tokens = self.tokenizer(label_set).to(device)
                with torch.no_grad():
                    image_features, class_embedding, logit_scale = self.med_model(torch.empty(len(label_set), 3, 224, 224).to(device), tokens)
                    class_embeddings.append(class_embedding)
        class_embeddings = torch.stack(class_embeddings, dim=0)
        batch, num, height, width, chan = coords.shape
        _, _, in_chan = class_embeddings.shape
        coords = coords.view(-1, height, width, chan) # b*n, h, w, c
        mask_embedding = self.mask_downscaling(coords).view(batch, -1, height, width, chan)
        b, n, h, w, out_chan = mask_embedding.shape
        if out_chan != in_chan:
            class_embeddings = nn.Linear(in_chan, out_chan).to('cuda')(class_embeddings)
        class_embeddings = class_embeddings.unsqueeze(2).unsqueeze(3)  # Shape becomes [b, n, 1, 1, c]
        class_embeddings = class_embeddings.expand(-1, -1, h, w, -1)
        mask_embedding += class_embeddings
        return mask_embedding

    def _embed_texts(self, texts: List[Union[str, List[str]]]) -> torch.Tensor:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.med_model = self.med_model.to(device)
        class_embeddings = []
        for label_set in texts:
                tokens = self.tokenizer(label_set).to(device)
                with torch.no_grad():
                    image_features, class_embedding, logit_scale = self.med_model(torch.empty(len(label_set), 3, 224, 224).to(device), tokens)
                    class_embeddings.append(class_embedding)
        text_embedding = torch.stack(class_embeddings, dim=0)
        return text_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, List[Union[str, List[str]]]]],
        boxes: Optional[Tuple[torch.Tensor, List[Union[str, List[str]]]]],
        masks: Optional[Tuple[torch.Tensor, List[Union[str, List[str]]]]],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes[0].shape[0]
        elif masks is not None:
            return masks[0].shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, List[Union[str, List[str]]]]],
        boxes: Optional[Tuple[torch.Tensor, List[Union[str, List[str]]]]],
        masks: Optional[Tuple[torch.Tensor, List[Union[str, List[str]]]]],
        texts: Optional[List[Union[str, List[str]]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None and len(points[1][0])>0:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if texts is not None:
            text_embeddings = self._embed_texts(texts)
            
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            middle=dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1)
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings, text_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1 # coords: [h, w, 2] or [n, 2, 2]
        coords = coords @ self.positional_encoding_gaussian_matrix # [h, w, c] or [n, 2, c]
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1) # [h, w, 2c] or [n, 2, 2c]

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1] # [n, 1, 1] / w
        coords[:, :, 1] = coords[:, :, 1] / image_size[0] # [n, 1, 1] / h Normalization
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

class SemanticAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SemanticAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)

    def forward(self, semantic_embeddings, position_embedding):
        # Reshape position embedding to match attention requirements
        position_embedding = position_embedding  # Add batch dimension for attention
        attn_output, _ = self.attention(position_embedding, semantic_embeddings, semantic_embeddings)
        return attn_output  # Remove batch dimension
