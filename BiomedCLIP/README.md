---
license: mit
language:
- en
pipeline_tag: zero-shot-classification
---

### Huggingface-friendly BiomedCLIP

1. pure torch and huggingface-based implementation of the original microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
2. rename the checkpoint state key names.

### Usage

```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
```
