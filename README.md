# SMS Spam Detection using DistilBERT

A Transformer-based SMS spam classifier fine-tuned on the SMS Spam Collection dataset.

# Live Demo
https://huggingface.co/spaces/fizzah26/sms-spam-detector
## Dataset
- **Source:** Kaggle — SMS Spam Collection (UCI)
- **Size:** 5,574 SMS messages
- **Classes:** Ham (4,827) | Spam (747)

## Technique
- **Model:** DistilBERT (Fine-tuned)
- **Library:** HuggingFace Transformers
- **Training:** Kaggle P100 GPU — 3 epochs

## Results
| Metric | Score |
|--------|-------|
| F1 (Spam) | 0.98 |
| Precision | 1.00 |
| Recall | 0.96 |
| Accuracy | 1.00 |

##  Files
| File | Description |
|------|-------------|
| SMS_Spam_Kaggle.ipynb | Training notebook |
| app.py | HuggingFace Spaces deployment |
| requirements.txt | Dependencies |

## Tech Stack
- Python
- HuggingFace Transformers
- DistilBERT
- Gradio
- Kaggle P100 GPU
