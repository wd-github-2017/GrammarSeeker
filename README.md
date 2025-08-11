# GrammarSeeker-SFT-Qwen2.5-7B

A fine-tuned Qwen2.5-7B-Instruct model specifically designed for grammatical project parsing systems.

## 🔗 Repository Links

- **📁 Source Code**: [GitHub Repository](https://github.com/wd-github-2017/GrammarSeeker) - Contains testing code and development scripts
- **🤗 Model Hub**: [Hugging Face Model](https://huggingface.co/WangDong2017/GrammarSeeker-SFT-Qwen2.5-7B) - Hosts the complete fine-tuned model

## 🎉 Latest Update (2025-08-11)

**✅ Model Successfully Deployed to Hugging Face!**

The fine-tuned model is now available for direct use without any additional steps.

## 📋 Model Information

- **Base Model**: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Fine-tuned Model**: [WangDong2017/GrammarSeeker-SFT-Qwen2.5-7B](https://huggingface.co/WangDong2017/GrammarSeeker-SFT-Qwen2.5-7B)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) during training, now provided as complete model
- **Task**: Binary classification for grammatical project annotation (T/F output)
- **Performance**: 
  - **F1 Score**: 0.9797 (97.97%)
  - **Positive Accuracy**: 0.9640 (96.40%)
  - **Negative Accuracy**: 0.9960 (99.60%)
  - **Test Samples**: 1000
  - **Test Date**: 2025-08-11

## 🎯 Use Case

This model serves as the **core component of a grammatical project parsing system**. It is designed to:

1. **Receive structured prompts** (as shown in GM-TestData.csv)
2. **Output binary decisions** (T/F) for grammatical annotation
3. **Enable automated grammar project marking** based on model predictions

## 🔧 Usage

### Installation

```bash
pip install transformers peft torch
```

### Testing Performance

```bash
# Test the model from Hugging Face
python test_hf_model.py
```

**Latest Test Results (2025-08-11)**:
- ✅ Model successfully loaded from HF repository
- ✅ All 1000 test samples processed
- ✅ F1 Score: 0.9797 (97.97%)
- ✅ Test completed in ~1 minute on RTX 4090

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the complete fine-tuned model directly from HF
model = AutoModelForCausalLM.from_pretrained(
    "WangDong2017/GrammarSeeker-SFT-Qwen2.5-7B",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("WangDong2017/GrammarSeeker-SFT-Qwen2.5-7B")
```

## 🏭 Production Environment Usage

**Recommended workflow**:

1. **Pre-filtering**: Use regular expressions for coarse screening
2. **String matching**: Trigger prompt generation based on string matching
3. **Model inference**: Send generated prompt to this model
4. **Output processing**: Model outputs T/F
5. **Automatic annotation**: Generate grammatical project markers based on T/F output

## 📊 Dataset

- **GM-TestData.csv**: 1000 test samples with prompts and expected answers
- **Format**: prompt1, prompt2, answer (T/F)
- **Test Results**: Successfully validated with 97.97% F1 score

## 🔬 Technical Details

- **Model Size**: ~15GB (complete fine-tuned model)
- **Performance**: 10 annotations per second on RTX 4090
- **Input Format**: Structured prompts following Qwen chat template
- **Output Format**: Binary classification (T/F)
- **Deployment Status**: ✅ Successfully deployed to Hugging Face

## 🚀 Deployment & Integration

### Hugging Face Integration

- **Model Hub**: [WangDong2017/GrammarSeeker-SFT-Qwen2.5-7B](https://huggingface.co/WangDong2017/GrammarSeeker-SFT-Qwen2.5-7B)
- **Direct Loading**: Available for immediate use
- **API Access**: Can be deployed through HF Inference API

### Local Development

```bash
# Clone the repository
git clone https://huggingface.co/WangDong2017/GrammarSeeker-SFT-Qwen2.5-7B

# Test the model
python test_hf_model.py
```

## 📝 Citation

```bibtex
@misc{GrammarSeeker-SFT-Qwen2.5-7B,
  title={GrammarSeeker-SFT-Qwen2.5-7B},
  author={Wang Dong},
  year={2025},
  url={https://huggingface.co/WangDong2017/GrammarSeeker-SFT-Qwen2.5-7B}
}
```

## 🔗 Quick Links

- **📁 Source Code**: [GitHub Repository](https://github.com/wd-github-2017/GrammarSeeker)
- **Model Hub**: [WangDong2017/GrammarSeeker-SFT-Qwen2.5-7B](https://huggingface.co/WangDong2017/GrammarSeeker-SFT-Qwen2.5-7B)
- **Test Script**: `test_hf_model.py`
- **Performance**: 97.97% F1 score on test dataset
- **Model Type**: Complete fine-tuned model (no adapter needed)

---

**Note**: This model has been successfully tested and deployed. For production use, please ensure proper testing and validation in your specific use case. 