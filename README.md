# stable_stable_diffusion# Fine-tuning T5 for Prompt Generation in Diffusion Models  

This repository documents my attempt to fine-tune a T5 transformer model to improve prompt generation for diffusion models like Stable Diffusion. The project addresses challenges with handling indirect or complex prompts, such as generating realistic results for prompts like "a small four-legged animal."  

---

## Table of Contents  
1. [Overview](#overview)  
2. [Setup and Dependencies](#setup-and-dependencies)  
3. [Datasets](#datasets)  
4. [Model Fine-Tuning](#model-fine-tuning)  
5. [Testing the Model](#testing-the-model)  
6. [Results](#results)  
7. [Future Improvements](#future-improvements)  
8. [Repository Structure](#repository-structure)  

---

## Overview  

### Initial Goal  
The initial goal was to fine-tune Stable Diffusion's text prompt-handling architecture to make it more consistent in generating realistic outputs for indirect or complex prompts.  


### Alternative Approach  
I pivoted to fine-tuning a **T5 Transformer model** using a custom dataset. The dataset consists of **819 organisms** (animals, birds, fishes) and **5 descriptions per organism**, resulting in **4,095 name-description pairs**. The dataset was generated using **three different LLMs**.  

---

## Setup and Dependencies  

### Key Tools and Libraries  
- Python (v3.x)  
- PyTorch  
- Transformers (Hugging Face)  
- Google Colab for testing  
- Akka server for training  

### Installing Dependencies  
```bash  
pip install torch transformers datasets  
```  

### Hardware  
- Dataset generation was completed on a personal machine and Google Colab.  
- Model training and fine-tuning were conducted on UMD’s Akka server.  

---

## Datasets  

### Dataset Generation  
- **Dataset 1:** Generated using an h2o transformer (`h20_gen.py`).  
- **Dataset 2 & 3:** Generated using `stable_beluga_7b` and `zephyr` models.  
  - Short descriptions dataset  
  - Long descriptions dataset  

### Dataset Folder Structure  
All datasets are available in the `datasets/` folder.  

---

## Model Fine-Tuning  

The fine-tuning process uses the T5 transformer model.  

### File: `T5_3rd_iteration.py`  
This script preprocesses the data and fine-tunes the T5 model.  

**Steps:**  
1. Load datasets from the `datasets/` folder.  
2. Tokenize data and prepare for training.  
3. Fine-tune T5 with 5-6 epochs on Akka server.  

**Training Details:**  
- Training accuracy: ~99% within 5-6 epochs  
- Testing accuracy: ~39% due to dataset size and inconsistencies  

---

## Testing the Model  

### File: `T5_prompt_test.py`  
This script tests the fine-tuned T5 model on new prompts.  

### How to Test  
1. Run the `T5_prompt_test.py` file to evaluate model performance.  
2. Alternatively, use the **Google Colab Notebook** linked below to test outcomes with Stable Diffusion XL.  

---

## Results  

### Fine-Tuning Summary  
- **Training Time:** ~1 minute per epoch (Akka server)  
- **Evaluation Time:** A few minutes  
- **Dataset Generation Time:** 3-4 hours per dataset  

### Accuracy  
- **Training Accuracy:** ~99%  
- **Testing Accuracy:** ~39%  

### Current Challenges  
- Small dataset size  
- Dataset inconsistencies  

---

## Future Improvements  

1. Expand the dataset by generating more name-description pairs with varied prompts and additional LLMs.  
2. Improve dataset quality and consistency to enhance testing accuracy.  
3. Test the model with other transformer architectures to compare performance.  

---

## Repository Structure  

```plaintext  
📂 datasets/  
  ├── beluga_dataset/  
  ├── zephyr_dataset/  
  ├── h20_gen.py  

📂 models/  
  ├── T5_3rd_iteration.py  
  ├── T5_prompt_test.py  

📂 colab/  
  ├── Colab Notebook (Link in README)  

📂 results/  
  ├── Training Results Screenshots  
  ├── Testing Results Screenshots  
```  

---

## Links  

### Fine-Tuned Model  
[Download the fine-tuned T5 model here](#)  

### Google Colab Notebook  
[Access the Colab Notebook](https://colab.research.google.com/drive/1GAWY2GBg0ulklI8FKMbmeeFfUJp8Wnt-?usp=sharing)  


---  


