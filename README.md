# RadiologySumm


Radiology reports are essential in medical practice as they provide physicians with a detailed analysis of imaging examinations such as X-ray, CT, MRI and ultrasound. The Impression section of the report is a critical component, as it summarizes the most important findings in a clear and concise manner, allowing physicians to quickly understand the results and make informed decisions about patient care. Radiology reports are also useful in tracking the progress of a patient's condition over time, and for monitoring the effectiveness of treatment. However, the process of generating impressions by summarizing findings is time-consuming for radiologists and may introduce errors, highlighting the need for careful review and accurate reporting.

This repo provides code for generating impressions using descriptions of findings and the imagages via multi-modal modeling.

## **Dependency Management**:

- ```
  requirements.txt
  ```

  - Lists all the necessary Python libraries for the project.
  - Specific location not provided, but typically at the root of the project.

## Training Instructions

### Step 1: Environment Setup
Ensure that you have set up your environment as per the `requirements.txt` file and that your dataset is properly organized.

### Step 2: Configuration
Before starting the training, you need to modify the configuration settings in the `config_match.py` file located in the `src` directory. 

- Open `src/config_match.py`.
- Locate the line containing `MODE = 'train' # train, inference, embeds`.
- Change `MODE` to one of the following as per your requirement:
  - `'train'` for training the model.
  - `'inference'` for making predictions.
  - `'embeds'` for extracting embeddings.

### Step 3: Start Training
After setting the mode, run `radiology-multimodal.py` to start the training process (or prediction or embedding extraction, depending on your MODE setting).

### Step 4: Monitoring and Evaluation
Monitor the training process for any errors and evaluate the model’s performance using appropriate metrics.


# CITE

```
@inproceedings{wang-etal-2023-utsa,
    title = "{UTSA}-{NLP} at {R}ad{S}um23: Multi-modal Retrieval-Based Chest {X}-Ray Report Summarization",
    author = "Wang, Tongnian  and
      Zhao, Xingmeng  and
      Rios, Anthony",
    editor = "Demner-fushman, Dina  and
      Ananiadou, Sophia  and
      Cohen, Kevin",
    booktitle = "The 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.bionlp-1.58",
    doi = "10.18653/v1/2023.bionlp-1.58",
    pages = "557--566",
    abstract = "Radiology report summarization aims to automatically provide concise summaries of radiology findings, reducing time and errors in manual summaries. However, current methods solely summarize the text, which overlooks critical details in the images. Unfortunately, directly using the images in a multimodal model is difficult. Multimodal models are susceptible to overfitting due to their increased capacity, and modalities tend to overfit and generalize at different rates. Thus, we propose a novel retrieval-based approach that uses image similarities to generate additional text features. We further employ few-shot with chain-of-thought and ensemble techniques to boost performance. Overall, our method achieves state-of-the-art performance in the F1RadGraph score, which measures the factual correctness of summaries. We rank second place in both MIMIC-CXR and MIMIC-III hidden tests among 11 teams.",
}
```
