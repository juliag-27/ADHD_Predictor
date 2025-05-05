# 🌎 WiDS Datathon 2025
Unraveling the Mysteries of the Female Brain: Sex Patterns in ADHD

[Link](https://www.kaggle.com/competitions/widsdatathon2025/overview) to competition on Kaggle

## **🧠 Meet Team Synapse:**
| Name | Contribution |
|------|-------|
| [Maya Patel](https://github.com/mpate154) | FMC Data: Experimenting with weighted classes, Oversampling and Undersampling, Node Connections, Optuna, Different GCN layers and Dropout, and Multi-Input model involving all 3 datasets. Quantitative Data: Exploration, Optimization |
| [Julia Gu](https://github.com/juliag-27) | Quantitative Data: Upload, Exploration, Initial Models, Export. GCN: Exploration, Optuna Optimization on Accuracy & Optuna Optimization on F1 Score, Export |
| [Kayla DePalma](https://github.com/kdepalma5) | Categorical Data: Exploratory Data Analysis + Visualization, Models, Hyperparameter Tuning, Feature Selection. Worked on Ensemble Model for Quantitative and Categorical data |
| [Jannatul Nayeem](https://github.com/jannatulnayeem964) |  Categorical Data Exploration + Model, Tested Different Sampling Techniques to Mitigate Bias: (Oversample, Undersample, Hybrid), Worked on Ensemble Model|

<br/>

## **💡Project Overview**
As fellows in the **Break Through Tech AI Program**, we participated in the WiDS Datathon 2025 on Kaggle. The WiDS Datathon Global Challenge was developed in partnership with the Ann S. Bowers Women’s Brain Health Initiative (WBHI), Cornell University, and UC Santa Barbara. The datasets and support are provided by the Healthy Brain Network (HBN), the signature scientific initiative of the Child Mind Institute, and the Reproducible Brain Charts project (RBC). This challenge provides a valuable opportunity to strengthen our data science skills while tackling an interesting and critical social impact challenge!

## **✅ Objective**
The goal of the competition is to develop a predictive model that accurately predicts both an ***individual’s sex*** and their ***ADHD diagnosis*** using functional brain imaging data of children and adolescents and their socio-demographic, emotions, and parenting information. The challenge lies in handling complex, potentially imbalanced datasets and extracting meaningful patterns that improve prediction accuracy.

## **🎯 Project Highlights**

* Built a Graph Convolution Network (GCN) using an adjacency matrix with a half-connected graph based on Functional Connectome Matrix data to predict ADHD Outcome and Sex labels. 
* Achieved an ***F1 score of 0.74*** and a ranking of 145 on the final Kaggle Leaderboard
* Used summations of the models guesses in each label to interpret model decisions
* Implemented device-aware computation using torch.device and reduced graph connectivity to lower computational overhead, while leveraging NumPy arrays for efficient data handling within Google Colab's hardware constraints.

🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

## **👩🏽‍💻 Setup & Execution**

1. For this project, we primarily relied on Google Colab. Instructions to reproduce this data are below.
2. Download the .ipynb notebook 
3. Open in the preferred editor (VSCode, Google Colab)
4. Select the T4 GPU option (in Colab, under Runtime -> Change Runtime Type)
5. Run the uploader, which uploads the data from a public Google Drive link 
6. Run all cells that include necessary installations and imports
7. Most constructions of models will have print statements upon completion indicating the accuracy, train/test score, and F1 score. 
8. Export data using cells at the end of the file 

## **📊 Data Exploration**

Training datasets provided by the competition included 4 types of information from over 1,200 subjects and were named:
* TRAIN_CATEGORICAL_METADATA.xlsx
    * The categorical dataset contains participant-level information across demographic, familial, and MRI-related dimensions. It includes 10 features (plus participant_id) that are largely categorical or ordinal in nature, sourced from validated instruments and demographic surveys.
    * Demographic:
         * Basic_Demos_Enroll_Year: Year the participant enrolled in the study (6 unique values)
         * Basic_Demos_Study_Site: Location of the phenotypic testing (4 unique sites)
    * Family Background (Barratt Scale):
         * Barratt_Barratt_P1_Edu & Barratt_Barratt_P2_Edu: Education levels of Parent 1 and Parent 2, with 7 categories ranging from less than 7th grade to graduate/professional training.
         * Barratt_Barratt_P1_Occ & Barratt_Barratt_P2_Occ: Parent occupations (10 categories), including stay-at-home, laborers, and professionals
    * MRI Scan Information:
         * MRI_Track_Scan_Location: Location where the participant's MRI was performed (4 locations, i.e. Staten Island, CBIC).
    * Child's Demographics:
         * PreInt_Demos_Fam_Child_Ethnicity: Ethnicity (i.e. Hispanic or Latino, Not Hispanic)
         * PreInt_Demos_Fam_Child_Race: Race (10 categories including White, Black, Asian, Multiracial).
    * ID Field:
         * participant_id: A unique identifier for each subject (1213 unique entries)
    * Data Quality Notes:
       * Some variables, especially Parent 2 contain substantial missing data (up to 18%)
       * All categorical variables are either enoded as integers or floats, and many have clear label definitions available via the data dictionary (i.e. 0 = "Not Hispanic", 1 = "Hispanic or Latino"
    * Example of Types of Data Exploration Done with Categorical Data
       * ![image](https://github.com/user-attachments/assets/f298fe53-c3a1-4f25-a26d-7fe1d895380a)
       * ![image](https://github.com/user-attachments/assets/5ad965dd-7c8b-4012-b47c-65a7ae6b8326)
       * ![image](https://github.com/user-attachments/assets/a6f8ac18-fbf0-4225-b2e5-45f5d807714d)
       * ![image](https://github.com/user-attachments/assets/6f337987-70a9-4e90-9833-c3acec12a7f1)
       * ![image](https://github.com/user-attachments/assets/b365efff-07c4-4d51-bdba-8a5d24956611)
       * ![image](https://github.com/user-attachments/assets/edd4944a-c18c-4c3c-a0de-f101a3b11270)
       * ![image](https://github.com/user-attachments/assets/c059b17e-bcd4-4d16-9737-5b91c793c8ed)
       * ![image](https://github.com/user-attachments/assets/0b7181af-a9d8-445a-b68a-aa734c4cdd0b)
* TRAIN_QUANTITATIVE_METADATA.xlsx
    * Examples of Data Exploration with Quantitative Data
      * ![image](https://github.com/user-attachments/assets/fd771b49-58b3-49c9-8847-4575f720c51a)
      * ![image](https://github.com/user-attachments/assets/3840bfbf-308a-4cd8-8f0a-66f234f28da3)

* TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv
    * Functional MRI connectome matrices of 999 regions of the brain where redundant (for example region 88's correlation with itself) were removed. For this dataset, adjacency matrices and connected graphs were used in order to utilize the dataset for training.
   * Visualization:
    * ![image](https://github.com/user-attachments/assets/5f8bb491-fc2b-415f-a2cd-639f77fbcaed)

* Corresponding test files for each dataset
    * Included the target variables Sex and ADHD diagnosis.
    * |            | 0      | 1            |
      |------------|--------|--------------|
      |ADHD Outcome| No ADHD| Has ADHD     |
      |Sex_F       | Male   | Female       |
      
![image](https://github.com/user-attachments/assets/ae3e3351-1237-4547-9212-6924f24f8b7c)


To conduct data exploration we:
* Used statistical methods such as .info() and .describe()
* Constructed histograms, bar plots, cross tab plots, and count plots to visualize distributions

During exploratory data analysis, we discovered:
* More training data samples were classified as having ADHD than not (831 with ADHD and 382 without)
* There were more males than females in the training data (797 males and 416 females)
* There were no clear trends associated with ADHD and gender with the categorical features
* Connectivity between nodes (the 200 brain regions) ranges from -0.2 to 0.4

## **🛠️ Model Development**

We used different models depending on which of the three datasets we worked with, listed below.

Quantitative 
* Decision Trees
* Gradient Boosted Classifiers 
* Logistic Regressors

Categorical 
* Decision Trees
* Gradient Boosted Classifiers
Functional Connectome Matrices
* Graph Convolutional Networks 

Key model development techniques also varied widely across the three datasets. They included:
* Oversampling, Undersampling, Hybrid Sampling
* Model Selection
* One Hot Encoding
* Feature selection based on exploratory data analysis
* Grid Search
* Optuna
* Ensemble methods
* Multi-Input Models

A variety of tools and libraries were implemented for these techniques, specifically,
* Torch Geometric 
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Optuna 
* Scikit-Learn
---

## **✨ Our Submitted Model**

We submitted a logistic regression model which utilized the quantitative dataset predicting for ADHD. Our final coefficients for each feature in the model was  [[-0.01324583 -0.0706093  -0.13277412 -0.04720644 -0.13399689  0.10343249   0.09659269  0.0764005  -0.29448281  0.27701216 -0.20428486  0.50050989   0.44302516  0.94194721 -0.07912481  0.07665316  0.14459435]], with an intercept of 1.15889162. The test and train accuracy was 0.806 for each. Notably, the column with the highest absolute weight was the SDQ_SDQ_Internalizing column, indicating that subjects with anxiety, depression, and social withdrawal indicate a higher risk for ADHD. The model took a few seconds to train. In terms of scoring, we knew women with ADHD were a minority in the dataset, so women were predicted for every example. This ended up working the best because the scoring for our predictions were heavily limited by the Sex_F prediction and the strategy used to calculate the final F1 score on Kaggle puts more weight on the cases which include women who have ADHD. Notably, there were significantly more men in the training dataset, thus introducing difficulty in accurately predicting it with a specific model. 

## **📈 Results & Key Findings**

Evaluation metrics primarily focused on F1 Score and Accuracy. Some models were also analyzed for performance on a test dataset (a subset of training data that is not used in training).

Our output dataset performed within the top 200 submissions in the WiDS Datathon, with an F1 score of 0.75. On the training set, our ADHD models had accuracy of 84.5% and train/test/split accuracy of 67.5%. On the other hand, the Sex_f models had an accuracy of 83.02% and a train/test/split accuracy of 62.3%. Because of the nature of the dataset and F1 evaluation criteria, we found a significant improvement of our Kaggle submission by predicting female for most samples.


## **🖼️ Impact Narrative**
ADHD has historically been underdiagnosed and understudied in females, contributing to gaps in care and support. It affects approximately 11% of adolescents, with around 14% of boys and 8% of girls receiving a diagnosis. However, evidence suggests that girls with ADHD are often overlooked because they tend to exhibit more inattentive symptoms, which are harder to detect. As a result, undiagnosed girls may continue to struggle with symptoms that burden their mental health and daily functioning. Machine learning models capable of predicting ADHD and identifying gender-specific patterns could improve early detection, particularly in females, where diagnosis is more challenging. Additionally, these models can provide insights into the brain mechanisms underlying ADHD in both males and females, paving the way for more targeted and effective personalized treatments. Early identification and tailored therapies could significantly enhance mental health outcomes for individuals with ADHD.

<br/>

## **🚀 Next Steps & Future Improvements**

* Potential Limitations
  
The model is subject to the following limitations: The dataset used was imbalanced, with a higher proportion of male subjects compared to female subjects, which may lead to greater-related bias in the predictions. Additionally, incorporating the categorical dataset may have introduced potential bias, as certain categories could disproportionately influence the model’s decisions. We also observed the risk of overfitting or underfitting, particularly due to the high ADHD and low female predictions, which resulted in inflated scores in the Kaggle submission. 

* What We Would Do Differently Given Time/Resources
  
With more time and resources, we would invest in exploring different model architectures and experiment with ensemble methods to enhance predictive performance. We would also dedicate more effort to extensive hyperparameter tuning to optimize the model’s accuracy. With additional resources, we would also consider using larger, more complex architectures to find more patterns and improve accuracy. Additionally, we would seek out larger or complementary datasets, particularly with more female subjects, to address the gender imbalance and reduce potential bias. 

* Additional Datasets/Techniques We Would Explore
  
To further enhance our model, we would explore additional datasets incorporating external, publicly available data. We would also experiment with the weights of each dataset and how much it factors into the overall prediction. We would also experiment with other ensemble techniques and deep learning models to improve predictive performance. 


---

## **📄 References & Additional Resources**

Alhamid, Mohammed. “Ensemble Models: What Are They and When Should You Use Them?” builtin, builtin, 2025, Ensemble Models: What Are They and When Should You Use Them? Accessed 24 2 2025.

Preferred Networks, Inc. “Optuna.” Optuna, 2017, https://optuna.org/. Accessed 11 2 2025.

PyG Team. “PyG Documentation.” pytorch-geometric, 2025, https://pytorch-geometric.readthedocs.io/en/latest/index.html. Accessed 28 1 2025.



