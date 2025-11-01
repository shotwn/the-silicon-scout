# Study Diary
This is the diary for the study. I try not to edit previous days after they are written, so it will be full of mistakes, dead ends, wrong conclusions, wasted days and so on.

It is mostly for my own reference, so anything but the latest entries should be taken with a few kilos of salt.

## Initial Research
I learnt about LHC Olympics[^1], LLMs, Tokenization, LoRA, HuggingFace Transformers[^4]. I also had a few crash-course sessions about collision events, jet clustering, and other related physics topics. I noticed that I know almost nothing about these topics. But I was able to find some common practices and examples to follow.

## 2025-10-14
I am starting the study diary here. First day of the log is actually the 3rd attempt to have a go at this. This time rather than focusing on the training first, I am focusing on understanding the data.

### Understanding the Dataset
I was able to successfully unpack the R&D dataset[^2]. It was obvious that the data needed some sort of pre-processing before it can be used in training.

Data consisted of a variable number of particles per event (up to 700). Each particle had 3 features pt, eta, phi. By observing the common practice again, I set required mass parameter to 0. So assumed massless particles.

Using examples from the LHC Olympics 2020, I understood that collimating data to jets was a common way to reduce the complexity of the data. In examples pyjet was used to cluster particles in to jets. But this library was deprecated and the author recommended using fastjet instead. Pyjet was not able to work with current numpy version so I had to use fastjet. 

I was able to use fastjet with awkward arrays. But loading the entire dataset in to memory -although serviceable- was not ideal. I created a chunk based data loader, which solved the problem and allowed me to use multiprocessing to speed up the jet clustering.

I read more about jet clustering and found out that anti-kt algorithm was the most common one. I used this algorithm with a radius parameter of 0.5, which was also commonly used in examples.

I also wanted to keep the number of detections (particles) per event, so I added this as an additional feature to the data.

After clustering I had 1 or 2 jets per event with parameters px py pz E.

For each process I created a seperated jsonl file, also keeping signal and background events in seperate files. At the end of a successful process I merge all the files in to 2 files (signal and background).

See ./rd_data_processing.py

### Preparing the Dataset for Training
I knew I should merge signal and background files in a shuffled manner for training. I also needed to split the data in to train, validation and test sets. I created a script to do this.

Later on I modified the script to give 1:1 ratio between signal and background events. This is because during training, the LoRA model was learning to always predict the majority class (background) and was not learning anything useful.

Weighted loss function was another option, but I couldn't find a reliable implementation for Transformers just yet. Also I want to have some results in 1:1 ratio first, then I can experiment with weighted loss function later on.

See ./training_data_preparation.py

### Training the Model
I use `mistralai/Mistral-7B-Instruct-v0.3`[^3] model as my initial base model. Mainly because it is a well known open weight model and it is relatively small (7B parameters).

It became apparent early on that with my system only reliable way to inject LHC data to the model was using LoRA fine tuning. This allowed me to train the model with limited resources.

```
My system specs:
CPU: AMD Ryzen 7 7700X 8-Core Processor
RAM: 32 GB DDR5 (2x16GB) 6000 MT/s
GPU: NVIDIA GeForce RTX 3070 @ 8 GB GDDR6
SSD: 2 TB NVMe M.2 Samsung 990 Pro
OS: Windows 11 Pro or Ubuntu 24 via WLS2
```

At this point I reliant on a lot of LLM help (which can be more miss than hit). I used a lot of examples from other sources, but almost every API there is related to HuggingFace Transformers library was changed in last few months. So I had to do a lot of back and forth to get a working training script. 

Most of the parameters I've used in the training script were just copy-paste attempts to start any training going. So as my first training run started I spend a lot of time to research what each parameter meant and if it was suitable for my use case.

See ./train.py

### Validating the Model
I knew I would need to validate the model during and after training. I created a validation script to do this.

This was also challenging due quantized nature of the LoRA layer. One behavior I am still not sure of is receiving the full prompt echoed back to me as model output. But I was able to get only the added part by the model to validate the model.

See ./validate.py

### One Miss Step
By some late-night mistake, I decided to to use 1 or 0 instead of signnal or background labels. Which would be more efficient on a more classical ML model. But this was a unworthy effort for an LLM model, since these two words would get tokenized anyway. Not to mention I had to change the validation script and weighted loss function to accomodate this change. Which was a waste of time.

### Weights, weights, weights
On my first iterations I noticed the model was not learning anything useful. It quickly switched to predicting background background background (I had validation token set to 3 during those tests). I first tried to use a compute loss function, but HF Transformers changed this API recently too and it was not reliably working. Even when it worked I wasn't sure if my model was failing because of compute loss weights or something else. So I temporarily switched to 1:1 signal background ratio in the training data. This way I was sure the model was at least seeing both classes equally.

Which is probably not the best way to do it, but at this point I am desperate for any statistically significant results.

### Let it cook
I finished the day with a training run going.

## 2025-10-15
### Training Results
Training ran around 9 hours. I was able to get to checkpoint-7700. When I ran the validation I saw a result I did not expect. Model was not predicting anything, at all. 

After some heated discussions with LLMs, I realized the model has completely failed. The best result was around checkpoint-2000 giving 76% accuracy. After that model was picking a side "signal" or "background" and sticking to it. Which was dropping accuracy to 50% and staying there. After checkpoint-5000 model was not predicting anything, just echoing the prompt back to me.

I theorized what is happening is likely overfitting. Model was learning the training data too tightly and was not able to generalize to validation data. After a certain point, model was just memorizing the training data then crashing to one side, which was not useful at all.

### New Prompt & Values
I changed the prompt to be more specific. Which I didn't before to keep the prompt short but it was obvious now the results became more important than training time. 

I also added special tokens [INST] and [/INST] to help the model understand where the instruction starts and ends.

```python
def format_example(example):
    jets = example["jets"]
    s = "[INST] Classify this event as 'signal' or 'background'.\n"
    s += "jets:\n"
    for i, j in enumerate(jets):
        s += f"  jet{i+1}: px={j['px']:.10f} py={j['py']:.10f} pz={j['pz']:.10f} E={j['E']:.10f}\n"
    s += f"num_particles: {example['num_particles']}[/INST]"

    # HF Trainer expects 'labels'
    return {"input_text": s, "labels": example["type"]}
```

- I tuned the LoRA parameters to r=12, which is 1.5 times more than before. This should give more knobs to turn for LoRA layers. 
- With this I also set lora_alpha to 48. It was recommended to set lora_alpha to 4 times r. This should help with the scaling of the LoRA layers.
- I set lora_dropout from 0.1 to 0.15. Lora dropout randomly drops some of the information during training. This should help with overfitting.
- I set learning rate from 2e-4 to 3e-5. This should help with overfitting too. Since model was learning too fast and was overfitting the training data.
- I also added an optimizer that uses 8-bit precision. This should help with memory usage but I did not see any significant change in memory usage. I might remove this later if it is not helping.
- Finally I doubled gradient_accumulation_steps from 8 to 16. This should help with stability of training. Since my GPU is limited in memory, I can't increase batch size. But appearently increasing gradient accumulation steps is an alternative to increase effective batch size.

Checkpoint-500
```
Validation Accuracy: 0.69
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.63      0.88      0.74       493
      signal       0.81      0.51      0.62       507

    accuracy                           0.69      1000
   macro avg       0.72      0.69      0.68      1000
weighted avg       0.72      0.69      0.68      1000
```

Checkpoint-700
```
Validation Accuracy: 0.768
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.86      0.63      0.73       493
      signal       0.72      0.90      0.80       507

    accuracy                           0.77      1000
   macro avg       0.79      0.77      0.76      1000
weighted avg       0.79      0.77      0.76      1000
```

Checkpoint-900
```
Validation Accuracy: 0.793
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.83      0.72      0.78       493
      signal       0.76      0.86      0.81       507

    accuracy                           0.79      1000
   macro avg       0.80      0.79      0.79      1000
weighted avg       0.80      0.79      0.79      1000
```

Checkpoint-1100
```
Validation Accuracy: 0.804
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.80      0.81      0.80       493
      signal       0.81      0.80      0.81       507

    accuracy                           0.80      1000
   macro avg       0.80      0.80      0.80      1000
weighted avg       0.80      0.80      0.80      1000
```

3am at Checkpoint-1100 before overnight training I've reached 80.4% accuracy. This is a good improvement from previous runs. I will let it train overnight and see if it can increase accuracy further.

This version was promising. So I decided to tag it as v0.1. 

#### About Tokenization and Floating Point Numbers
I am worried about tokenization of floating point numbers. Since the model doesn't understand the pure numerical values but token representation of them we might be training just a simple memorization (eventhough the model doesn't have access to validation data during training).

Appearently this is a common problem with LLMs. Some solutions like using another layer of numerical model on top of LLM under LoRA is being suggested to me by said LLMs but I first want to see how far I can go with pure LLM approach.

## 2025-10-16
### After Overnight Training
Checkpoint-4500
```
Validation Accuracy: 0.826
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.82      0.83      0.82       493
      signal       0.83      0.82      0.83       507

    accuracy                           0.83      1000
   macro avg       0.83      0.83      0.83      1000
weighted avg       0.83      0.83      0.83      1000
```

Checkpoint-4900
```
Validation Accuracy: 0.835
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.86      0.79      0.83       493
      signal       0.81      0.88      0.84       507

    accuracy                           0.83      1000
   macro avg       0.84      0.83      0.83      1000
weighted avg       0.84      0.83      0.83      1000
```

Diminishing returns confirmed. I will increase the learning rate slightly and see if it can push the accuracy further.

- I increased the learning rate from 3e-5 to 1e-4. This should help the model learn faster and hopefully push the model out of local minima.

Next idea could be to increase the parameters I offer in the prompt by doing some more numerical analysis on the root data.

Another idea is to extract logits at the output and give the probabilities as part of the output. This way I can see how confident the model is about its predictions. Also have a P score to compare with other models.

#### Increased learning rate
Model started as previous but after a point loss and gradient exploded. I paused the training at checkpoint-2300 and ran validation.
Checkpoint-2300
```
Validation Accuracy: 0.762
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.84      0.64      0.73       493
      signal       0.72      0.88      0.79       507

    accuracy                           0.76      1000
   macro avg       0.78      0.76      0.76      1000
weighted avg       0.78      0.76      0.76      1000
```

Smaller learning rate was better at this point. But maybe the model is trying to escape a local minima. I will let it run a bit more and see if it can recover.

## 2025-10-17
After overnight training
Checkpoint-7100
```
Validation Accuracy: 0.43
  background       0.82      0.05      0.10       493
      signal       0.73      0.89      0.80       507
     unknown       0.00      0.00      0.00         0

    accuracy                           0.48      1000
   macro avg       0.51      0.32      0.30      1000
weighted avg       0.77      0.48      0.46      1000
```
Catastrophic failure. Model completely failed to learn anything useful. A lot of whitespace predictions too.

Earlier checkpoints from same training session:
Checkpoint-3000
```
Validation Accuracy: 0.774
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.78      0.75      0.77       493
      signal       0.76      0.80      0.78       507

    accuracy                           0.77      1000
   macro avg       0.77      0.77      0.77      1000
weighted avg       0.77      0.77      0.77      1000
```

Checkpoint-4000
```
Validation Accuracy: 0.79
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.77      0.82      0.79       493
      signal       0.82      0.76      0.79       507

    accuracy                           0.79      1000
   macro avg       0.79      0.79      0.79      1000
weighted avg       0.79      0.79      0.79      1000
```

Checkpoint-5000
```
Validation Accuracy: 0.767
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.71      0.88      0.79       493
      signal       0.85      0.65      0.74       507

    accuracy                           0.77      1000
   macro avg       0.78      0.77      0.76      1000
weighted avg       0.78      0.77      0.76      1000
```

Checkpoint-6000
```
Validation Accuracy: 0.776
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.71      0.91      0.80       493
      signal       0.88      0.65      0.75       507

    accuracy                           0.78      1000
   macro avg       0.80      0.78      0.77      1000
weighted avg       0.80      0.78      0.77      1000
```

### Prepare longer prompt with more features
I decided to increase the number of features in the prompt. I calculated additional features.

I also started to use P_T (transverse momentum), psi, eta instead of px, py, pz. Since these might be more relevant for jet physics. I might consider adding lorentz invariants later on too.

First of all I had to increase the max_length from 256 to 512 in both training and validation scripts.

Per Jet Features:
- P_T: Transverse momentum
- eta: Pseudorapidity
- phi: Azimuthal angle
- m: Mass of the jet
- n_particles: Number of particles in the jet
- P_T_lead: Leading particle transverse momentum
- dR: Delta R between jets

Event Level Features:
- n_particles: Total number of particles in the event
- M_jj: Invariant mass of the bi-jet (or n-jet) system

I modified the learning rate back to 3e-5 and restarted the training with new prompt and features.
Also setup r=32 and lora_alpha=128, which should give more capacity to the LoRA layers.

Iterations per second of course decreased due to longer prompt.

But I am becoming more and more skeptical about the pure LLM approach. I think I will need to combine this with a more classical numerical model to get better results.

First results of the new prompt.
Checkpoint-200
```
Validation Accuracy: 0.722
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.90      0.49      0.64       494
      signal       0.66      0.94      0.77       506

    accuracy                           0.72      1000
   macro avg       0.78      0.72      0.71      1000
weighted avg       0.78      0.72      0.71      1000
```

I think this is a good candidate for overnight training. 

Example output:
```
<s>[INST] Classify this event as 'signal' or 'background'.
jets:
  jet1: P_T=1306.5740775925 eta=1.0441122764 phi=-0.4744111075 E=2124.1572287046 m=401.5012189251 n_particles=63 P_T_lead=253.7871246338
    dR_jet2=3.33
  jet2: P_T=1411.5709990887 eta=-0.2765506225 phi=2.7480957547 E=1468.9232156050 m=94.2792501634 n_particles=22 P_T_lead=354.7843017578
    dR_jet1=3.33
n_particles: 162 M_jj= 495.7804690885698[/INST]signal
```

Since dR_jet1 (angle between jet 1 to jet 2) and dR_jet2 (angle between jet 2 to jet 1) are the same value, it might be better to modify the prompt to only show dR once in the future iterations.

Before overnight training I tagged this version as v0.1.1 and ran until checkpoint-400.

Checkpoint-400
```
Validation Accuracy: 0.829
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.76      0.95      0.85       494
      signal       0.94      0.71      0.81       506

    accuracy                           0.83      1000
   macro avg       0.85      0.83      0.83      1000
weighted avg       0.85      0.83      0.83      1000
```

I would be excited but I did see 83% accuracy before and training is relatively slow with longer prompt. So I will let it train overnight and see if it can push the accuracy further.

I read more about putting a numberical MLP wrapper on top of the model. This could potentially help in capturing the complex relationships between the features more effectively. 

Also all logic behind using LLM was for natural language comprehension and explanations. So soon I will have to try to teach the model to explain its reasoning too.

Checkpoint-500
```
Validation Accuracy: 0.839
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.88      0.78      0.83       494
      signal       0.81      0.90      0.85       506

    accuracy                           0.84      1000
   macro avg       0.84      0.84      0.84      1000
weighted avg       0.84      0.84      0.84      1000
```

## 2025-10-18
### After Overnight Training
Checkpoint-2500
```
Validation Accuracy: 0.862
Number of unknown predictions: 0
              precision    recall  f1-score   support

  background       0.79      0.99      0.88       494
      signal       0.99      0.74      0.84       506

    accuracy                           0.86      1000
   macro avg       0.89      0.86      0.86      1000
weighted avg       0.89      0.86      0.86      1000
```

86% accuracy over 1:1 dataset is showing we are still hitting diminshing returns.

Then I wanted to see what will this model do with the original imbalanced dataset (1:10 signal to background ratio).

Checkpoint-2500 on imbalanced dataset
```
Number of correct background predictions: 901 out of 906
Number of correct signal predictions: 63 out of 94
Validation Accuracy: 0.964
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.97      0.99      0.98       906
      signal       0.93      0.67      0.78        94

    accuracy                           0.96      1000
   macro avg       0.95      0.83      0.88      1000
weighted avg       0.96      0.96      0.96      1000
```

This is a promising result. 96.4% accuracy on imbalanced dataset with 1:10 signal to background ratio. **BUT my guess is that that 1:10 ratio validation dataset did include too many events from 1:1 training dataset.** So model was able to memorize those events and give good results. I can't change this dataset at the moment since I would need to re-train the model from scratch.

Still we are observing a good recognition considering that the model is purely LLM based. Diminishing returns are still present but I will mark this as a progress.

Perhaps before doing any more changes, I should try my chances on the black-box datasets ?

### Black Box Dataset Testing
I prepared the black-box dataset in the same way as the R&D dataset. Jet clustering and feature extraction was done the same way. Because the challenge was done and master key was available, I was able to get the labels for validation.

Checkpoint-2500 on black-box dataset
```
Number of correct background predictions: 17581 out of 17979
Number of correct signal predictions: 6 out of 21
Validation Accuracy: 0.9770555555555556
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.98      0.99     17979
      signal       0.01      0.29      0.03        21

    accuracy                           0.98     18000
   macro avg       0.51      0.63      0.51     18000
weighted avg       1.00      0.98      0.99     18000
```

Eventhough accuracy is high (97.7%) due to imbalanced dataset, model was only able to identify 6 out of 21 signal events. This is not a good enough result, by comparison we know LHCO contenders were generally estimating more signals than there is, rather than the other way around.

Still it is good news to see the model did not collapse on signal or background only predictions and was able to perform both in 1:1 and 1:10 R&D datasets as well as black-box dataset. This indicates statistically significant learning has taken place.

Perhaps training with imbalanced dataset from start would yield better results on black-box dataset.

Because I enhanced the data processing and received statistically significant results, I tagged this version as v0.1.3. 

### New Data Processing
I modified the data processing to make sure there is no overlap between training and validation/test datasets even between balanced and inbalanced datasets.

I first create the original ratio dataset, then extract a 1:1 ratio dataset from it.

### Numeric Fusion Adapter
After some research, I created a numeric fusion adapter with some LLM help. I go line by line with API docs to understand what was recommended to me and it takes multiple attempts to make everything at least "run". So I am not sure about its parameters yet but it is a start. 

This is the numerical layer I was thinking about before. Appearently we can make the model learn numerical features by adding a small MLP layer on top of the LLM model. This method enhances the first token embeddings with numerical features during training.

It is recommended to use this adapter during validation/inference to improve performance on numerical tasks. But this defeats the purpose of LLM only approach for me. Perhaps in the future I can make a reasoning logic. So the model can prepare the data in a way where a simple external regex tool can extract the numerical features and finally push them in to the model to get more precise results.

But even without using the adapter during validation, I hope the model was *somehow* able to match the tokens with provided numerical values. I kept the validation script as same as it was (only textual prompt input) and I received the following results.

#### Checkpoint-300 with Numeric Fusion Adapter (no adapter during validation)
With original ratio dataset (1:10 signal to background)
```
Number of correct background predictions: 840 out of 918
Number of correct signal predictions: 59 out of 82
Validation Accuracy: 0.899
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.97      0.92      0.94       918
      signal       0.43      0.72      0.54        82

    accuracy                           0.90      1000
   macro avg       0.70      0.82      0.74      1000
weighted avg       0.93      0.90      0.91      1000
```

Eventhough training takes longer with the adapter, likely because my limited computational resources, results seems promising. With checkpoint 300 we are already reaching 89.9% accuracy on imbalanced dataset. 

I also ran the same model on 1:1 dataset. 

```
Number of correct background predictions: 464 out of 501
Number of correct signal predictions: 330 out of 499
Validation Accuracy: 0.794
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.73      0.93      0.82       501
      signal       0.90      0.66      0.76       499

    accuracy                           0.79      1000
   macro avg       0.82      0.79      0.79      1000
weighted avg       0.82      0.79      0.79      1000
```

This is surprising. I expected the model to perform better on 1:1 dataset but it is performing better on imbalanced dataset. At least on 1:1 dataset and at Checkpoint-300, accuracy seems same as previous attempts without the adapter.

Still, the day is over. So this setup is the new candidate for overnight training.

I will tag this version as v0.2.0 since it is a significant change in the training and data preparation process.

## 2025-10-19
### After Overnight Training with Numeric Fusion Adapter

#### Checkpoint-1600
##### 1:1 Dataset at 1000 samples
```
Number of correct background predictions: 452 out of 501
Number of correct signal predictions: 457 out of 499
Validation Accuracy: 0.909
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.91      0.90      0.91       501
      signal       0.90      0.92      0.91       499

    accuracy                           0.91      1000
   macro avg       0.91      0.91      0.91      1000
weighted avg       0.91      0.91      0.91      1000
```
This is a good improvement. 90.9% accuracy on 1:1 dataset with numeric fusion adapter.

I also ran the same model on imbalanced dataset. With 2000 validation samples (1:10 signal to background ratio).

##### 1:10 Dataset at 2000 samples
```
Number of correct background predictions: 1610 out of 1816
Number of correct signal predictions: 170 out of 184
Validation Accuracy: 0.89
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.99      0.89      0.94      1816
      signal       0.45      0.92      0.61       184

    accuracy                           0.89      2000
   macro avg       0.72      0.91      0.77      2000
weighted avg       0.94      0.89      0.91      2000
```

##### Black-box 1 Dataset on Original Ratio
```
Number of correct background predictions: 16744 out of 19982
Number of correct signal predictions: 14 out of 18
Validation Accuracy: 0.8379
WARNING: Number of unknown predictions: 1
              precision    recall  f1-score   support

  background       1.00      0.84      0.91     19982
      signal       0.00      0.78      0.01        18
     unknown       0.00      0.00      0.00         0

    accuracy                           0.84     20000
   macro avg       0.33      0.54      0.31     20000
weighted avg       1.00      0.84      0.91     20000
```
These results are promising. 
Model is able to generalize better with numeric fusion adapter eventhough we are not giving seperate numeric input. Previous blackbox test was only able to identify 6 out of 21 signal events. Now it is able to identify 14 out of 18 signal events.

I will tag this version as v0.2.1 before doing some changes to the validation script.

### Next Steps
- Modify validation script to use numeric fusion adapter during validation too. This should improve results further.

If numerical adapter during validation improves results significantly, next steps would be:
- Create a regex based numerical extractor to extract numerical features from the prompt.
- Use LLM to format the incoming data in to a format that can be parsed by the numerical extractor.

### A Crucial Mistake
As I was writing the new validation script, I realized I made a crucial mistake, I did not save the weights for the numeric fusion adapter during training. So I can't use the numeric inputs during validation. I have to re-train the model from scratch to save the adapter weights too.

I fixed the training and validation scripts to save and load the adapter weights respectively. I will start a new training run with the fixed scripts.

### Two Crucial Mistakes
As I was about to leave the system for overnight training, I realized that tokenize_example_refined function in training script was not embedding the numerical features at all. So the numeric fusion adapter was not receiving any numerical inputs during training. I am not sure when this mistake was introduced. From git blame it seems like since I added the numeric fusion adapter it never worked as intended. So all the results with numeric fusion adapter might be actually without numerical inputs. But I seem to remember like before yesterday's overnight training I fixed the training script to actually embed the numerical features. 

Current I am not sure about the timeline. But **it is safer to assume all results with numeric fusion adapter were just placebo so far.**

### Fixing the Training and Validation Scripts
I fixed the training script to actually embed the numerical features in to the token embeddings. I also fixed the validation script to load the adapter weights and use numerical features during validation.

Using float16 precision for numerical features caused some challenges. Also Transformers library had a "hidden" API again and I had to go search the source code to find out how to send numerical features to a collator successfully.

Collator should help with more efficient batching of data during training. Plus I added further normalization to the NumericFusionAdapter to keep the numerical features within similar ranges as token embeddings. This should prevent and mismatches between numerical and token embeddings.

This will be v0.2.2 since it is a significant fix. 

## 2025-10-20
### After Overnight Training with Fixed Numeric Fusion Adapter
I left model to train overnight, but we had a power failure in the area. Eventhough generator kicked in, because I didn't have a UPS, my system shutdown ungracefully. So training was interrupted and reached only checkpoint-1300. In previous attempts by the time model reached checkpoint-1300 it was already plateauing. But with fusion adapter logs indicated that model was still improving.


| Param         | Initial Value | Current Value | Comment
|---------------|---------------|---------------|------------------------------
| Loss          | 19.909        | 14.6236       | Decreasing, shows learning
| Grad Norm     | 34.41         | 9.92          | Decreasing, shows stability improving
| Learning Rate | 2.998e-5      | 2.992e-5      | Slightly decreasing, normal behavior

So I think I will keep running the training from checkpoint-1300. But first I ran validation to see where we are at.

#### Checkpoint-1300
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 504 out of 524
Number of correct signal predictions: 380 out of 476
Validation Accuracy: 0.884
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.84      0.96      0.90       524
      signal       0.95      0.80      0.87       476

    accuracy                           0.88      1000
   macro avg       0.90      0.88      0.88      1000
weighted avg       0.89      0.88      0.88      1000
```
##### 1:10 Dataset at 8000 samples & numeric input enabled 
```
Number of correct background predictions: 6989 out of 7250
Number of correct signal predictions: 605 out of 750
Validation Accuracy: 0.94925
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.98      0.96      0.97      7250
      signal       0.70      0.81      0.75       750

    accuracy                           0.95      8000
   macro avg       0.84      0.89      0.86      8000
weighted avg       0.95      0.95      0.95      8000
```

##### 1:10 Dataset at 8000 samples & numeric input DISABLED
```
Number of correct background predictions: 6972 out of 7250
Number of correct signal predictions: 609 out of 750
Validation Accuracy: 0.947625
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.98      0.96      0.97      7250
      signal       0.69      0.81      0.74       750

    accuracy                           0.95      8000
   macro avg       0.83      0.89      0.86      8000
weighted avg       0.95      0.95      0.95      8000
```

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 7517 out of 7989
Number of correct signal predictions: 6 out of 11
Validation Accuracy: 0.940375
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.94      0.97      7989
      signal       0.01      0.55      0.02        11

    accuracy                           0.94      8000
   macro avg       0.51      0.74      0.50      8000
weighted avg       1.00      0.94      0.97      8000
```
Missing almost half of the signal events, but there are only 11 of them.

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input DISABLED
```
Number of correct background predictions: 7429 out of 7989
Number of correct signal predictions: 7 out of 11
Validation Accuracy: 0.9295
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.93      0.96      7989
      signal       0.01      0.64      0.02        11

    accuracy                           0.93      8000
   macro avg       0.51      0.78      0.49      8000
weighted avg       1.00      0.93      0.96      8000
```

Microscopically better results with numeric input enabled. But overall results seems similar.

I will keep training from checkpoint-1300 and see if we can push the accuracy further. I think the numeric fusion adapter needs to stabilize before showing its true potential or lack thereof.

#### Checkpoint-2200
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 490 out of 524
Number of correct signal predictions: 412 out of 476
Validation Accuracy: 0.902
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.88      0.94      0.91       524
      signal       0.92      0.87      0.89       476

    accuracy                           0.90      1000
   macro avg       0.90      0.90      0.90      1000
weighted avg       0.90      0.90      0.90      1000
```

##### 1:1 Dataset at 1000 samples & numeric input DISABLED
```
Number of correct background predictions: 494 out of 524
Number of correct signal predictions: 413 out of 476
Validation Accuracy: 0.907
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.89      0.94      0.91       524
      signal       0.93      0.87      0.90       476

    accuracy                           0.91      1000
   macro avg       0.91      0.91      0.91      1000
weighted avg       0.91      0.91      0.91      1000
```
Better results without numeric input.

##### 1:10 Dataset at 8000 samples & numeric input enabled 
```
Number of correct background predictions: 6798 out of 7250
Number of correct signal predictions: 647 out of 750
Validation Accuracy: 0.930625
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.99      0.94      0.96      7250
      signal       0.59      0.86      0.70       750

    accuracy                           0.93      8000
   macro avg       0.79      0.90      0.83      8000
weighted avg       0.95      0.93      0.94      8000
```

##### 1:10 Dataset at 8000 samples & numeric input DISABLED
```
Number of correct background predictions: 6794 out of 7250
Number of correct signal predictions: 653 out of 750
Validation Accuracy: 0.930875
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.99      0.94      0.96      7250
      signal       0.59      0.87      0.70       750

    accuracy                           0.93      8000
   macro avg       0.79      0.90      0.83      8000
weighted avg       0.95      0.93      0.94      8000
```
Slightly better results without numeric input. But negligible difference.

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 7166 out of 7989
Number of correct signal predictions: 7 out of 11
Validation Accuracy: 0.896625
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.90      0.95      7989
      signal       0.01      0.64      0.02        11

    accuracy                           0.90      8000
   macro avg       0.50      0.77      0.48      8000
weighted avg       1.00      0.90      0.94      8000
```

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input DISABLED
```
Number of correct background predictions: 7165 out of 7989
Number of correct signal predictions: 7 out of 11
Validation Accuracy: 0.8965
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.90      0.95      7989
      signal       0.01      0.64      0.02        11

    accuracy                           0.90      8000
   macro avg       0.50      0.77      0.48      8000
weighted avg       1.00      0.90      0.94      8000
```
No difference with numeric input disabled.

### Observations
With numeric fusion adapter, at least on the validation, numeric input does not seem to improve results. In fact in some cases disabling numeric input yields slightly better results. This might be a failure in my implementation or simply text input does the job well enough.
Maybe numerics fusion adapter needs more training to show its true potential. I will keep training tonight since I don't have any time left to change the code today.

One idea for tomorrow could be to try to create a custom token and tie fusion adapter to that. So we are not messing with the first token embedding, but rather creating our own token that is dedicated to these numerical features. I think this type of an implementation would be cleaner and let us spot any issues more easily. Also it could make the model more flexible, since we can choose to include or exclude the custom token during training and validation.

Final version of this iteration was tagged as v0.2.3.

#### A last minute change
Just before starting the training, I decided to modify the numeric adapter to use 2 projection layers with SiLU activation in between instead of simply projecting one linear layer. For this I removed the previous scale and tanh normalization since SiLU should be enough.

## 2025-10-21
### After Overnight Training with Modified Numeric Fusion Adapter
#### Checkpoint-1500
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 518 out of 524
Number of correct signal predictions: 335 out of 476
Validation Accuracy: 0.853
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.79      0.99      0.88       524
      signal       0.98      0.70      0.82       476

    accuracy                           0.85      1000
   macro avg       0.88      0.85      0.85      1000
weighted avg       0.88      0.85      0.85      1000
```
##### 1:1 Dataset at 1000 samples & numeric input DISABLED
```
Number of correct background predictions: 517 out of 524
Number of correct signal predictions: 333 out of 476
Validation Accuracy: 0.85
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.78      0.99      0.87       524
      signal       0.98      0.70      0.82       476

    accuracy                           0.85      1000
   macro avg       0.88      0.84      0.84      1000
weighted avg       0.88      0.85      0.85      1000
```
##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 7127 out of 7250
Number of correct signal predictions: 527 out of 750
Validation Accuracy: 0.95675
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.97      0.98      0.98      7250
      signal       0.81      0.70      0.75       750

    accuracy                           0.96      8000
   macro avg       0.89      0.84      0.86      8000
weighted avg       0.95      0.96      0.96      8000
```
##### 1:10 Dataset at 8000 samples & numeric input DISABLED
```
Number of correct background predictions: 7127 out of 7250
Number of correct signal predictions: 530 out of 750
Validation Accuracy: 0.957125
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.97      0.98      0.98      7250
      signal       0.81      0.71      0.76       750

    accuracy                           0.96      8000
   macro avg       0.89      0.84      0.87      8000
weighted avg       0.96      0.96      0.96      8000
```

Negligible differences again.

### Found a bug ! 
I've noticed I am not feeding numerical features properly to the loss function during training. So model is not learning from numerical features at all.

I have fixed the training script to properly feed numerical features to the loss function. Now it is time to restart the training from scratch.

### After bugfix
#### Checkpoint-400
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 487 out of 524
Number of correct signal predictions: 356 out of 476
Validation Accuracy: 0.843
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.80      0.93      0.86       524
      signal       0.91      0.75      0.82       476

    accuracy                           0.84      1000
   macro avg       0.85      0.84      0.84      1000
weighted avg       0.85      0.84      0.84      1000
```
##### 1:1 Dataset at 1000 samples & numeric input DISABLED
```
Number of correct background predictions: 71 out of 524
Number of correct signal predictions: 475 out of 476
Validation Accuracy: 0.546
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.99      0.14      0.24       524
      signal       0.51      1.00      0.68       476

    accuracy                           0.55      1000
   macro avg       0.75      0.57      0.46      1000
weighted avg       0.76      0.55      0.45      1000
```

Finally ! We are seeing a solid difference with numeric fusion adapter enabled. Even on an early checkpoint like 400. 

Note that since now we correctly train with the fusion adapter, we might be making word-only training worse by disabling numeric input. We will need to compare 2 setups at later checkpoints to see the real difference.

## 2025-10-22
#### Checkpoint 2400
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 466 out of 524
Number of correct signal predictions: 431 out of 476
Validation Accuracy: 0.897
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.91      0.89      0.90       524
      signal       0.88      0.91      0.89       476

    accuracy                           0.90      1000
   macro avg       0.90      0.90      0.90      1000
weighted avg       0.90      0.90      0.90      1000
```

##### 1:1 Dataset at 1000 samples & numeric input DISABLED
```
Number of correct background predictions: 46 out of 524
Number of correct signal predictions: 475 out of 476
Validation Accuracy: 0.521
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.98      0.09      0.16       524
      signal       0.50      1.00      0.66       476

    accuracy                           0.52      1000
   macro avg       0.74      0.54      0.41      1000
weighted avg       0.75      0.52      0.40      1000
```

##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 6395 out of 7250
Number of correct signal predictions: 680 out of 750
Validation Accuracy: 0.884375
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.99      0.88      0.93      7250
      signal       0.44      0.91      0.60       750

    accuracy                           0.88      8000
   macro avg       0.72      0.89      0.76      8000
weighted avg       0.94      0.88      0.90      8000
```

##### 1:10 Dataset at 8000 samples & numeric input DISABLED
```
Number of correct background predictions: 741 out of 7250
Number of correct signal predictions: 749 out of 750
Validation Accuracy: 0.18625
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.10      0.19      7250
      signal       0.10      1.00      0.19       750

    accuracy                           0.19      8000
   macro avg       0.55      0.55      0.19      8000
weighted avg       0.91      0.19      0.19      8000
```

Huge difference again. Model is completely failing without numeric input. But is it more successful than previous attempts without numeric fusion adapter ?

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 6440 out of 7989
Number of correct signal predictions: 9 out of 11
Validation Accuracy: 0.806125
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.81      0.89      7989
      signal       0.01      0.82      0.01        11

    accuracy                           0.81      8000
   macro avg       0.50      0.81      0.45      8000
weighted avg       1.00      0.81      0.89      8000
```

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input DISABLED
```
Number of correct background predictions: 847 out of 7989
Number of correct signal predictions: 11 out of 11
Validation Accuracy: 0.10725
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.11      0.19      7989
      signal       0.00      1.00      0.00        11

    accuracy                           0.11      8000
   macro avg       0.50      0.55      0.10      8000
weighted avg       1.00      0.11      0.19      8000
```

### Comparisons
Up to bugfix (2025-10-21) we can assume models were not trained with numeric features at all. So we can compare the results before and after bugfix to see the impact of numeric fusion adapter.

We will use numeric adapter disabled results from 2025-10-21 as the baseline for comparison.
Precision is calculated as:

Date       | Dataset            | Checkpoint | Numeric Input | Accuracy | Background Precision | Signal Precision | F1 Score Signal | F1 Score Background
-----------|--------------------|------------|---------------|----------|----------------------|------------------|-----------------|-------------------
2025-10-20 | 1:1 @1000          | 2200       | Disabled      | 0.907    | 0.89                 | 0.93             | 0.90            | 0.91
2025-10-21 | 1:1 @1000          | 1500       | Disabled      | 0.85     | 0.78                 | 0.98             | 0.82            | 0.87
2025-10-22 | 1:1 @1000          | 2400       | Enabled       | 0.897    | 0.91                 | 0.88             | 0.89            | 0.90
2025-10-20 | 1:10 @8000         | 2200       | Disabled      | 0.930875 | 0.99                 | 0.59             | 0.87            | 0.96
2025-10-21 | 1:10 @8000         | 1500       | Disabled      | 0.957125 | 0.97                 | 0.81             | 0.76            | 0.98
2025-10-22 | 1:10 @8000         | 2400       | Enabled       | 0.884375 | 0.99                 | 0.44             | 0.60            | 0.93
2025-10-20 | Black-box 1 @8000  | 2200       | Disabled      | 0.8965   | 1.00                 | 0.01             | 0.02            | 0.95
2025-10-22 | Black-box 1 @8000  | 2400       | Enabled       | 0.806125 | 1.00                 | 0.01             | 0.01            | 0.89

### Observations
- On 1:1 dataset, numeric fusion adapter enabled model is performing similarly to the baseline model without numeric input. Slightly worse accuracy but better balance between background and signal precision.
- On 1:10 dataset, numeric fusion adapter enabled model is performing worse than the baseline model without numeric input. Significant drop in accuracy and signal precision.
- On Black-box dataset, numeric fusion adapter enabled model is performing worse than the baseline model without numeric input. Significant drop in accuracy.

So far, numeric fusion adapter does not seem to be helping the model to generalize better. In fact it seems to be hurting the performance on imbalanced and black-box datasets.

### Next Steps
- Continue training the model with numeric fusion adapter to see if performance improves with more training.
- Perhaps switch training to imbalanced dataset and see if that helps the model to generalize better on black-box dataset.
- Investigate alternative methods to incorporate numerical features in to the model.

## 2025-10-23
### After 2nd day of training
#### Checkpoint-5300
##### 1:1 Dataset at 1000 samples & numeric input enabled
Number of correct background predictions: 493 out of 524
Number of correct signal predictions: 405 out of 476
Validation Accuracy: 0.898
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.87      0.94      0.91       524
      signal       0.93      0.85      0.89       476

    accuracy                           0.90      1000
   macro avg       0.90      0.90      0.90      1000
weighted avg       0.90      0.90      0.90      1000

##### 1:1 Dataset at 1000 samples & numeric input DISABLED
Number of correct background predictions: 91 out of 524
Number of correct signal predictions: 469 out of 476
Validation Accuracy: 0.56
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.93      0.17      0.29       524
      signal       0.52      0.99      0.68       476

    accuracy                           0.56      1000
   macro avg       0.72      0.58      0.49      1000
weighted avg       0.73      0.56      0.48      1000


It is obvious that with numeric input disabled model is failing completely. So I will use numeric input enabled for rest of the tests.

##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 6797 out of 7250
Number of correct signal predictions: 635 out of 750
Validation Accuracy: 0.929
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.98      0.94      0.96      7250
      signal       0.58      0.85      0.69       750

    accuracy                           0.93      8000
   macro avg       0.78      0.89      0.83      8000
weighted avg       0.95      0.93      0.93      8000
```

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 7141 out of 7989
Number of correct signal predictions: 8 out of 11
Validation Accuracy: 0.893625
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.89      0.94      7989
      signal       0.01      0.73      0.02        11

    accuracy                           0.89      8000
   macro avg       0.50      0.81      0.48      8000
weighted avg       1.00      0.89      0.94      8000
```
## 2025-10-24
### After 3rd day of training
#### Checkpoint-7700
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 465 out of 524
Number of correct signal predictions: 432 out of 476
Validation Accuracy: 0.897
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.91      0.89      0.90       524
      signal       0.88      0.91      0.89       476

    accuracy                           0.90      1000
   macro avg       0.90      0.90      0.90      1000
weighted avg       0.90      0.90      0.90      1000
```
##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 6427 out of 7250
Number of correct signal predictions: 693 out of 750
Validation Accuracy: 0.89
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.99      0.89      0.94      7250
      signal       0.46      0.92      0.61       750

    accuracy                           0.89      8000
   macro avg       0.72      0.91      0.77      8000
weighted avg       0.94      0.89      0.91      8000
```

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 6589 out of 7989
Number of correct signal predictions: 9 out of 11
Validation Accuracy: 0.82475
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.82      0.90      7989
      signal       0.01      0.82      0.01        11

    accuracy                           0.82      8000
   macro avg       0.50      0.82      0.46      8000
weighted avg       1.00      0.82      0.90      8000
```

I will continue training for one more night, but I will change the training dataset to imbalanced 1:10 dataset. Hopefully this will help the model to generalize better on black-box dataset.

## 2025-10-25
### Switching to imbalanced dataset for training
I have modified the training script to use imbalanced 1:10 dataset for training. I will continue training from checkpoint-7700.

#### Checkpoint-9800
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 514 out of 524
Number of correct signal predictions: 361 out of 476
Validation Accuracy: 0.875
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.82      0.98      0.89       524
      signal       0.97      0.76      0.85       476

    accuracy                           0.88      1000
   macro avg       0.90      0.87      0.87      1000
weighted avg       0.89      0.88      0.87      1000
```

##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 7091 out of 7250
Number of correct signal predictions: 573 out of 750
Validation Accuracy: 0.958
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.98      0.98      0.98      7250
      signal       0.78      0.76      0.77       750

    accuracy                           0.96      8000
   macro avg       0.88      0.87      0.88      8000
weighted avg       0.96      0.96      0.96      8000
```

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 7692 out of 7989
Number of correct signal predictions: 5 out of 11
Validation Accuracy: 0.962125
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.96      0.98      7989
      signal       0.02      0.45      0.03        11

    accuracy                           0.96      8000
   macro avg       0.51      0.71      0.51      8000
weighted avg       1.00      0.96      0.98      8000
```
## 2025-10-26
### After training on imbalanced dataset
#### Checkpoint-11800
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 517 out of 524
Number of correct signal predictions: 325 out of 476
Validation Accuracy: 0.842
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.77      0.99      0.87       524
      signal       0.98      0.68      0.80       476

    accuracy                           0.84      1000
   macro avg       0.88      0.83      0.84      1000
weighted avg       0.87      0.84      0.84      1000
```

##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 7162 out of 7250
Number of correct signal predictions: 533 out of 750
Validation Accuracy: 0.961875
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.97      0.99      0.98      7250
      signal       0.86      0.71      0.78       750

    accuracy                           0.96      8000
   macro avg       0.91      0.85      0.88      8000
weighted avg       0.96      0.96      0.96      8000
```

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 7806 out of 7989
Number of correct signal predictions: 6 out of 11
Validation Accuracy: 0.9765
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.98      0.99      7989
      signal       0.03      0.55      0.06        11

    accuracy                           0.98      8000
   macro avg       0.52      0.76      0.52      8000
weighted avg       1.00      0.98      0.99      8000
```

## 2025-10-27
#### Checkpoint-17300
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 519 out of 524
Number of correct signal predictions: 315 out of 476
Validation Accuracy: 0.834
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.76      0.99      0.86       524
      signal       0.98      0.66      0.79       476

    accuracy                           0.83      1000
   macro avg       0.87      0.83      0.83      1000
weighted avg       0.87      0.83      0.83      1000
```

##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 7168 out of 7250
Number of correct signal predictions: 522 out of 750
Validation Accuracy: 0.96125
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.97      0.99      0.98      7250
      signal       0.86      0.70      0.77       750

    accuracy                           0.96      8000
   macro avg       0.92      0.84      0.87      8000
weighted avg       0.96      0.96      0.96      8000
```

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 7809 out of 7989
Number of correct signal predictions: 3 out of 11
Validation Accuracy: 0.9765
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.98      0.99      7989
      signal       0.02      0.27      0.03        11

    accuracy                           0.98      8000
   macro avg       0.51      0.63      0.51      8000
weighted avg       1.00      0.98      0.99      8000
```

False signal predictions are decreasing, but we started to lose true signal predictions as well. Model seems to be collapsing to mostly predicting background.

### A Change
I've changed NFA from float16 to float32. Probably I was mistaken to not set this up earlier. But somehow I thought f32 would be incompatible with the quantized model. Hopefully higher precision will help the model to learn better.

I will continue training overnight with 1:10 ratio and see the results tomorrow.

## 2025-10-28
### After training with float32 NFA overnight
Checkpoints increased only by 1400, float32 must be slowing down the training or the system slept during the night. I will diagnose it further if needed.

#### Checkpoint-18700
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 520 out of 524
Number of correct signal predictions: 315 out of 476
Validation Accuracy: 0.835
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.76      0.99      0.86       524
      signal       0.99      0.66      0.79       476

    accuracy                           0.83      1000
   macro avg       0.88      0.83      0.83      1000
weighted avg       0.87      0.83      0.83      1000
```
##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 7181 out of 7250
Number of correct signal predictions: 512 out of 750
Validation Accuracy: 0.961625
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.97      0.99      0.98      7250
      signal       0.88      0.68      0.77       750

    accuracy                           0.96      8000
   macro avg       0.92      0.84      0.87      8000
weighted avg       0.96      0.96      0.96      8000
```
##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 7833 out of 7989
Number of correct signal predictions: 3 out of 11
Validation Accuracy: 0.9795
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.98      0.99      7989
      signal       0.02      0.27      0.04        11

    accuracy                           0.98      8000
   macro avg       0.51      0.63      0.51      8000
weighted avg       1.00      0.98      0.99      8000
```

### Seperating Numeric Feature Optimization From LoRA
I think reaching results of Checkpoint-18700 was an overall improvement compared to previous attempts. We were able to reduce the wrong signal predictions but it came at the cost of losing true signal predictions as well. So eventhough rough accuracy improved, model is still not very useful for our task. 

I seperated the optimizer parameters (AdamW8Bit) for NFA and LoRA. Now NFA uses 32 bit option with a higher learning rate of 3e-3 while LoRA optimizer is kept at 3e-5. Additionally, I added another layer of SiLU + Linear projection to the NFA module to increase its capacity. With complex exponential relationships in the numerical features, I think NFA needs more capacity to model them properly. Hopefully this will help the NFA to converge better without disturbing the LoRA training.

I also added a few debug prints to monitor the NFA weights during training. Which proved useful because AdamW8Bit was not updating the NFA weights at all but giving no errors or warnings. After some investigation I found out that I had to explicitly set the NFA parameters to require gradients and use 32-bit optimization. But without proper debugging I would stay clueless for a long time.

Finally, I started experimenting with tensorboard to monitor the training process better. 

#### Restarting Training
Sadly these changes meant I have to restart the training from scratch.

I started training again but returned to 1:1 dataset. I can compare the results better this way with previous attempts. Plus I think using unbalanced dataset early on might collapse the model to predicting mostly background.

## 2025-10-29
### After restarting training with seperated optimizers
#### Checkpoint-2200
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 498 out of 524
Number of correct signal predictions: 397 out of 476
Validation Accuracy: 0.895
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.86      0.95      0.90       524
      signal       0.94      0.83      0.88       476

    accuracy                           0.90      1000
   macro avg       0.90      0.89      0.89      1000
weighted avg       0.90      0.90      0.89      1000
```
##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 6928 out of 7250
Number of correct signal predictions: 628 out of 750
Validation Accuracy: 0.9445
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.98      0.96      0.97      7250
      signal       0.66      0.84      0.74       750

    accuracy                           0.94      8000
   macro avg       0.82      0.90      0.85      8000
weighted avg       0.95      0.94      0.95      8000
```

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 7398 out of 7989
Number of correct signal predictions: 7 out of 11
Validation Accuracy: 0.925625
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.93      0.96      7989
      signal       0.01      0.64      0.02        11

    accuracy                           0.93      8000
   macro avg       0.51      0.78      0.49      8000
weighted avg       1.00      0.93      0.96      8000
```

### Comparison with Float16 NFA Training
Checkpoint         | Verification Dataset | NFA Type   | Accuracy | Background Precision | Signal Precision  | F1 Score Background | F1 Score Signal
-------------------|----------------------|------------|----------|----------------------|-------------------|---------------------|-------------------
2025-10-22 - 2400  | 1:1 @1000            | Float16    | 0.897    | 0.91                 | 0.88              | 0.90                | 0.89
2025-10-29 - 2200  | 1:1 @1000            | Float32    | 0.895    | 0.86                 | 0.94              | 0.90                | 0.88 
2025-10-22 - 2400  | 1:10 @8000           | Float16    | 0.884    | 0.99                 | 0.44              | 0.93                | 0.60
2025-10-29 - 2200  | 1:10 @8000           | Float32    | 0.945    | 0.98                 | 0.66              | 0.97                | 0.74
2025-10-22 - 2400  | Black-box 1 @8000    | Float16    | 0.806    | 1.00                 | 0.01              | 0.89                | 0.01
2025-10-29 - 2200  | Black-box 1 @8000    | Float32    | 0.926    | 1.00                 | 0.01              | 0.96                | 0.02

/// caption
All results with numeric input enabled and LoRA + NFA training with 1:1 dataset.
///


#### Observations
- On 1:1 dataset, float32 NFA training shows similar accuracy but with a trade-off between background and signal precision.
- On 1:10 dataset, float32 NFA training shows significant improvement in accuracy, background precision, signal precision, and F1 scores for both classes.
- On Black-box dataset, float32 NFA training shows significant improvement in accuracy and F1 score for background class, while signal precision remains low.

### Parameters Sanity Check
I checked the NFA weights and noticed that eventhough they are getting gradients their weights were constant. So the adapter was providing support to the model but not learning anything new. This was due ordering of the operations in the training script. LoRA being loaded after NFA was causing the optimizer to not update NFA weights properly. I have fixed this issue and restarted the training again.

I'm noticing that watching training process with multiple debug metrics via tensorboard is very useful. I can see the losses and weights changing live and notice issues like this much faster. Without this monitoring, since there was a statistically significant improvement in results, I would not have suspected an issue with NFA weights.

## 2025-10-30
### After fixing NFA weight update issue
#### Checkpoint-2200
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 487 out of 524
Number of correct signal predictions: 396 out of 476
Validation Accuracy: 0.883
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.86      0.93      0.89       524
      signal       0.91      0.83      0.87       476

    accuracy                           0.88      1000
   macro avg       0.89      0.88      0.88      1000
weighted avg       0.89      0.88      0.88      1000
```

##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 6808 out of 7250
Number of correct signal predictions: 628 out of 750
Validation Accuracy: 0.9295
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.98      0.94      0.96      7250
      signal       0.59      0.84      0.69       750

    accuracy                           0.93      8000
   macro avg       0.78      0.89      0.83      8000
weighted avg       0.95      0.93      0.93      8000
```

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 7129 out of 7989
Number of correct signal predictions: 9 out of 11
Validation Accuracy: 0.89225
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.89      0.94      7989
      signal       0.01      0.82      0.02        11

    accuracy                           0.89      8000
   macro avg       0.51      0.86      0.48      8000
weighted avg       1.00      0.89      0.94      8000
```

#### Checkpoint-4600
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 463 out of 524
Number of correct signal predictions: 435 out of 476
Validation Accuracy: 0.898
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.92      0.88      0.90       524
      signal       0.88      0.91      0.90       476

    accuracy                           0.90      1000
   macro avg       0.90      0.90      0.90      1000
weighted avg       0.90      0.90      0.90      1000
```

##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 6408 out of 7250
Number of correct signal predictions: 691 out of 750
Validation Accuracy: 0.887375
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.99      0.88      0.93      7250
      signal       0.45      0.92      0.61       750

    accuracy                           0.89      8000
   macro avg       0.72      0.90      0.77      8000
weighted avg       0.94      0.89      0.90      8000
```

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 6462 out of 7989
Number of correct signal predictions: 9 out of 11
Validation Accuracy: 0.808875
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.81      0.89      7989
      signal       0.01      0.82      0.01        11

    accuracy                           0.81      8000
   macro avg       0.50      0.81      0.45      8000
weighted avg       1.00      0.81      0.89      8000
```

Results are dissapointing. NFA weights are changing now but model performance is degraded compared to previous validation run at Checkpoint-2200. Previous attempt before fixing NFA weight update issue was performing better also. 

I will keep the training but reducing learning rate for NFA to 3e-4. Loss was still decreasing but with a lower rate now, so I hope this will lead to better results. If not, I will switch to imbalanced dataset for rest of the training run.

## 2025-10-31
### After lowering NFA learning rate & overnight training
#### Checkpoint-6900
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 441 out of 524
Number of correct signal predictions: 445 out of 476
Validation Accuracy: 0.886
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.93      0.84      0.89       524
      signal       0.84      0.93      0.89       476

    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```

##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 6101 out of 7250
Number of correct signal predictions: 713 out of 750
Validation Accuracy: 0.85175
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.99      0.84      0.91      7250
      signal       0.38      0.95      0.55       750

    accuracy                           0.85      8000
   macro avg       0.69      0.90      0.73      8000
weighted avg       0.94      0.85      0.88      8000
```

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 6070 out of 7989
Number of correct signal predictions: 9 out of 11
Validation Accuracy: 0.759875
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.76      0.86      7989
      signal       0.00      0.82      0.01        11

    accuracy                           0.76      8000
   macro avg       0.50      0.79      0.44      8000
weighted avg       1.00      0.76      0.86      8000
```

### Summary of Results So Far
On checkpoint-6900, model performance has degraded further compared to previous checkpoints.

I changed training dataset to imbalanced 1:10 ratio to see if that helps the model to generalize better on black-box dataset.

### After Further Training on Imbalanced Dataset
#### Checkpoint-8400
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 517 out of 524
Number of correct signal predictions: 355 out of 476
Validation Accuracy: 0.872
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.81      0.99      0.89       524
      signal       0.98      0.75      0.85       476

    accuracy                           0.87      1000
   macro avg       0.90      0.87      0.87      1000
weighted avg       0.89      0.87      0.87      1000
```

Slightly better results than checkpoint-6900. But this seems like because background predictions increased. Signal predictions are even lower than before.

##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 7135 out of 7250
Number of correct signal predictions: 566 out of 750
Validation Accuracy: 0.962625
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.97      0.98      0.98      7250
      signal       0.83      0.75      0.79       750

    accuracy                           0.96      8000
   macro avg       0.90      0.87      0.89      8000
weighted avg       0.96      0.96      0.96      8000
```

Looks like an improvement over the surface. Again model is favoring background predictions more, so the wrong signal predictions are reduced but true signal predictions are lost as well. This effects the scores positively but it is expected since we trained on imbalanced dataset. As the training continues, it might end up collapsing to predicting only background.

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 7747 out of 7989
Number of correct signal predictions: 4 out of 11
Number of correct unknown predictions: 0 out of 0
Validation Accuracy: 0.968875
WARNING: Number of unknown predictions: 1
              precision    recall  f1-score   support

  background       1.00      0.97      0.98      7989
      signal       0.02      0.36      0.03        11
     unknown       0.00      0.00      0.00         0

    accuracy                           0.97      8000
   macro avg       0.34      0.44      0.34      8000
weighted avg       1.00      0.97      0.98      8000
```
Here too model is favoring background predictions heavily. So false signal predictions are reduced but true signal predictions are also lower than before.

I will continue training overnight and see if the model improves. I might try to change the loss function or NFA architecture if results do not improve further.

## 2025-11-01
### After Further Training on Imbalanced Dataset Overnight
#### Checkpoint-10200
##### 1:1 Dataset at 1000 samples & numeric input enabled
```
Number of correct background predictions: 517 out of 524
Number of correct signal predictions: 319 out of 476
Validation Accuracy: 0.836
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.77      0.99      0.86       524
      signal       0.98      0.67      0.80       476

    accuracy                           0.84      1000
   macro avg       0.87      0.83      0.83      1000
weighted avg       0.87      0.84      0.83      1000

SIC (Significance Improvement Characteristic): 5.7983
```

##### 1:10 Dataset at 8000 samples & numeric input enabled
```
Number of correct background predictions: 7174 out of 7250
Number of correct signal predictions: 528 out of 750
Validation Accuracy: 0.96275
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       0.97      0.99      0.98      7250
      signal       0.87      0.70      0.78       750

    accuracy                           0.96      8000
   macro avg       0.92      0.85      0.88      8000
weighted avg       0.96      0.96      0.96      8000

SIC (Significance Improvement Characteristic): 6.8760
```

##### Black-box 1 Dataset at 8000 samples on original ratio & numeric input enabled
```
Number of correct background predictions: 7832 out of 7989
Number of correct signal predictions: 2 out of 11
Validation Accuracy: 0.97925
All predictions classified as 'signal' or 'background'.
              precision    recall  f1-score   support

  background       1.00      0.98      0.99      7989
      signal       0.01      0.18      0.02        11

    accuracy                           0.98      8000
   macro avg       0.51      0.58      0.51      8000
weighted avg       1.00      0.98      0.99      8000

SIC (Significance Improvement Characteristic): 1.2970
```

With these I also added SIC (Significance Improvement Characteristic) so I can compare the model performance better.[^5] SIC is calculated as:
$$
SIC = \frac{\epsilon_S}{\sqrt{\epsilon_B}}
$$

Where the signal efficiency is defined as:
$$
\epsilon_S = \frac{N_{correct\_{signal}}}{N_{total\_{signal}}}
$$

and background efficiency defined as:
$$
\epsilon_B = \frac{N_{selected\_{background}}}{N_{total\_{background}}}
$$

Notice selected background is the number of background events that are incorrectly classified as signal. This metric gives an idea of how well the model is able to distinguish signal from background, taking into account both true positive rate and false positive rate.


[^1]: [LHC Olympics 2020 Homepage](https://lhco2020.github.io/homepage/)
[^2]: [R&D Dataset](https://zenodo.org/records/4536377)
[^3]: [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
[^4]: [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/index)
[^5]: [Agents of Discovery](https://arxiv.org/abs/2509.08535)