# Study Diary

## Initial Research
I learnt about LHC Olympics, LoRA, HuggingFace Transformers, and other related topics. I also had a few crash-course sessions about collision events, jet clustering, and other related physics topics. I noticed that I know almost nothing about these topics. But I was able to find some common practices and examples to follow.

## 2025-10-14
I am starting the study diary here. First day of the log is actually the 3rd attempt to have a go at this. This time rather than focusing on training first, I am focusing on understanding the data first.

### Understanding the Dataset
I was able to successfully unpack the R&D dataset. It was obvious that the data needed some sort of pre-processing before it can be used in training.

Data consisted of a variable number of particles per event (up to 2100). Each particle had 3 features pt, eta, phi. By observing the common practice again I set required mass parameter to 0. So assumed massless particles.

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
I use `mistralai/Mistral-7B-Instruct-v0.3` model as my initial base model. Mainly because it is a well known open weight model and it is relatively small (7B parameters).

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