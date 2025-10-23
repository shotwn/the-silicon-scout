# Interim Report 1
2025-10-23

## What is the goal of the study?
The goal is to fine-tune one or more large language models to help with physics discovery. They should be able to ingest a specialized physics knowledge and create statistically significant results for the users. Using the LLM for this task should allow the model to:
- Accept input data in natural language form in various formats
- Discuss it with itself or other models
- Do multi-step reasoning
- Prepare it to be processed by external tools
- Give back results in natural language form

## Proof of concept
For proof of concept we selected Large Hadron Collider Olympics 2020 (LHCO) challenges. These challenges were designed with machine learning in mind and has established simulated data. Using simulated LHCO data we can create the framework to train an LLM, then we can create our own unique simulated data and make sure framework is working as intended with a completely unique dataset. 

## What Did The Study Accomplished So Far?
### Data Preparation
First challenge was to unpack the LHCO R&D data. Which was pretty straightforward but required some multiproccessing work to handle big batches in a quick manner. 

I found in LHCO examples that we can use jet clustering to reduce the data while keeping most of the information intact. This allowed me to reduce the data size significantly. I ran fastjet clustering with anti-kt algorithm using R=1.0. I extracted inclusive jets with pT > 1.2 GeV as given in the LHCO documentation. After clustering I created jsonl files with event data. 

Each event data contains:
```json
{
  "type": "signal", 
  "jets": [{
      "P_T": 1583.804514476072, 
      "eta": -0.185737493733154, 
      "phi": 0.23724726720847764, 
      "E": 1676.0144181790492, 
      "m": 461.5742352387667, 
      "n_particles": 61, 
      "P_T_lead": 323.76043701171875, 
      "dR": {"jet2": 2.9516807884922214}
    }, 
    {
      "P_T": 1914.9441593481358, 
      "eta": 0.36952945735006965, 
      "phi": -2.6617349835458803, 
      "E": 2049.8762080184856, 
      "m": 105.03455038005487, 
      "n_particles": 23, 
      "P_T_lead": 606.7150268554688, 
      "dR": {"jet1": 2.9516807884922214}
    }], 
  "n_particles": 143, 
  "M_jj": 566.6087856188216
}
```
- Type: signal or background
- Jets: List of jets with their kinematic properties and substructure information
    - $P_T$: Transverse momentum
    - $\eta$: Pseudorapidity
    - $\phi$: Azimuthal angle
    - $E$: Energy
    - $m$: Mass
    - $n_{particles}$: Number of constituent particles in the jet
    - $P_{T_{lead}}$: Transverse momentum of the leading constituent particle
    - $dR$: Delta R angular distances to other jets
- $n_{particles}$: Total number of particles in the event
- $M_{jj}$: Invariant mass of the leading jets

In R&D I observed there can be 1 or 2 jets in an event. So I created the data preparation and training scripts to handle variable number of jets.

I created another script to shuffle and split the data in to training and validation sets. I created both balanced (1:1 signal to background) and imbalanced (1:10 signal to background) datasets for training and validation.

After data preparation, I had to find a way to train an LLM with the limited computational resources I have.

| Resource         | Specification
|------------------|----------------------
| GPU              | NVIDIA RTX 3070 (8GB VRAM)
| CPU              | AMD Ryzen 7 7700X (8 cores / 16 threads)
| RAM              | 32 GB DDR5 @6000 MT/s
| Storage          | 2 TB NVMe SSD

I selected the mistral-7b-instruct-v0.3 model from HuggingFace as the base model. It is a 7 billion parameter model fine-tuned for instruction following tasks. It has good performance while being relatively lightweight.

I found that using low rank adaptation (LoRA) would allow me to fine-tune the model within my resource constraints. I used peft library via Hugging Face transformers to implement LoRA fine-tuning. 

For training I created textual prompts that describe the event data quite plainly. I also created a validation script to evaluate the model's performance on validation datasets after each training epoch. 

As initial validation I wanted model to use just 2 words "signal" or "background" to classify events. I had to try multiple prompt formats to find the best performing one. I also wrapped the prompts with `[INST]` and `[/INST]` tokens to make the model understand where the instruction ends and response begins.

```
[INST] Classify this event as 'signal' or 'background'
jets:
    Jet 1: P_T=1583.80, eta=-0.19, phi=0.24, E=1676.01, m=..
      dR_2: 2.95
    Jet 2: P_T=1914.94, eta=0.37, phi=-2.66, E=2049.88, m=..
      dR_1: 2.95
n_particles: 143
M_jj: 566.61
[/INST]
```
Example prompt with made up numbers for illustration.

I did not do further customization to the prompt at this stage, such as units or reducing dR information. Also *I decided to use the balanced dataset for initial training runs* to make sure model doesn't collapse to predicting only background. Which was the case in my initial tests.

### Training Attempts
After some runs I found that I was hitting a ceiling in accuracy around 84% with balanced validation dataset and around 90% with imbalanced validation dataset. Imbalanced set seemed like it was performing better but this was misleading since background events were dominant. So I needed to find a way to improve the model's performance further.

#### Numeric Fusion Adapter

My reading about how tokenization works made it clear that LLM was not really understanding the numerical values. It was just treating them as words. So I needed a way to infuse numerical information directly in to the model's hidden states.

Further research led me to try to implement a numeric fusion adapter. Basically creating a small projection network over the first token embedding to infuse numerical information directly. This took multiple days to implement and debug since I fell in to almost all the pitfalls of working with APIs behind HuggingFace transformers. 

During these days I also started to use BlackBox 1 data from LHCO as part of my manual validation/statistics runs. After a few nights of useless training sessions, I finally was able to implement a working numeric fusion adapter.

### Latest Results
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

Please see study diary for more detailed results.


***Results with numeric fusion adapter are dissapointing so far. Initial data shows that model is not performing better compared to previous attempts. Perhaps I need to train a custom token for numeric fusion adapter instead of using the first token embedding. Or perhaps numeric fusion adapter needs more training to stabilize. Maybe I need to increase the capacity of the adapter network. More experiments are needed to find out.***


## What Would Be The Ideal Outcome?
The ideal outcome of this study would be to have a fine-tuned LLM that can classify events with high accuracy, can use tools and have conscise explanations for its decisions. 

An example interaction could be:
```
User: Here is an event data: [event data]. Is it signal or background ? Explain your reasoning.

Model > Reformats the event data to be more easily understandable through external tools like fastjet clustering.
Model > Uses fastjet clustering tool to reduce the dimensionality of the event data.
Model > Uses its learned physics knowledge to analyze the clustered event data.

Model: After analyzing the event data, I conclude that it is a 'signal' event because [detailed reasoning].
```

Or within network of other models:
```
User: Here is an event data: [event data]. Is it signal or background ? Explain your reasoning.

Model A: Based on the event data, I think it is a 'signal' event because [initial reasoning].
Model B: I disagree with Model A. After analyzing the event data, I think it is a 'background' event because [detailed reasoning].

Model C: After considering both Model A and Model B's arguments, I conclude that the event is a 'signal' event because [final reasoning].
```

## Next Steps
I think it is time to consider the viability of the project. If I can not improve the model's performance, I might need to consider alternative approaches. Especially with black-box data performance was quite low. False positive rate was very high which is not acceptable for physics discovery tasks.

I will continue to train the latest setup and perhaps try to change how the numeric fusion adapter is implemented. While doing that I probably need to have a meeting with my advisor to discuss the future direction of the project.