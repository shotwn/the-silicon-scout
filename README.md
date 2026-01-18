The Silicon Scout: Feasibility of Autonomous Anomaly Detection via Agentic LLM Orchestration
===============================
BSc. Thesis Project by

Fikret Anıl Haksever

Yıldız Technical University, 2025-2026

## Experiment I
[Study Diary](https://shotwn.github.io/bsc-project/study-diary.html) | [Interim Reports](https://shotwn.github.io/bsc-project/interim-reports/interim-report-1.html)
## Experiment II
### Run Framework
1. Make sure you have Python 3.12+ installed.
2. Clone the repository:
   ```bash
   git clone https://github.com/shotwn/the-silicon-scout.git
   cd the-silicon-scout
   ```
3. Install the required packages:
   ```bash
    pip install -r requirements.txt
    ```
4. Run the framework:
    ```bash
    python -m framework
    ```
### Configuration
You can create an .env file in the root directory to set your environment variables. Here are the variables you can set:
```
GEMMA_API_KEY=API KEY FROM GOOGLE CLOUD
LACATHODE_BATCH_SIZE=4096
LACATHODE_FLOW_EPOCHS=500
LACATHODE_CLASSIFIER_EPOCHS=250
```