# Appian Operations Center: How to Start the Project

This guide explains the simplest way to boot up the entire project at its full potential. 

Since you have already set up the Docker containers, you only need to run three commands in three separate terminal windows. 

---

## Step 1: Start the Docker Infrastructure
This starts the databases (Postgres/Redis), the Machine Learning tracker (MLflow), and the real-time event streaming engine (Kafka).

1. Open your **Terminal (PowerShell)**.
2. Navigate to the project folder: `cd "c:\DL CP"`
3. Run the following command:
```bash
docker-compose up -d
```
*(You can close this terminal window once it says the containers are started).*

---

## Step 2: Start the Live Data Stream (Kafka)
This step pumps live, simulated operational tasks into the system so the AI has real-time data to analyze.

1. Open a **NEW Terminal (PowerShell)**.
2. Navigate to the project folder: `cd "c:\DL CP"`
3. Run the following commands:
```bash
.\venv\Scripts\Activate.ps1
python -m phase1_pipeline.kafka_producer
```
*(Leave this terminal open! It is constantly pushing live data into the system).*

---

## Step 3: Start the AI Engine & Dashboard
This starts the backend server, the predictive AI models, the Reinforcement Learning auto-optimizer, and the user interface.

1. Open a **NEW Terminal (PowerShell)**.
2. Navigate to the project folder: `cd "c:\DL CP"`
3. Run the following commands:
```bash
.\venv\Scripts\Activate.ps1
python -m phase5_api.runner
```
*(Leave this terminal open! It runs the website).*

---

## Accessing Your Dashboards

Once all three steps are running, you can access everything in your web browser:

👉 **Main Appian Operations Dashboard:** [http://localhost:8000](http://localhost:8000)
👉 **MLflow Dashboard (AI Tracking):** [http://localhost:5001](http://localhost:5001)
👉 **Kafka UI (Data Stream Monitor):** [http://localhost:8080](http://localhost:8080)

---

### To Turn Everything Off
When you are done working, simply go to any terminal, navigate to the folder, and type:
```bash
docker-compose down
```
And you can close the terminal windows that are running Python.
