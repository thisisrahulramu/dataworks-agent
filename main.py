import os
import json
import openai
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables (AIPROXY_TOKEN must be set via secrets)
from dotenv import load_dotenv
load_dotenv()

# Set up OpenAI client using your AI Proxy token
openai.api_key = os.environ.get("AIPROXY_TOKEN")
if not openai.api_key:
    raise Exception("AIPROXY_TOKEN is not set. Please set it in your environment variables.")

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins (adjust if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Import task functions
from tasksA import (
    task_A1_install_and_run_datagen,
    task_A2_format_markdown,
    task_A3_count_weekdays,
    task_A4_sort_contacts,
    task_A5_recent_log_lines,
    task_A6_extract_markdown_titles,
    task_A7_extract_email_sender,
    task_A8_extract_credit_card,
    task_A9_find_similar_comments,
    task_A10_compute_gold_ticket_sales,
)

task_mapping = {
    "A1": task_A1_install_and_run_datagen,
    "A2": task_A2_format_markdown,
    "A3": task_A3_count_weekdays,
    "A4": task_A4_sort_contacts,
    "A5": task_A5_recent_log_lines,
    "A6": task_A6_extract_markdown_titles,
    "A7": task_A7_extract_email_sender,
    "A8": task_A8_extract_credit_card,
    "A9": task_A9_find_similar_comments,
    "A10": task_A10_compute_gold_ticket_sales,
    }

def classify_task(task: str) -> str:
    """
    For simplicity, we use basic keyword matching to classify the task.
    (In a real solution, you would call the LLM with GPT-4o-Mini to get a structured category.)
    """
    task_lower = task.lower()
    if "datagen.py" in task_lower or "install uv" in task_lower:
        return "A1"
    elif "format" in task_lower and "prettier" in task_lower:
        return "A2"
    elif "wednesday" in task_lower or "count" in task_lower and "dates.txt" in task_lower:
        return "A3"
    elif "sort contacts" in task_lower:
        return "A4"
    elif "recent" in task_lower and ".log" in task_lower:
        return "A5"
    elif "markdown" in task_lower and "index" in task_lower:
        return "A6"
    elif "email.txt" in task_lower and "sender" in task_lower:
        return "A7"
    elif "credit card" in task_lower:
        return "A8"
    elif "comments" in task_lower and "similar" in task_lower:
        return "A9"
    elif "ticket-sales.db" in task_lower or "gold" in task_lower:
        return "A10"
    else:
        raise HTTPException(status_code=400, detail="Task not recognized")

@app.post("/run")
def run_task(task: str = Query(..., description="Plain-English task description")):
    try:
        classification = classify_task(task)
        result = task_mapping[classification]()
        return {"status": "success", "task": classification, "result": result}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read", response_class=PlainTextResponse)
def read_file(path: str = Query(..., description="Path to file within /data")):
    if not (os.path.abspath(path).startswith(os.path.abspath("/data")) or os.path.abspath(path).startswith(os.path.abspath("./data"))):
        raise HTTPException(status_code=400, detail="Access to files outside /data is forbidden")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
