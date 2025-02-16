# tasksA.py
import os
import re
import json
import subprocess
import sqlite3
from datetime import datetime
import requests
import numpy as np
from openai import ChatCompletion
from dateutil.parser import parse
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

DATA_DIR = "/data"

# Helper: Ensure file path is within DATA_DIR
def ensure_data_path(filepath: str) -> str:
    if not os.path.abspath(filepath).startswith(os.path.abspath(DATA_DIR)):
        raise Exception("Access denied: file is outside /data")
    return filepath

# A1
def task_A1_install_and_run_datagen():
    email = "23f1000426@ds.study.iitm.ac.in"
    # Check if "uv" command exists
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
    except Exception:
        subprocess.run(["pip", "install", "uv"], check=True)
    # Download datagen.py from GitHub
    datagen_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    response = requests.get(datagen_url)
    if response.status_code != 200:
        raise Exception("Failed to download datagen.py")
    datagen_path = os.path.join("/tmp", "datagen.py")
    with open(datagen_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    # Run the script with email as argument
    subprocess.run(["python", datagen_path, email], check=True)
    return {"message": "A1 executed: datagen.py run successfully."}

# A2
def task_A2_format_markdown():
    file_path = os.path.join(DATA_DIR, "format.md")
    if not os.path.exists(file_path):
        raise Exception(f"{file_path} not found")
    # Use npx to run prettier (assumes Node.js is installed in container)
    subprocess.run(["npx", "prettier@3.4.2", "--write", file_path], check=True)
    return {"message": "A2 executed: format.md formatted."}

# A3
def task_A3_count_weekdays():
    input_file = os.path.join(DATA_DIR, "dates.txt")
    output_file = os.path.join(DATA_DIR, "dates-wednesdays.txt")
    if not os.path.exists(input_file):
        raise Exception(f"{input_file} not found")
    count = 0
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    dt = parse(line)
                    if dt.weekday() == 2:  # Wednesday: 0=Mon,2=Wed
                        count += 1
                except Exception:
                    continue
    with open(output_file, "w") as f:
        f.write(str(count))
    return {"message": f"A3 executed: {count} Wednesdays counted."}

# A4
def task_A4_sort_contacts():
    input_file = os.path.join(DATA_DIR, "contacts.json")
    output_file = os.path.join(DATA_DIR, "contacts-sorted.json")
    if not os.path.exists(input_file):
        raise Exception(f"{input_file} not found")
    with open(input_file, "r") as f:
        contacts = json.load(f)
    contacts.sort(key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))
    with open(output_file, "w") as f:
        json.dump(contacts, f, indent=2)
    return {"message": "A4 executed: contacts sorted."}

# A5
def task_A5_recent_log_lines():
    logs_dir = os.path.join(DATA_DIR, "logs")
    output_file = os.path.join(DATA_DIR, "logs-recent.txt")
    if not os.path.exists(logs_dir):
        raise Exception(f"{logs_dir} not found")
    log_files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".log")]
    if not log_files:
        raise Exception("No .log files found")
    log_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    recent_logs = log_files[:10]
    with open(output_file, "w") as out:
        for log_file in recent_logs:
            with open(log_file, "r") as f:
                first_line = f.readline().strip()
                out.write(first_line + "\n")
    return {"message": "A5 executed: recent log lines written."}

# A6
def task_A6_extract_markdown_titles():
    docs_dir = os.path.join(DATA_DIR, "docs")
    output_file = os.path.join(docs_dir, "index.json")
    if not os.path.exists(docs_dir):
        raise Exception(f"{docs_dir} not found")
    index = {}
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("# "):
                            title = line[2:].strip()
                            # store relative path (remove docs/ prefix)
                            rel_path = os.path.relpath(file_path, docs_dir)
                            index[rel_path] = title
                            break
    with open(output_file, "w") as f:
        json.dump(index, f, indent=2)
    return {"message": "A6 executed: markdown titles indexed."}

# A7
def task_A7_extract_email_sender():
    input_file = os.path.join(DATA_DIR, "email.txt")
    output_file = os.path.join(DATA_DIR, "email-sender.txt")
    if not os.path.exists(input_file):
        raise Exception(f"{input_file} not found")
    with open(input_file, "r") as f:
        content = f.read()
    # Call LLM (using GPT-4o-Mini) with a short prompt.
    prompt = f"Extract only the sender's email address from the following email message:\n\n{content}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    sender_email = response.choices[0].message.content.strip()
    with open(output_file, "w") as f:
        f.write(sender_email)
    return {"message": f"A7 executed: sender email extracted ({sender_email})."}

# A8
def task_A8_extract_credit_card():
    input_image = os.path.join(DATA_DIR, "credit-card.png")
    output_file = os.path.join(DATA_DIR, "credit-card.txt")
    if not os.path.exists(input_image):
        raise Exception(f"{input_image} not found")
    # Read image as base64 string
    import base64
    with open(input_image, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
    prompt = f"Extract only the credit card number from the image represented by this base64 string. Return the number without spaces:\n\n{img_base64}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    card_number = response.choices[0].message.content.strip().replace(" ", "")
    with open(output_file, "w") as f:
        f.write(card_number)
    return {"message": f"A8 executed: credit card number extracted ({card_number})."}

# A9
def task_A9_find_similar_comments():
    input_file = os.path.join(DATA_DIR, "comments.txt")
    output_file = os.path.join(DATA_DIR, "comments-similar.txt")
    if not os.path.exists(input_file):
        raise Exception(f"{input_file} not found")
    with open(input_file, "r") as f:
        comments = [line.strip() for line in f if line.strip()]
    if len(comments) < 2:
        raise Exception("Not enough comments to compare.")
    
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=comments,
    )
    embeddings = [item["embedding"] for item in response["data"]]
    max_sim = -1
    best_pair = (None, None)
    for i in range(len(comments)):
        for j in range(i+1, len(comments)):
            sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            if sim > max_sim:
                max_sim = sim
                best_pair = (comments[i], comments[j])
    with open(output_file, "w") as f:
        f.write(best_pair[0] + "\n" + best_pair[1] + "\n")
    return {"message": "A9 executed: similar comments extracted."}

# A10
def task_A10_compute_gold_ticket_sales():
    db_file = os.path.join(DATA_DIR, "ticket-sales.db")
    output_file = os.path.join(DATA_DIR, "ticket-sales-gold.txt")
    if not os.path.exists(db_file):
        raise Exception(f"{db_file} not found")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE LOWER(TRIM(type)) = 'gold'")
    result = cursor.fetchone()[0]
    conn.close()
    total = result if result is not None else 0
    with open(output_file, "w") as f:
        f.write(str(total))
    return {"message": f"A10 executed: total gold ticket sales = {total}."}

task_A1_install_and_run_datagen()
task_A2_format_markdown()