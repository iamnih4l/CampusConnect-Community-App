import os
from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from database import SessionLocal, engine
import models
import uuid
import datetime
import requests
from pydantic import BaseModel
import random
import re

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
import os

# Add this at the top of your file with other directory definitions
PROFILE_DIR = "profile_images"
os.makedirs(PROFILE_DIR, exist_ok=True) 


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Face detection helper
def detect_face(file: UploadFile):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    return len(faces) > 0

# Login endpoint
@app.post("/login/")
async def login(
    username: str = Form(...),
    password: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not detect_face(file):
        return {"success": False, "message": "No face detected. Please try again."}

    user = db.query(models.User).filter(models.User.username == username,
                                        models.User.password == password).first()
    if not user:
        return {"success": False, "message": "Invalid username or password."}
    return {"success": True, "message": f"Welcome {username}!"}

# Endpoint for face verification (used by profile image & webcam)
@app.post("/verify-face/")
async def verify_face(file: UploadFile = File(...)):
    if detect_face(file):
        return {"verified": True, "message": "Face detected! ✅"}
    return {"verified": False, "message": "No face detected. ❌"}


# Signup endpoint
@app.post("/signup/")
async def signup(
    username: str = Form(...),
    password: str = Form(...),
    profile_image: UploadFile = File(None),
    webcam_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Check webcam image for human verification
    if not detect_face(webcam_image):
        return {"success": False, "message": "No face detected in webcam image. Cannot create account."}

    # 2. Optional profile image face check
    profile_path = None
    if profile_image:
        if not detect_face(profile_image):
            return {"success": False, "message": "Profile image must contain a face."}
        profile_path = os.path.join(PROFILE_DIR, profile_image.filename)
        with open(profile_path, "wb") as f:
            f.write(await profile_image.read())

    # 3. Check if username exists
    existing_user = db.query(models.User).filter(models.User.username == username).first()
    if existing_user:
        return {"success": False, "message": "Username already exists."}

    # 4. Create new user
    new_user = models.User(username=username, password=password, profile_pic=profile_path or "")
    db.add(new_user)
    db.commit()

    return {"success": True, "message": f"Account created for {username}!"}

class ChatRequest(BaseModel):
    message: str

def detect_intent(message: str):
    message = message.lower()

    if any(word in message for word in ["event", "happening", "fest", "function"]):
        return "CAMPUS_EVENTS"

    if any(word in message for word in ["club", "team", "community", "join"]):
        return "CLUB_INFO"

    if any(word in message for word in ["exam", "timetable", "schedule", "test"]):
        return "EXAM_INFO"

    if any(word in message for word in ["canteen", "food", "lunch", "eat"]):
        return "CANTEEN"

    if any(word in message for word in ["location", "where", "directions", "block"]):
        return "NAVIGATION"

    if any(word in message for word in ["hi", "hello", "hey", "good morning", "good evening"]):
        return "GREETINGS"

    return "GENERAL"

def handle_greetings():
    responses = [
        "Hey! How can I help you today? 😊",
        "Hello! What can I assist you with?",
        "Hi there! Need help with events, clubs, canteen or anything else?"
    ]
    return random.choice(responses)

def handle_events():
    events = [
        "Synergia Tech Fest starts next week! Don’t miss the AI & ML workshops.",
        "Sports Day is coming up soon. Registrations open on the student portal.",
        "Photography Club is conducting a short film contest this Saturday."
    ]
    return f"Here are some campus events:\n- " + "\n- ".join(events)

def handle_clubs():
    return (
        "Here are some active clubs on campus:\n"
        "- *Nexus Media Club* – photography, videography, editing\n"
        "- *Developer Student Club* – coding, tech sessions\n"
        "- *Cultural Club* – dance, music, drama\n"
        "If you want, I can help you join one!"
    )

def handle_exam_info():
    return (
        "Mid-semester exams start next month.\n"
        "You can download the latest timetable from the student portal.\n"
        "Do you want a link?"
    )

def handle_canteen():
    menu = ["Biryani", "Sandwich", "Pasta", "Dosa", "Juice", "Tea"]
    return (
        f"Today's canteen specials:\n- " +
        "\n- ".join(menu) +
        "\nWant me to suggest the best item?"
    )

def handle_navigation(msg: str):
    if "library" in msg:
        return "The library is in Block A, 2nd floor."
    if "canteen" in msg:
        return "The canteen is behind Block C near the sports ground."
    if "lab" in msg:
        return "The AI & ML lab is on Block B, 3rd floor."
    return "Which place do you need directions to?"

def handle_general(msg: str):
    return (
    "I'm not fully sure about that yet 🤔\n"
    "But I can help with:\n"
    "- Campus events\n"
    "- Clubs\n"
    "- Exam information\n"
    "- Canteen menu\n"
    "- Navigation\n"
    "Ask me anything!"
)





# In-memory feeds
college_feed = []
global_feed = []


# Gemini API key
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_API_URL = "https://gemini.googleapis.com/v1/models/gemini-1.5-t:generateText"
# Dependency

import random
import datetime

# Sample users
sample_users = [
    "Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Hannah",
    "Ian", "Julia", "Kevin", "Lara", "Mike", "Nina", "Oscar", "Priya",
    "Quinn", "Ryan", "Sophia", "Tara"
]

# Sample content
sample_contents = [
    "Just finished my AI model. Feeling unstoppable!",
    "Anyone available for a DSA practice session tonight?",
    "College canteen food hits differently today 😄",
    "Trying out FastAPI — backend feels so clean!",
    "Completed a 10km morning run! Energy boosted.",
    "UI/UX design is harder than it looks 😂",
    "Campus is looking beautiful in the evening sunlight.",
    "Working on my capstone project. I need caffeine!",
    "Anyone tried LangChain? Need some guidance.",
    "Exams are coming... panic mode activated.",
    "Machine learning is fun until your model doesn't learn.",
    "Procrastinated all day. Now regretting everything.",
    "How do people code for 10 hours straight?",
    "The library is my second home at this point.",
    "Just submitted an assignment 2 minutes before deadline.",
    "Looking for a partner for the upcoming hackathon.",
    "Taking a break from social media... sort of.",
    "Life update: I'm tired.",
    "Learned something new about neural networks today!",
    "College fest vibes incoming soon."
]

# Random placeholder images
sample_images = [
    "https://picsum.photos/400/200?random=101",
    "https://picsum.photos/400/200?random=102",
    "https://picsum.photos/400/200?random=103",
    "https://picsum.photos/400/200?random=104",
    "https://picsum.photos/400/200?random=105",
]

import random

# Synonym map
SYNONYMS = {
    "finished": "completed",
    "feeling": "experiencing",
    "available": "free",
    "practice": "rehearsal",
    "today": "this day",
    "beautiful": "gorgeous",
    "working": "engaged",
    "need": "require",
    "guidance": "assistance",
    "coming": "approaching",
    "fun": "enjoyable",
    "tired": "exhausted",
    "learned": "discovered",
    "vibes": "atmosphere",
    "college": "campus",
    "run": "jog",
    "assignment": "task",
    "project": "initiative",
    "exams": "tests"
}

STOP_WORDS = ["the", "is", "in", "on", "at", "a", "an", "for", "and", "but", "or", "of", "to"]

def simulate_summary(text: str) -> str:
    words = text.replace("\n", " ").split()
    
    # Pick important words: skip stop words
    important_words = [SYNONYMS.get(w.lower().strip(".,!?"), w) 
                       for w in words if w.lower() not in STOP_WORDS]

    # Pick top 6–12 words randomly for “summary”
    summary_words = random.sample(important_words, min(len(important_words), 8))
    
    # Capitalize first word and join
    summary = " ".join(summary_words)
    return summary[0].upper() + summary[1:] + "."
# ---------- Chatbot: RAG-lite helper (add to app.py) ----------
from typing import List, Dict
import re
import os
import json
import time

# Stop words reused (feel free to expand)
CHAT_STOP_WORDS = set(STOP_WORDS)

def tokenize(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t for t in text.split() if t and t not in CHAT_STOP_WORDS]
    return tokens

def score_text(query_tokens: List[str], text: str) -> int:
    tokens = tokenize(text)
    # simple overlap count (works well for short social posts)
    return sum(1 for t in tokens if t in query_tokens)

def retrieve_contexts(query: str, top_k: int = 3) -> List[Dict]:
    """
    Score each post by token overlap with query and return top_k posts.
    """
    q_tokens = tokenize(query)
    candidates = []

    # search both feeds
    for feed in (college_feed, global_feed):
        for post in feed:
            score = score_text(q_tokens, post.get("content", ""))
            # small boost if post has summary
            if post.get("summary"):
                score += 0.5
            if score > 0:
                candidates.append((score, post))

    # sort by score then time (if equal)
    candidates.sort(key=lambda x: (-x[0], x[1].get("time", "")))
    top_posts = [p for _, p in candidates[:top_k]]
    return top_posts

def build_context_prompt(query: str, contexts: List[Dict]) -> str:
    """
    Build a small prompt using the retrieved posts (content + summary).
    """
    parts = []
    parts.append("You are CampusConnect helper. Use the following community posts as context to answer the user's question.\n")
    for i, p in enumerate(contexts, 1):
        parts.append(f"Context #{i} ({p.get('username')} at {p.get('time')}):")
        parts.append(p.get("content", ""))
        if p.get("summary"):
            parts.append("Summary: " + p.get("summary"))
        parts.append("")  # blank line

    parts.append("User question: " + query)
    parts.append("Answer concisely and mention which contexts you used (e.g., Context #1). If none apply, say you couldn't find related posts but provide helpful guidance.")
    return "\n".join(parts)

# If you have GEMINI_API_KEY and want to call remote LLM:
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # keep using your env approach
GEMINI_API_URL = "https://gemini.googleapis.com/v1/models/gemini-1.5-t:generateText"

def call_gemini(prompt: str, max_tokens: int = 200):
    """
    Call Gemini text generation. Returns text or raises.
    (If you want to use a different LLM, modify this function.)
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gemini-1.5-t",
        "input": prompt,
        "temperature": 0.2,
        "max_output_tokens": max_tokens
    }
    # Using requests (make sure requests is installed)
    resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # try common field names
    if isinstance(data, dict):
        # gemini-like wrapper: prefer output_text or choices
        if "output_text" in data:
            return data["output_text"]
        if "candidates" in data and isinstance(data["candidates"], list):
            return " ".join(c.get("content","") for c in data["candidates"])
        # fallback: stringified
        return json.dumps(data)
    return str(data)

def local_answer_from_context(query: str, contexts: List[Dict]) -> str:
    """
    If there's no LLM key, produce a local helpful answer by:
    - listing top contexts used
    - providing suggestions/actions
    """
    if not contexts:
        # generic fallback guidance
        return ("I couldn't find related community posts. Here are a few suggestions:\n"
                "1) Try searching by keywords (event names, project, course code).\n"
                "2) Ask in a relevant community (e.g., AI Enthusiasts).\n"
                "3) Tell me more details and I can help draft a post or recommendation.")
    # build a combined reply using summaries and content
    used_refs = []
    for i, p in enumerate(contexts, 1):
        summary = p.get("summary") or (p.get("content")[:140] + ("..." if len(p.get("content",""))>140 else ""))
        used_refs.append(f"Context #{i} ({p.get('username')}): {summary}")

    answer_lines = [
        "Based on the following posts I found in CampusConnect:",
        *used_refs,
        "",
        "Here's a helpful answer:"
    ]
    # naive synthesis: echo key verbs / give next steps
    # pick nouns/verbs from query for a directive:
    q_tokens = tokenize(query)
    verbs_hint = ", ".join(q_tokens[:6]) if q_tokens else ""
    answer_lines.append(f"- Quick suggestion: {verbs_hint} (use these keywords to search or tag posts).")
    answer_lines.append("- If you want, I can draft a post asking for help or summarise these contexts further.")
    answer = "\n".join(answer_lines)
    return answer

# Chat endpoint
@app.post("/chat/")
async def chat_endpoint(payload: dict):
    """
    Expects JSON: { "user": "username", "message": "text" }
    Returns: { "reply": "...", "contexts": [...], "used_api": "gemini" or "local" }
    """
    user = payload.get("user", "anonymous")
    message = payload.get("message", "").strip()
    if not message:
        return JSONResponse({"reply": "Please send a message.", "contexts": [], "used_api": "none"})

    # 1. Retrieve contexts
    contexts = retrieve_contexts(message, top_k=3)

    # 2. Build prompt
    prompt = build_context_prompt(message, contexts)

    # 3. If API key present, call Gemini (safe-guarded), else local answer
    try:
        if GEMINI_API_KEY:
            reply_text = call_gemini(prompt, max_tokens=220)
            used_api = "gemini"
        else:
            reply_text = local_answer_from_context(message, contexts)
            used_api = "local"
    except Exception as e:
        # fallback to local answer if API call fails
        reply_text = local_answer_from_context(message, contexts) + f"\n\n(Note: external API failed: {e})"
        used_api = "local"

    # Optional: store recent chat in-memory (not persistent). Keep small.
    # We will return the contexts and reply to the frontend for display.
    return JSONResponse({"reply": reply_text, "contexts": contexts, "used_api": used_api})
# ---------- end Chatbot helpers ----------



# FastAPI endpoint
@app.post("/summarize/")
async def summarize_post(data: dict):
    text = data.get("text", "")
    if not text:
        return {"summary": ""}

    summary = simulate_summary(text)
    return {"summary": summary}


def generate_post():
    content = random.choice(sample_contents)
    return {
        "id": str(uuid.uuid4()),
        "username": random.choice(sample_users),
        "avatar": f"https://i.pravatar.cc/50?img={random.randint(1, 70)}",
        "time": (datetime.datetime.now() - datetime.timedelta(
                minutes=random.randint(1, 20000))).strftime("%Y-%m-%d %H:%M:%S"),
        "content": content,
        "summary": simulate_summary(content),  # <-- add summary
        "image": random.choice(sample_images) if random.random() < 0.4 else "",
        "likes": random.randint(0, 200),
        "comments": []
    }

# Add 20 sample posts to college feed
for _ in range(20):
    college_feed.append(generate_post())

# Add 20 sample posts to global feed
for _ in range(20):
    global_feed.append(generate_post())


@app.post("/create-post/")
async def create_post(
    username: str = Form(...),
    content: str = Form(...),
    post_image: UploadFile = File(None),
    feed_type: str = Form("college")  # "college" or "global"
):
    image_url = ""
    if post_image:
        ext = os.path.splitext(post_image.filename)[1]
        filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(POST_IMAGE_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(await post_image.read())
        image_url = file_path  # You can serve static files later if needed

    post = {
    "id": str(uuid.uuid4()),
    "username": username,
    "avatar": f"https://i.pravatar.cc/50?img=12",
    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "content": content,
    "summary": simulate_summary(content),  # <-- add summary
    "image": image_url,
    "likes": 0,
    "comments": []
}
    if feed_type == "college":
        college_feed.insert(0, post)
    else:
        global_feed.insert(0, post)

    return JSONResponse({"success": True, "post": post})

@app.get("/get-feed/")
async def get_feed(feed_type: str = "college"):
    if feed_type == "college":
        return {"feed": college_feed}
    else:
        return {"feed": global_feed}

@app.post("/chatbot")
async def chatbot_endpoint(req: ChatRequest):
    user_msg = req.message.strip()

    intent = detect_intent(user_msg)

    if intent == "GREETINGS":
        reply = handle_greetings()
    elif intent == "CAMPUS_EVENTS":
        reply = handle_events()
    elif intent == "CLUB_INFO":
        reply = handle_clubs()
    elif intent == "EXAM_INFO":
        reply = handle_exam_info()
    elif intent == "CANTEEN":
        reply = handle_canteen()
    elif intent == "NAVIGATION":
        reply = handle_navigation(user_msg)
    else:
        reply = handle_general(user_msg)

    return {"reply": reply}
