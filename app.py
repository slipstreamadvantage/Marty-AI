import asyncio
import html
import os
import secrets
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import DateTime, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from openai import OpenAI

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./marty.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


class Base(DeclarativeBase):
    pass


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source: Mapped[str] = mapped_column(String(50), default="api")
    actor: Mapped[str] = mapped_column(String(200), default="unknown")
    instruction: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Audit(Base):
    __tablename__ = "audit_log"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    event: Mapped[str] = mapped_column(String(100))
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


Base.metadata.create_all(engine)
app = FastAPI(title="Marty AI", version="1.1.0")


class TaskCreate(BaseModel):
    instruction: str
    source: str = "api"
    actor: str = "unknown"


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


def now():
    return datetime.now(timezone.utc)


def audit(session: Session, event: str, task_id: Optional[str] = None, detail: Optional[str] = None):
    session.add(
        Audit(
            id=str(uuid.uuid4()),
            task_id=task_id,
            event=event,
            detail=detail,
            created_at=now(),
        )
    )


def require_chat_password(password: Optional[str]):
    expected = os.getenv("MARTY_CHAT_PASSWORD")
    if not expected:
        raise HTTPException(503, "Private chat password has not been configured.")
    if not password or not secrets.compare_digest(password, expected):
        raise HTTPException(401, "Invalid Marty chat password.")


def enqueue(instruction: str, source: str, actor: str) -> str:
    task_id = str(uuid.uuid4())
    ts = now()
    with Session(engine) as session:
        session.add(
            Task(
                id=task_id,
                source=source,
                actor=actor,
                instruction=instruction,
                status="queued",
                created_at=ts,
                updated_at=ts,
            )
        )
        audit(session, "task.queued", task_id, f"{source}:{actor}")
        session.commit()
    return task_id


def model_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY has not been configured.")
    return OpenAI(api_key=api_key), os.getenv("OPENAI_MODEL", "gpt-5-mini")


def run_model(instruction: str) -> str:
    client, model = model_client()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are Marty, Slipstream Advantage's AI operations teammate. "
                    "Be concise, careful, practical, and action-oriented. "
                    "Never claim an external action occurred unless a tool actually performed it."
                ),
            },
            {"role": "user", "content": instruction},
        ],
    )
    return response.output_text


def run_chat(conversation_id: str, message: str) -> str:
    with Session(engine) as session:
        prior = session.scalars(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(20)
        ).all()
        prior = list(reversed(prior))

    input_messages = [
        {
            "role": "system",
            "content": (
                "You are Marty, Slipstream Advantage's always-on AI team member. "
                "You are speaking with Wade and the Slipstream team through Marty's private web chat. "
                "Be concise, practical, technical when useful, and preserve context across the conversation. "
                "Never claim you accessed Jira, email, Slack, GitHub, databases, or any other external system "
                "unless that access was actually performed by an integrated tool."
            ),
        }
    ]
    input_messages.extend({"role": m.role, "content": m.content} for m in prior)
    input_messages.append({"role": "user", "content": message})

    client, model = model_client()
    response = client.responses.create(model=model, input=input_messages)
    return response.output_text


def process_one() -> bool:
    with Session(engine) as session:
        task = session.scalars(
            select(Task)
            .where(Task.status == "queued")
            .order_by(Task.created_at)
            .limit(1)
        ).first()
        if not task:
            return False
        task.status = "running"
        task.updated_at = now()
        audit(session, "task.started", task.id)
        session.commit()
        task_id, instruction = task.id, task.instruction

    try:
        result = run_model(instruction)
        with Session(engine) as session:
            task = session.get(Task, task_id)
            task.status = "completed"
            task.result = result
            task.updated_at = now()
            audit(session, "task.completed", task_id)
            session.commit()
    except Exception as exc:
        with Session(engine) as session:
            task = session.get(Task, task_id)
            task.status = "failed"
            task.error = str(exc)
            task.updated_at = now()
            audit(session, "task.failed", task_id, str(exc))
            session.commit()
    return True


async def worker_loop():
    while True:
        try:
            worked = await asyncio.to_thread(process_one)
            await asyncio.sleep(0.2 if worked else 2.0)
        except Exception:
            await asyncio.sleep(5.0)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker_loop())


@app.get("/")
def root():
    return {
        "name": "Marty",
        "status": "live",
        "version": "1.1.0",
        "chat": "/chat",
    }


@app.get("/health")
def health():
    try:
        with Session(engine) as session:
            session.execute(select(1))
        db = "ok"
    except Exception as exc:
        db = f"error: {exc}"
    return {
        "status": "ok" if db == "ok" else "degraded",
        "database": db,
        "chat_password_configured": bool(os.getenv("MARTY_CHAT_PASSWORD")),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return HTMLResponse(CHAT_HTML)


@app.post("/api/chat")
def chat_api(payload: ChatRequest, x_marty_chat_password: Optional[str] = Header(default=None)):
    require_chat_password(x_marty_chat_password)
    message = payload.message.strip()
    if not message:
        raise HTTPException(400, "message is required")

    conversation_id = payload.conversation_id or str(uuid.uuid4())
    ts = now()

    with Session(engine) as session:
        session.add(
            ChatMessage(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role="user",
                content=message,
                created_at=ts,
            )
        )
        session.commit()

    try:
        answer = run_chat(conversation_id, message)
    except Exception as exc:
        raise HTTPException(502, f"Marty model request failed: {exc}")

    with Session(engine) as session:
        session.add(
            ChatMessage(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                created_at=now(),
            )
        )
        session.commit()

    return {"conversation_id": conversation_id, "message": answer}


@app.get("/api/chat/{conversation_id}")
def chat_history(conversation_id: str, x_marty_chat_password: Optional[str] = Header(default=None)):
    require_chat_password(x_marty_chat_password)
    with Session(engine) as session:
        messages = session.scalars(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.created_at)
        ).all()
    return {
        "conversation_id": conversation_id,
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at}
            for m in messages
        ],
    }


@app.post("/tasks")
def create_task(payload: TaskCreate, x_marty_chat_password: Optional[str] = Header(default=None)):
    require_chat_password(x_marty_chat_password)
    if not payload.instruction.strip():
        raise HTTPException(400, "instruction is required")
    task_id = enqueue(payload.instruction.strip(), payload.source, payload.actor)
    return {"id": task_id, "status": "queued"}


@app.get("/tasks/{task_id}")
def get_task(task_id: str, x_marty_chat_password: Optional[str] = Header(default=None)):
    require_chat_password(x_marty_chat_password)
    with Session(engine) as session:
        task = session.get(Task, task_id)
        if not task:
            raise HTTPException(404, "task not found")
        return {
            "id": task.id,
            "source": task.source,
            "actor": task.actor,
            "instruction": task.instruction,
            "status": task.status,
            "result": task.result,
            "error": task.error,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
        }


@app.post("/webhooks/jira")
def jira_webhook(payload: dict, x_marty_secret: Optional[str] = Header(default=None)):
    expected = os.getenv("MARTY_WEBHOOK_SECRET")
    if expected and x_marty_secret != expected:
        raise HTTPException(401, "invalid webhook secret")
    instruction = payload.get("instruction") or payload.get("comment") or payload.get("text")
    if not instruction:
        instruction = f"Process this Jira event: {payload}"
    task_id = enqueue(str(instruction), "jira", str(payload.get("actor", "jira")))
    return {"id": task_id, "status": "queued"}


CHAT_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Marty</title>
  <style>
    :root { color-scheme: dark; --bg:#090d14; --panel:#111827; --panel2:#172033; --text:#eef2ff; --muted:#97a3b6; --line:#263247; --accent:#8ec5ff; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    .app { min-height:100vh; display:grid; grid-template-rows:auto 1fr auto; max-width:980px; margin:0 auto; }
    header { padding:22px 24px 16px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:14px; }
    .avatar { width:42px; height:42px; border:1px solid var(--line); border-radius:13px; display:grid; place-items:center; font-weight:800; color:var(--accent); background:var(--panel2); }
    h1 { margin:0; font-size:19px; }
    .status { font-size:12px; color:var(--muted); margin-top:3px; }
    #messages { padding:26px 24px 150px; overflow:auto; }
    .msg { max-width:78%; margin:0 0 16px; padding:13px 15px; border-radius:16px; white-space:pre-wrap; line-height:1.48; border:1px solid var(--line); }
    .user { margin-left:auto; background:#1d2b42; }
    .assistant { background:var(--panel); }
    .meta { font-size:11px; color:var(--muted); margin-bottom:5px; }
    .composer { position:fixed; bottom:0; left:0; right:0; background:linear-gradient(transparent,var(--bg) 18%); padding:28px 20px 22px; }
    .composer-inner { max-width:932px; margin:0 auto; display:flex; gap:10px; align-items:flex-end; }
    textarea { flex:1; resize:none; min-height:54px; max-height:180px; padding:15px 16px; border-radius:15px; border:1px solid var(--line); background:var(--panel); color:var(--text); outline:none; font:inherit; }
    button { height:54px; padding:0 20px; border:0; border-radius:14px; background:var(--text); color:#10141c; font-weight:750; cursor:pointer; }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .gate { position:fixed; inset:0; background:rgba(3,6,12,.88); display:grid; place-items:center; padding:24px; z-index:10; backdrop-filter:blur(10px); }
    .gate-card { width:min(430px,100%); background:var(--panel); border:1px solid var(--line); border-radius:20px; padding:26px; }
    .gate-card h2 { margin:0 0 6px; }
    .gate-card p { color:var(--muted); margin:0 0 18px; line-height:1.45; }
    .gate-row { display:flex; gap:9px; }
    input { width:100%; border:1px solid var(--line); background:#0d1421; color:var(--text); border-radius:12px; padding:13px; font:inherit; outline:none; }
    .hidden { display:none; }
    .error { color:#ffb4b4; font-size:12px; margin-top:10px; }
    @media(max-width:640px) { .msg{max-width:92%} .composer{padding-left:12px;padding-right:12px} header,#messages{padding-left:16px;padding-right:16px} }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="avatar">M</div>
      <div><h1>Marty</h1><div class="status">Slipstream AI team member · private chat</div></div>
    </header>
    <main id="messages">
      <div class="msg assistant"><div class="meta">Marty</div>I'm online. What are we working on?</div>
    </main>
    <div class="composer">
      <div class="composer-inner">
        <textarea id="input" placeholder="Message Marty…" rows="1"></textarea>
        <button id="send">Send</button>
      </div>
    </div>
  </div>

  <div id="gate" class="gate">
    <div class="gate-card">
      <h2>Private Marty</h2>
      <p>Enter your Marty chat password. It stays in this browser session and is sent only over HTTPS.</p>
      <div class="gate-row"><input id="password" type="password" autocomplete="current-password" placeholder="Password"><button id="unlock">Open</button></div>
      <div id="gateError" class="error"></div>
    </div>
  </div>

<script>
const messages = document.getElementById('messages');
const input = document.getElementById('input');
const send = document.getElementById('send');
const gate = document.getElementById('gate');
const password = document.getElementById('password');
const unlock = document.getElementById('unlock');
const gateError = document.getElementById('gateError');
let chatPassword = sessionStorage.getItem('marty_chat_password') || '';
let conversationId = localStorage.getItem('marty_conversation_id') || '';

function addMessage(role, text) {
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = role === 'user' ? 'You' : 'Marty';
  el.appendChild(meta);
  el.appendChild(document.createTextNode(text));
  messages.appendChild(el);
  window.scrollTo({top:document.body.scrollHeight, behavior:'smooth'});
}
async function authFetch(url, opts={}) {
  opts.headers = Object.assign({}, opts.headers || {}, {'X-Marty-Chat-Password': chatPassword});
  return fetch(url, opts);
}
async function testPassword() {
  gateError.textContent = '';
  chatPassword = password.value || chatPassword;
  if (!chatPassword) { gateError.textContent='Enter a password.'; return; }
  try {
    if (conversationId) {
      const r = await authFetch('/api/chat/' + encodeURIComponent(conversationId));
      if (r.status === 401 || r.status === 503) throw new Error((await r.json()).detail || 'Access denied');
      if (r.ok) {
        const data = await r.json();
        messages.innerHTML='';
        data.messages.forEach(m => addMessage(m.role, m.content));
      }
    } else {
      const r = await authFetch('/tasks/not-a-real-task');
      if (r.status === 401 || r.status === 503) throw new Error((await r.json()).detail || 'Access denied');
    }
    sessionStorage.setItem('marty_chat_password', chatPassword);
    gate.classList.add('hidden');
    input.focus();
  } catch (e) { gateError.textContent = e.message || 'Could not unlock Marty.'; }
}
unlock.onclick = testPassword;
password.addEventListener('keydown', e => { if (e.key === 'Enter') testPassword(); });
if (chatPassword) { password.value=chatPassword; testPassword(); }

async function sendMessage() {
  const text = input.value.trim();
  if (!text || send.disabled) return;
  input.value='';
  addMessage('user', text);
  send.disabled=true;
  const typing = document.createElement('div');
  typing.className='msg assistant';
  typing.textContent='Marty is thinking…';
  messages.appendChild(typing);
  try {
    const r = await authFetch('/api/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text, conversation_id:conversationId || null})
    });
    const data = await r.json();
    typing.remove();
    if (!r.ok) throw new Error(data.detail || 'Request failed');
    conversationId = data.conversation_id;
    localStorage.setItem('marty_conversation_id', conversationId);
    addMessage('assistant', data.message);
  } catch(e) {
    typing.remove();
    addMessage('assistant', 'Error: ' + (e.message || 'Unable to reach Marty.'));
  } finally {
    send.disabled=false;
    input.focus();
  }
}
send.onclick=sendMessage;
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
</script>
</body>
</html>"""
