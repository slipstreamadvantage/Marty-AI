import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import DateTime, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from openai import OpenAI

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./marty.db')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql+psycopg://', 1)
elif DATABASE_URL.startswith('postgresql://'):
    DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+psycopg://', 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

class Base(DeclarativeBase):
    pass

class Task(Base):
    __tablename__ = 'tasks'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source: Mapped[str] = mapped_column(String(50), default='api')
    actor: Mapped[str] = mapped_column(String(200), default='unknown')
    instruction: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default='queued', index=True)
    result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

class Audit(Base):
    __tablename__ = 'audit_log'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    event: Mapped[str] = mapped_column(String(100))
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

Base.metadata.create_all(engine)
app = FastAPI(title='Marty AI', version='1.0.0')

class TaskCreate(BaseModel):
    instruction: str
    source: str = 'api'
    actor: str = 'unknown'

def now():
    return datetime.now(timezone.utc)

def audit(session: Session, event: str, task_id: Optional[str] = None, detail: Optional[str] = None):
    session.add(Audit(id=str(uuid.uuid4()), task_id=task_id, event=event, detail=detail, created_at=now()))

def enqueue(instruction: str, source: str, actor: str) -> str:
    task_id = str(uuid.uuid4())
    ts = now()
    with Session(engine) as session:
        session.add(Task(id=task_id, source=source, actor=actor, instruction=instruction,
                         status='queued', created_at=ts, updated_at=ts))
        audit(session, 'task.queued', task_id, f'{source}:{actor}')
        session.commit()
    return task_id

def run_model(instruction: str) -> str:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return 'Marty is live, but OPENAI_API_KEY has not yet been configured.'
    client = OpenAI(api_key=api_key)
    model = os.getenv('OPENAI_MODEL', 'gpt-5-mini')
    response = client.responses.create(
        model=model,
        input=[
            {'role': 'system', 'content': 'You are Marty, Slipstream Advantage\'s AI operations teammate. Be concise, careful, and action-oriented. Never claim an external action occurred unless a tool actually performed it.'},
            {'role': 'user', 'content': instruction},
        ],
    )
    return response.output_text

def process_one() -> bool:
    with Session(engine) as session:
        task = session.scalars(select(Task).where(Task.status == 'queued').order_by(Task.created_at).limit(1)).first()
        if not task:
            return False
        task.status = 'running'
        task.updated_at = now()
        audit(session, 'task.started', task.id)
        session.commit()
        task_id, instruction = task.id, task.instruction
    try:
        result = run_model(instruction)
        with Session(engine) as session:
            task = session.get(Task, task_id)
            task.status = 'completed'
            task.result = result
            task.updated_at = now()
            audit(session, 'task.completed', task_id)
            session.commit()
    except Exception as exc:
        with Session(engine) as session:
            task = session.get(Task, task_id)
            task.status = 'failed'
            task.error = str(exc)
            task.updated_at = now()
            audit(session, 'task.failed', task_id, str(exc))
            session.commit()
    return True

async def worker_loop():
    while True:
        try:
            worked = await asyncio.to_thread(process_one)
            await asyncio.sleep(0.2 if worked else 2.0)
        except Exception:
            await asyncio.sleep(5.0)

@app.on_event('startup')
async def startup_event():
    asyncio.create_task(worker_loop())

@app.get('/')
def root():
    return {'name': 'Marty', 'status': 'live', 'version': '1.0.0'}

@app.get('/health')
def health():
    try:
        with Session(engine) as session:
            session.execute(select(1))
        db = 'ok'
    except Exception as exc:
        db = f'error: {exc}'
    return {'status': 'ok' if db == 'ok' else 'degraded', 'database': db}

@app.post('/tasks')
def create_task(payload: TaskCreate):
    if not payload.instruction.strip():
        raise HTTPException(400, 'instruction is required')
    task_id = enqueue(payload.instruction.strip(), payload.source, payload.actor)
    return {'id': task_id, 'status': 'queued'}

@app.get('/tasks/{task_id}')
def get_task(task_id: str):
    with Session(engine) as session:
        task = session.get(Task, task_id)
        if not task:
            raise HTTPException(404, 'task not found')
        return {
            'id': task.id, 'source': task.source, 'actor': task.actor,
            'instruction': task.instruction, 'status': task.status,
            'result': task.result, 'error': task.error,
            'created_at': task.created_at, 'updated_at': task.updated_at,
        }

@app.post('/webhooks/jira')
def jira_webhook(payload: dict, x_marty_secret: Optional[str] = Header(default=None)):
    expected = os.getenv('MARTY_WEBHOOK_SECRET')
    if expected and x_marty_secret != expected:
        raise HTTPException(401, 'invalid webhook secret')
    instruction = payload.get('instruction') or payload.get('comment') or payload.get('text')
    if not instruction:
        instruction = f'Process this Jira event: {payload}'
    task_id = enqueue(str(instruction), 'jira', str(payload.get('actor', 'jira')))
    return {'id': task_id, 'status': 'queued'}
