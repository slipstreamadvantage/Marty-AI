# Marty AI

Marty is Slipstream Advantage's always-on AI operations teammate.

## Current v1

- FastAPI service with health endpoint
- Persistent PostgreSQL task and audit records
- Background task worker inside the service
- OpenAI Responses API adapter
- Jira webhook intake protected by `MARTY_WEBHOOK_SECRET`
- Safe default: no external side-effect tools are enabled yet

## Endpoints

- `GET /` — service identity
- `GET /health` — service/database health
- `POST /tasks` — queue a task
- `GET /tasks/{id}` — task status/result
- `POST /webhooks/jira` — queue Jira-originated work

## Render

Runtime: Python

Build command:

`pip install -r requirements.txt`

Start command:

`uvicorn app:app --host 0.0.0.0 --port $PORT`

Required environment variables:

- `DATABASE_URL`
- `OPENAI_API_KEY` (optional for health/deployment; required for model work)
- `OPENAI_MODEL` (defaults to `gpt-5-mini`)
- `MARTY_WEBHOOK_SECRET`

## Security

Do not commit API keys or OAuth credentials. Keep external side-effect capabilities behind explicit permissions and audit them.
