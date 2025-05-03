set dotenv-load

default:
    just --list

init-venv:
    uv venv -p 3.12

fmt:
    uv run ruff check --fix-only ai_voice_assistant
    uv run isort ai_voice_assistant
    uv run ruff format ai_voice_assistant

lint:
    uv run isort --check ai_voice_assistant
    uv run ruff format --check ai_voice_assistant
    uv run ruff check ai_voice_assistant
    uv run mypy ai_voice_assistant

update-deps:
    uv pip compile --no-header --upgrade pyproject.toml -o requirements.txt

install-deps:
    uv pip install -r requirements.txt

run:
    uv run python -m ai_voice_assistant
