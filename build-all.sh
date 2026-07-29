#!/bin/bash
set -e

echo "🧹 Tidying Go dependencies and building Scheduler..."
cd cmd/scheduler || cd . # Adjust if root contains go.mod
go mod tidy
cd - > /dev/null

docker build -f cmd/scheduler/Dockerfile -t utsavmistry/niyojak-scheduler:latest .
docker push utsavmistry/niyojak-scheduler:latest

echo "🧹 Updating AI Service dependencies and building..."
cd ai_service
# Optional: if you want pip to pull latest safe versions of requirements
pip install --upgrade --no-cache-dir -r requirements.txt || true
docker build -t utsavmistry/niyojak-aiservice:latest .
docker push utsavmistry/niyojak-aiservice:latest
cd ..

echo "🧹 Fixing To-Do App npm vulnerabilities and building..."
cd sample_app
npm audit fix --force || true
docker build -t utsavmistry/niyojak-todo-app:latest .
docker push utsavmistry/niyojak-todo-app:latest
cd ..

echo "✨ All dependencies audited, images built, and pushed!"
