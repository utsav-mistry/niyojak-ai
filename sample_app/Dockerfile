# syntax=docker/dockerfile:1

FROM node:20-alpine AS runtime

WORKDIR /app

# Install build tools needed by better-sqlite3 (native module)
RUN apk add --no-cache python3 make g++

COPY package.json ./
RUN npm install --omit=dev

COPY backend/ ./backend/
COPY public/  ./public/

ENV PORT=3000 \
    AI_SERVICE_URL=http://niyojak-aiservice:8000 \
    DB_PATH=/data/tasks.db \
    NIYOJAK_NAMESPACE=niyojak-system

EXPOSE 3000

VOLUME ["/data"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:3000/health || exit 1

CMD ["node", "backend/server.js"]
