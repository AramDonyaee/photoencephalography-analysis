# ============================================================
# Stage 1 — Build the React frontend
# ============================================================
FROM node:20-alpine AS frontend
WORKDIR /fe

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ .
RUN npm run build

# ============================================================
# Stage 2 — Python backend + built frontend
# ============================================================
FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# Copy the built frontend into backend's working directory
COPY --from=frontend /fe/dist ./frontend_dist
# If you use Create React App instead of Vite:
# COPY --from=frontend /fe/build ./frontend_dist

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]