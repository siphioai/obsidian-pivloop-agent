"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.chat import router as chat_router
from app.config import get_settings
from app.dependencies import logger

settings = get_settings()

app = FastAPI(title="Obsidian PivLoop Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "vault_path": str(settings.vault_path),
    }


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API info."""
    return {"name": "Obsidian PivLoop Agent", "version": "0.1.0", "docs": "/docs"}


logger.info("app_startup", extra={"host": settings.host, "port": settings.port})
