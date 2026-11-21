"""Bot Telegram MaintiDoc (étape 2.6) : client léger de la recherche, en long polling."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from maintidoc.answering.citations import citation_of
from maintidoc.retrieval import search as S

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx écrit le token dans les URLs
log = logging.getLogger("maintidoc.telegram")

TOP_N, MAX_QUESTION, MAX_IMAGES = 5, 500, 2
MODES = {
    "texte": lambda q, g, n: S.search_text(q, g, k=n),
    "hybride": lambda q, g, n: S.search(q, g, use_reranker=False, top_n=n, candidates=15),
    "reranker": lambda q, g, n: S.search(q, g, use_reranker=True, top_n=n, candidates=15),
}
DEFAULT_MODE = "reranker"


def load_users(path: str = "configs/users.yaml") -> dict[int, list[str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {int(uid): list(groups) for uid, groups in (data.get("users") or {}).items()}


USERS = load_users()


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "MaintiDoc : aide documentaire de maintenance.\n"
        "Elle ne pilote aucun équipement et ne remplace ni les procédures de sécurité "
        "ni un technicien habilité.\n\n"
        "Posez une question. Commandes : /id, /mode texte|hybride|reranker")


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"Votre identifiant Telegram : {update.effective_user.id}")


async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args and context.args[0] in MODES:
        context.user_data["mode"] = context.args[0]
    await update.message.reply_text(
        f"Mode actuel : {context.user_data.get('mode', DEFAULT_MODE)} (choix : {', '.join(MODES)})")


async def answer(update: Update, context: ContextTypes.DEFAULT_TYPE, question: str) -> None:
    uid = update.effective_user.id
    groups = USERS.get(uid)
    if not groups:
        log.warning("access_denied user=%s", uid)
        await update.message.reply_text(
            f"Accès non autorisé. Transmettez cet identifiant à l'administrateur : {uid}")
        return

    mode = context.user_data.get("mode", DEFAULT_MODE)
    await context.bot.send_chat_action(update.effective_chat.id, "typing")
    hits = await asyncio.to_thread(MODES[mode], question[:MAX_QUESTION], groups, TOP_N)
    log.info("search user=%s mode=%s results=%d", uid, mode, len(hits))  # métadonnées seulement

    if not hits:
        await update.message.reply_text("Aucun résultat dans les documents accessibles pour votre profil.")
        return

    lines = ["Sources retrouvées (pas de réponse rédigée à ce stade) :"]
    for rank, h in enumerate(hits, start=1):
        c = citation_of(h)
        extrait = (h.get("text") or "").replace("\n", " ")[:260]
        lines.append(f"\n{rank}. {c.document_id} · p.{c.page} · {c.element_id}\n{extrait}")
    # texte brut (pas de parse_mode) : le contenu des documents n'est jamais interprété
    await update.message.reply_text("\n".join(lines)[:4000])

    sent = 0
    for h in hits:
        path = h.get("image_path")
        if path and Path(path).exists() and sent < MAX_IMAGES:
            c = citation_of(h)
            with open(path, "rb") as f:
                await update.message.reply_photo(f, caption=f"{c.element_id} · p.{c.page}")
            sent += 1


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await answer(update, context, update.message.text.strip())


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Photo reçue, mais elle n'est pas encore analysée : l'analyse visuelle arrive à l'étape 3. "
        "Une ressemblance visuelle ne suffit de toute façon pas à confirmer un diagnostic.")
    if update.message.caption:
        await answer(update, context, update.message.caption.strip())


def main() -> None:
    app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    log.info("Préchargement des modèles...")
    S.search("test", next(iter(USERS.values()), ["maintenance"]))  # évite 30 s au premier message
    log.info("Bot prêt.")
    app.run_polling()


if __name__ == "__main__":
    main()