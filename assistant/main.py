"""
assistant/main.py
─────────────────
CLI entry point for the VIKMO Dealer Assistant.

Run:
    python -m assistant.main

Or directly:
    python assistant/main.py

Requires:
    GEMINI_API_KEY environment variable to be set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Make sure project root is on the path ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rich.console import Console
from rich.panel   import Panel
from rich.text    import Text
from rich.prompt  import Prompt
from rich.rule    import Rule

from assistant.agent import DealerAssistant

console = Console()


def print_banner() -> None:
    console.print(
        Panel(
            Text(
                "VIKMO Dealer Assistant\n"
                "Auto Parts · Inventory · Orders",
                justify="center",
                style="bold white",
            ),
            style       = "bold blue",
            expand      = False,
            subtitle    = "Type 'quit' or 'exit' to leave | 'reset' to clear history",
        )
    )


def run_cli() -> None:
    print_banner()
    console.print()

    try:
        agent = DealerAssistant()
    except EnvironmentError as e:
        console.print(f"[bold red]Setup error:[/bold red] {e}")
        sys.exit(1)

    console.print("[green]✓ Catalogue index ready. Start chatting![/green]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "reset":
            agent.reset()
            console.print(Rule("[dim]Conversation reset[/dim]"))
            continue

        console.print()
        with console.status("[bold yellow]Thinking…[/bold yellow]"):
            try:
                reply = agent.chat(user_input)
            except Exception as exc:
                reply = f"[Error] {exc}"

        console.print(
            Panel(reply, title="[bold green]VIKMO Assistant[/bold green]", border_style="green")
        )
        console.print()


if __name__ == "__main__":
    run_cli()
