#!/usr/bin/env python3
"""Beautiful TUI for viewing trajectory datasets.

Usage:
    python -m src.cli.view_trajectories
    python -m src.cli.view_trajectories --path data/trajectories/checkpoint/dataset
    python -m src.cli.view_trajectories --help
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from rich.console import Console
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Rule,
    Static,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# Custom CSS Theme - Cyberpunk / Neon
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
$primary: #00ff9f;
$secondary: #ff00ff;
$accent: #00d4ff;
$warning: #ffaa00;
$error: #ff4444;
$surface: #0d1117;
$surface-light: #161b22;
$surface-lighter: #21262d;

Screen {
    background: $surface;
}

Header {
    background: $surface-light;
    color: $primary;
    text-style: bold;
    border-bottom: tall $primary 30%;
}

Footer {
    background: $surface-light;
    color: $accent;
    border-top: tall $primary 30%;
}

#main-container {
    layout: horizontal;
}

#sidebar {
    width: 45;
    background: $surface-light;
    border-right: tall $primary 30%;
    padding: 0 1;
}

#content-area {
    width: 1fr;
    background: $surface;
}

#detail-container {
    height: 100%;
    width: 100%;
}

TrajectoryDetailView {
    height: 100%;
    width: 100%;
}

#trajectory-list {
    height: 100%;
    scrollbar-color: $primary 30%;
    scrollbar-color-hover: $primary 50%;
    scrollbar-color-active: $primary;
}

#trajectory-list > .datatable--header {
    background: $surface-lighter;
    color: $accent;
    text-style: bold;
}

#trajectory-list > .datatable--cursor {
    background: $primary 30%;
    color: $primary;
    text-style: bold;
}

#trajectory-list > .datatable--even-row {
    background: $surface-light;
}

#trajectory-list > .datatable--odd-row {
    background: $surface;
}

#detail-header {
    height: auto;
    max-height: 15;
    padding: 1 2;
    background: $surface-lighter;
    border-bottom: tall $primary 30%;
}

.stat-label {
    color: #8b949e;
}

.stat-value {
    color: $primary;
    text-style: bold;
}

.stat-success {
    color: #3fb950;
}

.stat-failure {
    color: $error;
}

#conversations-scroll {
    height: 1fr;
    min-height: 10;
    padding: 1 2;
    scrollbar-color: $accent 30%;
    scrollbar-color-hover: $accent 50%;
}

.message-container {
    height: auto;
    margin-bottom: 1;
    padding: 0 1;
    background: $surface-light;
    border-left: tall $surface-lighter;
}

.message-role {
    text-style: bold;
    height: auto;
}

.role-system {
    color: $warning;
}

.role-user {
    color: $accent;
}

.role-assistant {
    color: $primary;
}

.role-tool {
    color: $secondary;
}

.message-content {
    color: #c9d1d9;
    height: auto;
    padding: 0;
}

#tool-calls-scroll {
    height: 1fr;
    min-height: 10;
    padding: 1 2;
    scrollbar-color: $secondary 30%;
}

.tool-call-card {
    margin-bottom: 1;
    padding: 1;
    background: $surface-light;
    border: tall $secondary 30%;
}

.tool-header {
    layout: horizontal;
    height: auto;
    padding-bottom: 1;
    border-bottom: tall $surface-lighter;
}

.tool-name {
    color: $secondary;
    text-style: bold;
    width: 1fr;
}

.tool-turn {
    color: #8b949e;
    width: auto;
}

.tool-metrics {
    layout: horizontal;
    height: auto;
    padding: 1 0;
    color: #8b949e;
}

.metric {
    width: 1fr;
}

.metric-label {
    color: #6e7681;
}

.metric-value {
    color: $accent;
}

.tool-thought {
    padding: 1;
    margin: 1 0;
    background: $surface;
    border-left: tall $warning;
    color: #c9d1d9;
}

.tool-io {
    margin-top: 1;
}

.tool-io-label {
    color: #8b949e;
    text-style: italic;
    padding-bottom: 1;
}

.tool-io-content {
    padding: 1;
    background: $surface;
    color: #c9d1d9;
}

#stats-panel {
    height: auto;
    padding: 1 2;
    background: $surface-lighter;
    border-top: tall $primary 30%;
}

.stats-grid {
    layout: grid;
    grid-size: 4;
    grid-gutter: 1 2;
}

.sidebar-title {
    text-align: center;
    text-style: bold;
    color: $primary;
    padding: 1;
    background: $surface;
    border-bottom: tall $primary 30%;
}

#empty-state {
    width: 100%;
    height: 100%;
    content-align: center middle;
    color: #8b949e;
    text-style: italic;
}

TabbedContent {
    height: 100%;
}

TabbedContent > ContentSwitcher {
    height: 1fr;
}

TabPane {
    padding: 0;
}

Tabs {
    background: $surface-light;
    border-bottom: tall $primary 30%;
}

Tab {
    color: #8b949e;
    background: transparent;
}

Tab:hover {
    color: $accent;
    background: $surface-lighter;
}

Tab.-active {
    color: $primary;
    background: $surface-lighter;
    text-style: bold;
}

Underline {
    color: $primary;
}

Rule {
    color: $surface-lighter;
}

#loading-indicator {
    width: 100%;
    height: 100%;
    content-align: center middle;
    color: $primary;
}

.code-block {
    padding: 1;
    background: #0d1117;
    border: tall #30363d;
    margin: 1 0;
}

.detail-scroll {
    height: 1fr;
    padding: 1 2;
    scrollbar-color: $primary 30%;
    scrollbar-color-hover: $primary 50%;
}

.section-header {
    text-align: center;
    text-style: bold;
    color: $accent;
    padding: 1 0;
    margin: 1 0;
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Widgets
# ═══════════════════════════════════════════════════════════════════════════════


class MessageWidget(Static):
    """Widget to display a single conversation message."""

    def __init__(self, role: str, content: str, sender_name: str = None) -> None:
        super().__init__()
        self.role = role
        self.content = content
        self.sender_name = sender_name

    def compose(self) -> ComposeResult:
        role_class = f"role-{self.role.lower()}"
        role_label = self.role.upper()
        if self.sender_name:
            role_label += f" ({self.sender_name})"

        with Container(classes="message-container"):
            yield Static(role_label, classes=f"message-role {role_class}")
            # Truncate very long content for display
            display_content = self.content
            if len(display_content) > 5000:
                display_content = display_content[:5000] + "\n\n... [truncated]"
            yield Static(display_content, classes="message-content")


class ToolCallWidget(Static):
    """Widget to display a single tool call."""

    def __init__(self, tool_call: dict) -> None:
        super().__init__()
        self.tool_call = tool_call

    def compose(self) -> ComposeResult:
        tc = self.tool_call

        with Container(classes="tool-call-card"):
            # Header with tool name and turn
            with Horizontal(classes="tool-header"):
                yield Static(f"🔧 {tc.get('tool_name', 'unknown')}", classes="tool-name")
                yield Static(f"Turn {tc.get('turn', '?')}", classes="tool-turn")

            # Metrics row
            with Horizontal(classes="tool-metrics"):
                energy = tc.get("energy_joules", 0)
                latency = tc.get("latency_seconds", 0)
                cost = tc.get("cost_usd", 0)
                tokens = tc.get("tokens", {})
                total_tokens = tokens.get("total_tokens", 0) if isinstance(tokens, dict) else 0

                yield Static(f"⚡ {energy:.4f}J", classes="metric")
                yield Static(f"⏱ {latency:.2f}s", classes="metric")
                yield Static(f"💰 ${cost:.6f}", classes="metric")
                yield Static(f"📝 {total_tokens} tok", classes="metric")

            # Thought (if present)
            thought = tc.get("thought", "")
            if thought:
                yield Static("💭 Thought:", classes="tool-io-label")
                thought_display = thought[:2000] + "..." if len(thought) > 2000 else thought
                yield Static(thought_display, classes="tool-thought")

            # Input
            tool_input = tc.get("tool_input", "")
            if tool_input:
                yield Static("📥 Input:", classes="tool-io-label")
                input_display = tool_input[:2000] + "..." if len(tool_input) > 2000 else tool_input
                try:
                    # Try to parse and pretty-print JSON
                    parsed = json.loads(input_display)
                    input_display = json.dumps(parsed, indent=2)
                except (json.JSONDecodeError, TypeError):
                    pass
                yield Static(input_display, classes="tool-io-content code-block")

            # Output
            tool_output = tc.get("tool_output", "")
            if tool_output:
                yield Static("📤 Output:", classes="tool-io-label")
                output_display = tool_output[:3000] + "..." if len(tool_output) > 3000 else tool_output
                yield Static(output_display, classes="tool-io-content code-block")

            # Final answer indicator
            if tc.get("is_final_answer"):
                yield Static("✅ FINAL ANSWER", classes="stat-success")


class TrajectoryDetailView(ScrollableContainer):
    """Full detail view for a trajectory - all scrollable."""

    def __init__(self, trajectory: dict) -> None:
        super().__init__(classes="detail-scroll")
        self.trajectory = trajectory

    def compose(self) -> ComposeResult:
        t = self.trajectory
        conversations = t.get("conversations", [])
        tool_calls = t.get("tool_calls", [])

        # Compact header info
        success = "✅" if t.get("success", False) else "❌"
        yield Static(
            f"{success} {t.get('sample_id', 'N/A')} | {t.get('category', 'N/A')} | "
            f"⚡{t.get('total_energy_joules', 0):.3f}J | ⏱{t.get('total_latency_seconds', 0):.1f}s | "
            f"💰${t.get('total_cost_usd', 0):.5f} | 📝{t.get('total_tokens', 0)} tok",
            classes="section-header"
        )

        # Ground truth
        ground_truth = t.get("ground_truth", "")
        if ground_truth:
            yield Static(f"🎯 Ground Truth: {ground_truth}", classes="stat-label")

        yield Rule()

        # Conversations section
        yield Static("═══ 💬 CONVERSATIONS ═══", classes="section-header")
        if not conversations:
            yield Static("No conversations in this trajectory", classes="stat-label")
        else:
            for msg in conversations:
                yield MessageWidget(
                    role=msg.get("role", "unknown"),
                    content=msg.get("content", ""),
                    sender_name=msg.get("name"),
                )

        yield Rule()

        # Tool calls section
        yield Static("═══ 🔧 TOOL CALLS ═══", classes="section-header")
        if not tool_calls:
            yield Static("No tool calls in this trajectory", classes="stat-label")
        else:
            for tc in tool_calls:
                yield ToolCallWidget(tc)


# ═══════════════════════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════════════════════


class TrajectoryViewer(App):
    """A beautiful TUI for viewing trajectory datasets."""

    TITLE = "🚀 Trajectory Viewer"
    SUB_TITLE = "Intelligence Per Watt"
    CSS = CSS

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("g", "goto_first", "First"),
        Binding("G", "goto_last", "Last", key_display="shift+g"),
        Binding("u", "scroll_up", "Scroll Up"),
        Binding("d", "scroll_down", "Scroll Down"),
        Binding("r", "refresh", "Refresh"),
        Binding("?", "help", "Help"),
    ]

    selected_index: reactive[int] = reactive(0)
    dataset_path: str = ""
    trajectories: list[dict] = []

    def __init__(self, dataset_path: str) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.trajectories = []

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="main-container"):
            with Vertical(id="sidebar"):
                yield Static("📊 TRAJECTORIES", classes="sidebar-title")
                yield DataTable(id="trajectory-list", cursor_type="row")

            with Vertical(id="content-area"):
                yield Container(id="detail-container")

        yield Footer()

    def on_mount(self) -> None:
        """Load data when app starts."""
        self.load_dataset()

    @work(thread=True)
    def load_dataset(self) -> None:
        """Load dataset in background thread."""
        try:
            dataset = load_from_disk(self.dataset_path)
            # Convert to list of dicts
            self.trajectories = [dict(row) for row in dataset]
            self.call_from_thread(self.populate_table)
        except Exception as e:
            self.call_from_thread(self.show_error, str(e))

    def populate_table(self) -> None:
        """Populate the trajectory table."""
        table = self.query_one("#trajectory-list", DataTable)
        table.clear(columns=True)

        # Add columns
        table.add_column("", key="status", width=3)
        table.add_column("ID", key="id", width=12)
        table.add_column("Cat", key="category", width=8)
        table.add_column("⚡J", key="energy", width=8)
        table.add_column("💰$", key="cost", width=10)
        table.add_column("🔄", key="turns", width=4)

        # Add rows
        for i, traj in enumerate(self.trajectories):
            success = traj.get("success", False)
            status = "✅" if success else "❌"
            sample_id = str(traj.get("sample_id", ""))[:10]
            category = str(traj.get("category", ""))[:6]
            energy = f"{traj.get('total_energy_joules', 0):.3f}"
            cost = f"${traj.get('total_cost_usd', 0):.5f}"
            turns = str(traj.get("num_turns", 0))

            table.add_row(status, sample_id, category, energy, cost, turns, key=str(i))

        # Select first row
        if self.trajectories:
            table.cursor_type = "row"
            self.show_trajectory_detail(0)

    def show_error(self, error: str) -> None:
        """Show error message."""
        container = self.query_one("#detail-container")
        container.remove_children()
        container.mount(Static(f"❌ Error loading dataset:\n\n{error}", id="empty-state"))

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        if event.row_key:
            index = int(event.row_key.value)
            self.selected_index = index
            self.show_trajectory_detail(index)

    def show_trajectory_detail(self, index: int) -> None:
        """Show details for selected trajectory."""
        if index >= len(self.trajectories):
            return

        trajectory = self.trajectories[index]
        container = self.query_one("#detail-container")
        container.remove_children()

        # Mount the detail view widget
        container.mount(TrajectoryDetailView(trajectory))

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        table = self.query_one("#trajectory-list", DataTable)
        table.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        table = self.query_one("#trajectory-list", DataTable)
        table.action_cursor_up()

    def action_goto_first(self) -> None:
        """Go to first row."""
        table = self.query_one("#trajectory-list", DataTable)
        table.move_cursor(row=0)
        if self.trajectories:
            self.show_trajectory_detail(0)

    def action_goto_last(self) -> None:
        """Go to last row."""
        table = self.query_one("#trajectory-list", DataTable)
        if self.trajectories:
            table.move_cursor(row=len(self.trajectories) - 1)
            self.show_trajectory_detail(len(self.trajectories) - 1)

    def action_scroll_up(self) -> None:
        """Scroll detail view up."""
        try:
            scroll = self.query_one(".detail-scroll", ScrollableContainer)
            scroll.scroll_up(animate=False)
        except Exception:
            pass

    def action_scroll_down(self) -> None:
        """Scroll detail view down."""
        try:
            scroll = self.query_one(".detail-scroll", ScrollableContainer)
            scroll.scroll_down(animate=False)
        except Exception:
            pass

    def action_refresh(self) -> None:
        """Reload dataset."""
        self.load_dataset()

    def action_help(self) -> None:
        """Show help."""
        self.notify(
            "j/k: Navigate list | g/G: First/Last | u/d: Scroll detail | r: Refresh | q: Quit",
            title="Keyboard Shortcuts",
            timeout=5,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Beautiful TUI for viewing trajectory datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        type=str,
        default="data/trajectories/checkpoint/dataset",
        help="Path to the trajectory dataset (arrow/parquet format)",
    )
    args = parser.parse_args()

    # Resolve path
    dataset_path = Path(args.path)
    if not dataset_path.is_absolute():
        # Try relative to script location first
        script_dir = Path(__file__).parent.parent.parent
        candidate = script_dir / args.path
        if candidate.exists():
            dataset_path = candidate
        else:
            dataset_path = Path.cwd() / args.path

    if not dataset_path.exists():
        console = Console()
        console.print(f"[red]Error:[/red] Dataset not found at {dataset_path}")
        console.print("\nAvailable options:")
        console.print("  1. Specify path with --path /path/to/dataset")
        console.print("  2. Generate trajectories first with generate_trajectories.py")
        sys.exit(1)

    app = TrajectoryViewer(str(dataset_path))
    app.run()


if __name__ == "__main__":
    main()
