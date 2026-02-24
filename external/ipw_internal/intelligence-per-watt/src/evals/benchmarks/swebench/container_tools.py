"""
Container-executed tools for SWE-bench agents.

These tools wrap bash commands that execute INSIDE the Docker container
via SWEBenchEnv.communicate(). The tools themselves are Python functions
running on the HOST, but all file operations happen in the container.

Tool set matches SWE-agent's default SWE-bench config EXACTLY:
- bash
- str_replace_editor (view, create, str_replace, insert, undo_edit)
- submit

That's it. No search tools - agent uses bash + grep/find directly.

Reference: SWE-agent config/default.yaml defines these 3 tools:
  - enable_bash_tool: true              → bash
  - tools/edit_anthropic/config.yaml    → str_replace_editor
  - tools/review_on_submit_m/config.yaml → submit

The str_replace_editor docstring was taken from OpenHands:
https://github.com/All-Hands-AI/OpenHands/blob/main/openhands/agenthub/codeact_agent/function_calling.py

Advanced features ported from SWE-agent's tools/edit_anthropic/bin/str_replace_editor:
- WindowExpander: Smart viewport expansion to function/class boundaries
- Linting: flake8 integration to catch syntax errors after edits
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .swe_env_wrapper import SWEBenchEnv


# File history for undo support (per-path)
_file_history: dict[str, list[str]] = {}

# Constants for tool behavior
SNIPPET_LINES = 4


class WindowExpander:
    """
    Expand viewports to include whole functions, classes, etc.
    
    Ported from SWE-agent's tools/edit_anthropic/bin/str_replace_editor.
    """
    
    def __init__(self, suffix: str = ""):
        self.suffix = suffix
        if self.suffix and not self.suffix.startswith("."):
            self.suffix = "." + self.suffix
    
    def _find_breakpoints(
        self, lines: list[str], current_line: int, direction: int = 1, max_added_lines: int = 30
    ) -> int:
        """
        Find good breakpoint for viewport expansion.
        
        Args:
            lines: All lines of the file
            current_line: 1-based line number
            direction: 1 for down, -1 for up
            max_added_lines: Maximum lines to extend
            
        Returns:
            1-based line number of breakpoint (inclusive)
        """
        if not (1 <= current_line <= len(lines)):
            return current_line
        
        if direction == 1:
            if current_line == len(lines):
                return current_line
            iter_lines = range(current_line, 1 + min(current_line + max_added_lines, len(lines)))
        else:  # direction == -1
            if current_line == 1:
                return current_line
            iter_lines = range(current_line, -1 + max(current_line - max_added_lines, 1), -1)
        
        best_score = 0
        best_breakpoint = current_line
        
        for i_line in iter_lines:
            line = lines[i_line - 1]
            next_line = None
            if i_line + direction in iter_lines:
                next_line = lines[i_line + direction - 1]
            
            score = 0
            if line == "":
                score = 1
                if next_line == "":
                    score = 2  # Double blank line
            
            # Python-specific: function/class definitions
            if self.suffix == ".py" and any(
                re.match(regex, line) for regex in [r"^\s*def\s+", r"^\s*class\s+", r"^\s*@"]
            ):
                score = 3
            
            if score > best_score:
                best_score = score
                best_breakpoint = i_line
                if direction == 1 and i_line != current_line:
                    best_breakpoint -= 1
            
            # File boundaries are good breakpoints
            if i_line == 1 or i_line == len(lines):
                if 3 > best_score:
                    best_score = 3
                    best_breakpoint = i_line
        
        # Don't shrink the viewport
        if direction == 1 and best_breakpoint < current_line:
            return current_line
        if direction == -1 and best_breakpoint > current_line:
            return current_line
        
        return best_breakpoint
    
    def expand_window(
        self, lines: list[str], start: int, stop: int, max_added_lines: int = 30
    ) -> tuple[int, int]:
        """
        Expand viewport to include whole functions/classes.
        
        Args:
            lines: All lines of the file
            start: 1-based start line
            stop: 1-based end line (inclusive)
            max_added_lines: Max lines to add on each side
            
        Returns:
            Tuple of (new_start, new_stop), both 1-based and inclusive
        """
        if not lines or max_added_lines <= 0:
            return start, stop
        
        start = max(1, min(start, len(lines)))
        stop = max(start, min(stop, len(lines)))
        
        new_start = self._find_breakpoints(lines, start, direction=-1, max_added_lines=max_added_lines)
        new_stop = self._find_breakpoints(lines, stop, direction=1, max_added_lines=max_added_lines)
        
        return new_start, new_stop


def _run_flake8(env: "SWEBenchEnv", path: str) -> str:
    """Run flake8 on a Python file in the container."""
    if not path.endswith(".py"):
        return ""
    # Focus on serious errors: undefined names, syntax errors
    cmd = f"flake8 --isolated --select=F821,F822,F831,E111,E112,E113,E999,E902 '{path}' 2>/dev/null || true"
    return env.communicate(cmd, timeout=30)


def _format_lint_errors(lint_output: str) -> str:
    """Format flake8 output for display."""
    if not lint_output.strip():
        return ""
    lines = []
    for line in lint_output.strip().split("\n"):
        if line.strip():
            # Extract just the error message
            parts = line.split(": ", 1)
            if len(parts) == 2:
                lines.append(f"- {parts[1]}")
            else:
                lines.append(f"- {line}")
    return "\n".join(lines)


def create_tools(env: "SWEBenchEnv") -> list[Callable]:
    """
    Create tools that execute commands in the SWE-bench container.
    
    Matches SWE-agent's default SWE-bench config exactly:
    - bash
    - str_replace_editor
    - submit
    
    Args:
        env: Initialized SWEBenchEnv instance (must call env.start() first)
        
    Returns:
        List of tool functions for the agent
    """
    
    def bash(command: str) -> str:
        """
        Execute a bash command in the container.
        
        Working directory is /testbed (the repository root).
        
        Args:
            command: Bash command to execute
            
        Returns:
            Command output (stdout + stderr combined)
        """
        return env.communicate(command, timeout=300)
    
    def str_replace_editor(
        command: str,
        path: str,
        file_text: Optional[str] = None,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        insert_line: Optional[int] = None,
        view_range: Optional[list[int]] = None,
    ) -> str:
        """
        Custom editing tool for viewing, creating and editing files.
        
        * State is persistent across command calls and discussions with the user
        * If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
        * The `create` command cannot be used if the specified `path` already exists as a file
        * If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
        * The `undo_edit` command will revert the last edit made to the file at `path`
        
        Notes for using the `str_replace` command:
        * The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
        * If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
        * The `new_str` parameter should contain the edited lines that should replace the `old_str`
        
        Args:
            command: The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
            path: Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.
            file_text: Required parameter of `create` command, with the content of the file to be created.
            old_str: Required parameter of `str_replace` command containing the string in `path` to replace.
            new_str: Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
            insert_line: Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
            view_range: Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
            
        Returns:
            Result of the operation
        """
        if command == "view":
            return _view(env, path, view_range)
        elif command == "create":
            if file_text is None:
                return "Error: file_text is required for create command"
            return _create(env, path, file_text)
        elif command == "str_replace":
            if old_str is None:
                return "Error: old_str is required for str_replace command"
            return _str_replace(env, path, old_str, new_str or "")
        elif command == "insert":
            if new_str is None:
                return "Error: new_str is required for insert command"
            if insert_line is None:
                return "Error: insert_line is required for insert command"
            return _insert(env, path, insert_line, new_str)
        elif command == "undo_edit":
            return _undo_edit(env, path)
        else:
            return f"Error: Unknown command '{command}'. Use: view, create, str_replace, insert, undo_edit"
    
    def submit() -> str:
        """
        Submit the patch for evaluation.
        
        Returns the final git diff that will be evaluated.
        
        Returns:
            Final patch content
        """
        return env.communicate("git -c core.fileMode=false diff HEAD", timeout=30)
    
    # Return all tools as a list - EXACTLY matching SWE-agent's SWE-bench config
    return [
        bash,
        str_replace_editor,
        submit,
    ]


def _view(env: "SWEBenchEnv", path: str, view_range: Optional[list[int]] = None) -> str:
    """View file or directory contents."""
    # Check if it's a directory
    is_dir = env.communicate(f"test -d '{path}' && echo 'dir'", timeout=10)
    if "dir" in is_dir:
        if view_range:
            return "Error: view_range parameter is not allowed when path points to a directory."
        # List directory up to 2 levels deep (non-hidden files) - matches SWE-agent
        result = env.communicate(f"find '{path}' -maxdepth 2 -not -path '*/\\.*'", timeout=30)
        return f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{result}\n"
    
    # It's a file - read contents
    content = env.communicate(f"cat '{path}'", timeout=30)
    if "No such file" in content or "cat:" in content:
        return f"Error: The path {path} does not exist. Please provide a valid path."
    
    file_lines = content.split("\n")
    n_lines = len(file_lines)
    suffix = Path(path).suffix
    
    if view_range:
        if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
            return "Error: Invalid view_range. It should be a list of two integers."
        
        init_line, final_line = view_range
        
        if init_line < 1 or init_line > n_lines:
            return f"Error: Invalid view_range: {view_range}. First element {init_line} should be within [1, {n_lines}]"
        if final_line > n_lines:
            return f"Error: Invalid view_range: {view_range}. Second element {final_line} should be <= {n_lines}"
        if final_line != -1 and final_line < init_line:
            return f"Error: Invalid view_range: {view_range}. Second element should be >= first element"
        
        if final_line == -1:
            final_line = n_lines
        
        # Expand viewport to function/class boundaries
        init_line, final_line = WindowExpander(suffix=suffix).expand_window(
            file_lines, init_line, final_line, max_added_lines=30
        )
        
        display_lines = file_lines[init_line - 1:final_line]
    else:
        init_line = 1
        display_lines = file_lines
    
    # Format with line numbers (like cat -n)
    numbered = "\n".join([f"{i + init_line:6}\t{line}" for i, line in enumerate(display_lines)])
    
    return f"Here's the result of running `cat -n` on {path}:\n{numbered}\n"


def _create(env: "SWEBenchEnv", path: str, file_text: str) -> str:
    """Create a new file."""
    # Check if file exists
    check = env.communicate(f"test -f '{path}' && echo 'exists'", timeout=10)
    if "exists" in check:
        return f"Error: File already exists at {path}. Cannot overwrite files using command `create`."
    
    # Check if parent directory exists
    parent_dir = str(Path(path).parent)
    parent_check = env.communicate(f"test -d '{parent_dir}' && echo 'exists'", timeout=10)
    if "exists" not in parent_check:
        return f"Error: The parent directory {parent_dir} does not exist. Please create it first."
    
    env.write_file(path, file_text)
    return f"File created successfully at: {path}"


def _str_replace(env: "SWEBenchEnv", path: str, old_str: str, new_str: str) -> str:
    """Replace old_str with new_str in the file (must be unique match)."""
    global _file_history
    
    # Read current content
    content = env.communicate(f"cat '{path}'", timeout=30)
    if "No such file" in content or "cat:" in content:
        return f"Error: The path {path} does not exist. Please provide a valid path."
    
    # Normalize tabs
    content = content.expandtabs()
    old_str = old_str.expandtabs()
    new_str = new_str.expandtabs()
    
    # Check uniqueness
    occurrences = content.count(old_str)
    if occurrences == 0:
        return f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
    elif occurrences > 1:
        # Find which lines contain the string
        file_content_lines = content.split("\n")
        lines = [idx + 1 for idx, line in enumerate(file_content_lines) if old_str in line]
        return f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique"
    
    if old_str == new_str:
        return f"No replacement was performed, old_str `{old_str}` is the same as new_str `{new_str}`."
    
    # Run linter before edit
    pre_edit_lint = _run_flake8(env, path)
    
    # Save to history for undo
    if path not in _file_history:
        _file_history[path] = []
    _file_history[path].append(content)
    
    # Perform replacement
    new_content = content.replace(old_str, new_str)
    env.write_file(path, new_content)
    
    # Run linter after edit
    post_edit_lint = _run_flake8(env, path)
    
    # Calculate snippet window
    suffix = Path(path).suffix
    replacement_line = content.split(old_str)[0].count("\n")
    start_line = max(1, replacement_line - SNIPPET_LINES + 1)
    end_line = min(replacement_line + SNIPPET_LINES + new_str.count("\n") + 1, len(new_content.splitlines()))
    
    # Expand window to function/class boundaries
    new_lines = new_content.splitlines()
    start_line, end_line = WindowExpander(suffix=suffix).expand_window(
        new_lines, start_line, end_line, max_added_lines=10
    )
    
    snippet_lines_content = new_lines[start_line-1:end_line]
    numbered = "\n".join([f"{i + start_line:6}\t{line}" for i, line in enumerate(snippet_lines_content)])
    
    result = f"The file {path} has been edited. Here's the result of running `cat -n` on a snippet of {path}:\n{numbered}\nReview the changes and make sure they are as expected. Edit the file again if necessary."
    
    # Add lint warnings if new errors appeared
    if post_edit_lint and post_edit_lint != pre_edit_lint:
        errors = _format_lint_errors(post_edit_lint)
        if errors:
            result += f"\n\n<NOTE>Your edits introduced linter warnings:</NOTE>\n{errors}\nPlease review and fix if needed."
    
    return result


def _insert(env: "SWEBenchEnv", path: str, insert_line: int, new_str: str) -> str:
    """Insert new_str after insert_line."""
    global _file_history
    
    # Read current content
    content = env.communicate(f"cat '{path}'", timeout=30)
    if "No such file" in content or "cat:" in content:
        return f"Error: The path {path} does not exist. Please provide a valid path."
    
    # Normalize tabs (matching SWE-agent)
    content = content.expandtabs()
    new_str = new_str.expandtabs()
    
    lines = content.split("\n")
    n_lines = len(lines)
    
    if insert_line < 0 or insert_line > n_lines:
        return f"Error: Invalid insert_line parameter: {insert_line}. It should be within the range [0, {n_lines}]"
    
    # Save to history
    if path not in _file_history:
        _file_history[path] = []
    _file_history[path].append(content)
    
    # Insert the new content
    new_str_lines = new_str.split("\n")
    result_lines = lines[:insert_line] + new_str_lines + lines[insert_line:]
    new_content = "\n".join(result_lines)
    
    env.write_file(path, new_content)
    
    # Show snippet with context
    start_line = max(1, insert_line - SNIPPET_LINES + 1)
    end_line = min(insert_line + len(new_str_lines) + SNIPPET_LINES, len(result_lines))
    
    snippet = result_lines[start_line-1:end_line]
    numbered = "\n".join([f"{i + start_line:6}\t{line}" for i, line in enumerate(snippet)])
    
    return f"The file {path} has been edited. Here's the result of running `cat -n` on a snippet of {path}:\n{numbered}\nReview the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."


def _undo_edit(env: "SWEBenchEnv", path: str) -> str:
    """Undo the last edit to a file."""
    global _file_history
    
    if path not in _file_history or not _file_history[path]:
        return f"Error: No edit history for {path}"
    
    # Pop the last version and restore
    previous_content = _file_history[path].pop()
    env.write_file(path, previous_content)
    
    return f"Last edit to {path} undone successfully. The file has been reverted to its previous state."


def get_tool_descriptions() -> str:
    """
    Get formatted descriptions of all available tools.
    
    Returns:
        Markdown-formatted tool documentation
    """
    return """
## Available Tools

You have the following tools to solve this issue:

### bash(command: str) -> str
Execute any bash command in the container. Working directory is /testbed (repo root).

### str_replace_editor(command, path, [file_text], [old_str], [new_str], [insert_line], [view_range])
Custom editing tool for viewing, creating and editing files.

* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Commands:
- **view**: Display file contents. Use view_range=[start, end] for specific lines.
- **create**: Create a new file. Requires file_text parameter.
- **str_replace**: Replace old_str with new_str. old_str must match exactly ONCE in the file.
- **insert**: Insert new_str after insert_line.
- **undo_edit**: Revert the last edit to the file.

Notes for str_replace:
- old_str must match EXACTLY one or more consecutive lines (be mindful of whitespace!)
- If old_str is not unique, include more surrounding context to make it unique
- new_str contains the edited lines that replace old_str

### submit() -> str
Submit your changes. Returns the final git diff for evaluation.
"""
