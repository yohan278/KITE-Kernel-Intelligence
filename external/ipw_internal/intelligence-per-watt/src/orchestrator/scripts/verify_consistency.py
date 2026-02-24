#!/usr/bin/env python3
"""Verification script for trajectory data consistency.

Verifies that generated trajectories match the format expected by training and inference.

Usage:
    python scripts/verify_consistency.py <trajectory_file.jsonl>
    python scripts/verify_consistency.py data/qwen3_235b_trajectories/successful_trajectories.jsonl
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any


def check_trajectory_format(trajectory: Dict[str, Any]) -> List[str]:
    """Check a single trajectory for format issues.

    Returns:
        List of issues found (empty if valid)
    """
    issues = []

    # Check for conversations/messages key
    if 'conversations' not in trajectory and 'messages' not in trajectory:
        issues.append("Missing 'conversations' or 'messages' key")
        return issues

    messages_key = 'conversations' if 'conversations' in trajectory else 'messages'
    messages = trajectory[messages_key]

    if not isinstance(messages, list):
        issues.append(f"'{messages_key}' is not a list")
        return issues

    if len(messages) < 2:
        issues.append(f"Only {len(messages)} messages (need at least system + user + assistant)")
        return issues

    # Check first message is system prompt
    first_msg = messages[0]
    if first_msg.get('role') != 'system':
        issues.append("First message role is not 'system'")
    else:
        content = first_msg.get('content', '')
        if 'orchestrator' not in content.lower():
            issues.append("System prompt doesn't contain 'orchestrator'")

    # Check for tool usage pattern
    has_tool_usage = False
    for msg in messages:
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if 'TOOL:' in content:
                has_tool_usage = True
                break

    if not has_tool_usage:
        issues.append("No tool usage found (TOOL: syntax)")

    # Check conversation structure (alternating user/assistant/tool after system)
    roles = [msg.get('role') for msg in messages[1:]]  # Skip system message
    for i, role in enumerate(roles):
        if role not in ['user', 'assistant', 'tool']:
            issues.append(f"Invalid role at position {i+1}: {role}")

    return issues


def verify_file(filepath: Path, sample_size: int = 100) -> Dict[str, Any]:
    """Verify a JSONL file of trajectories.

    Args:
        filepath: Path to JSONL file
        sample_size: Number of samples to check (or -1 for all)

    Returns:
        Dict with verification results
    """
    if not filepath.exists():
        return {
            'success': False,
            'error': f"File not found: {filepath}"
        }

    results = {
        'success': True,
        'total_checked': 0,
        'valid': 0,
        'invalid': 0,
        'issues': [],
        'sample_issues': []
    }

    print(f"Verifying: {filepath}")
    print(f"Sample size: {'all' if sample_size == -1 else sample_size}")
    print()

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if sample_size > 0 and i >= sample_size:
                break

            if not line.strip():
                continue

            try:
                trajectory = json.loads(line)
            except json.JSONDecodeError as e:
                results['invalid'] += 1
                results['issues'].append(f"Line {i+1}: JSON decode error: {e}")
                continue

            issues = check_trajectory_format(trajectory)
            results['total_checked'] += 1

            if issues:
                results['invalid'] += 1
                results['sample_issues'].extend([f"Line {i+1}: {issue}" for issue in issues])
                if len(results['sample_issues']) < 10:  # Keep first 10 issues
                    for issue in issues:
                        print(f"  ⚠ Line {i+1}: {issue}")
            else:
                results['valid'] += 1

    return results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filepath = Path(sys.argv[1])
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    results = verify_file(filepath, sample_size)

    if not results['success']:
        print(f"✗ Error: {results['error']}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("Verification Results")
    print("=" * 70)
    print(f"Total checked: {results['total_checked']}")
    print(f"Valid:         {results['valid']} ({results['valid']/results['total_checked']*100:.1f}%)")
    print(f"Invalid:       {results['invalid']} ({results['invalid']/results['total_checked']*100:.1f}%)")

    if results['issues']:
        print()
        print("Critical Issues:")
        for issue in results['issues'][:10]:
            print(f"  ✗ {issue}")

    if results['sample_issues']:
        print()
        print("Sample Issues (first 10):")
        for issue in results['sample_issues'][:10]:
            print(f"  ⚠ {issue}")

    print()
    if results['invalid'] == 0:
        print("✓ All trajectories valid!")
        sys.exit(0)
    else:
        print(f"✗ Found {results['invalid']} invalid trajectories")
        sys.exit(1)


if __name__ == "__main__":
    main()
