#!/usr/bin/env python3
"""
Run BFCL v4 function calling tests against OminiX-API.

Usage:
    python scripts/bfcl_test.py [--categories simple_python,multiple] [--limit 20]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

API_BASE = "http://localhost:8080/v1"
BFCL_DATA = Path("/Users/yuechen/home/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data")

CATEGORIES = {
    "simple_python": ("BFCL_v4_simple_python.json", "possible_answer/BFCL_v4_simple_python.json"),
    "multiple": ("BFCL_v4_multiple.json", "possible_answer/BFCL_v4_multiple.json"),
    "parallel": ("BFCL_v4_parallel.json", "possible_answer/BFCL_v4_parallel.json"),
    "parallel_multiple": ("BFCL_v4_parallel_multiple.json", "possible_answer/BFCL_v4_parallel_multiple.json"),
    "irrelevance": ("BFCL_v4_irrelevance.json", None),
}


def load_data(category: str):
    test_file, ans_file = CATEGORIES[category]
    with open(BFCL_DATA / test_file) as f:
        data = [json.loads(line) for line in f]
    answers = None
    if ans_file:
        with open(BFCL_DATA / ans_file) as f:
            answers = [json.loads(line) for line in f]
    return data, answers


def convert_function_to_tool(func_def: dict) -> dict:
    params = json.loads(json.dumps(func_def.get("parameters", {})))

    def fix_types(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "dict":
                obj["type"] = "object"
            for v in obj.values():
                fix_types(v)
        elif isinstance(obj, list):
            for item in obj:
                fix_types(item)

    fix_types(params)
    return {
        "type": "function",
        "function": {
            "name": func_def["name"],
            "description": func_def.get("description", ""),
            "parameters": params,
        },
    }


def call_api(messages: list, tools: list) -> dict:
    resp = requests.post(
        f"{API_BASE}/chat/completions",
        json={
            "model": "minicpm-sala-9b-8bit",
            "messages": messages,
            "tools": tools,
            "temperature": 0.01,
            "max_tokens": 1024,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def extract_tool_calls(response: dict) -> list:
    """Extract [{name, arguments}] from API response."""
    choices = response.get("choices", [])
    if not choices:
        return []

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    result = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        result.append({name: args})

    # Fallback: content might be a bare JSON tool call
    if not result and message.get("content"):
        content = message["content"].strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "name" in parsed:
                args = parsed.get("arguments", parsed.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args)
                result.append({parsed["name"]: args})
        except (json.JSONDecodeError, TypeError):
            pass

    return result


def normalize(v):
    """Normalize value for comparison."""
    if isinstance(v, str):
        s = v.strip()
        # Try numeric conversion
        try:
            n = int(s)
            return n
        except ValueError:
            pass
        try:
            n = float(s)
            if n == int(n):
                return int(n)
            return n
        except ValueError:
            pass
        return s
    if isinstance(v, float) and v == int(v):
        return int(v)
    if isinstance(v, list):
        return [normalize(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize(val) for k, val in v.items()}
    return v


def check_one_call(predicted_call: dict, ground_truth_call: dict) -> bool:
    """
    Check one predicted call against one ground truth call.

    predicted_call: {"func_name": {"param": value, ...}}
    ground_truth_call: {"func_name": {"param": [acceptable_values], ...}}
    """
    if len(predicted_call) != 1 or len(ground_truth_call) != 1:
        return False

    pred_name = list(predicted_call.keys())[0]
    gt_name = list(ground_truth_call.keys())[0]

    if pred_name != gt_name:
        return False

    pred_args = predicted_call[pred_name]
    gt_args = ground_truth_call[gt_name]

    # Check each expected parameter
    for param, acceptable_values in gt_args.items():
        if param not in pred_args:
            # Check if all acceptable values include empty/None (optional param)
            norm_acceptable = [normalize(av) for av in acceptable_values]
            if "" in norm_acceptable or None in norm_acceptable:
                continue  # Optional param not provided is OK
            return False

        pred_val = normalize(pred_args[param])

        # acceptable_values is a list of possible correct values
        matched = False
        for av in acceptable_values:
            if normalize(av) == pred_val:
                matched = True
                break
            # Also try string comparison
            if str(normalize(av)) == str(pred_val):
                matched = True
                break
        if not matched:
            return False

    return True


def evaluate(predicted: list, ground_truth: list) -> bool:
    """
    Check if predicted tool calls match ground truth.

    predicted: [{"func_name": {args}}, ...]
    ground_truth: [{"func_name": {param: [acceptable_values]}}, ...]
    """
    if len(predicted) != len(ground_truth):
        return False

    # Try to match each predicted call with a ground truth call
    gt_remaining = list(range(len(ground_truth)))
    for pred in predicted:
        found = False
        for gi in gt_remaining:
            if check_one_call(pred, ground_truth[gi]):
                gt_remaining.remove(gi)
                found = True
                break
        if not found:
            return False
    return True


def run_test(category: str, limit: int = None):
    print(f"\n{'='*60}")
    print(f"  BFCL v4 Test: {category}")
    print(f"{'='*60}")

    data, answers = load_data(category)
    if limit:
        data = data[:limit]
        if answers:
            answers = answers[:limit]

    total = len(data)
    print(f"  Samples: {total}")

    correct = 0
    errors = 0

    for i, entry in enumerate(data):
        test_id = entry.get("id", f"{category}_{i}")
        messages = entry["question"][0]
        functions = entry["function"]
        tools = [convert_function_to_tool(f) for f in functions]

        try:
            start = time.time()
            response = call_api(messages, tools)
            latency = time.time() - start

            predicted = extract_tool_calls(response)

            if category == "irrelevance":
                # Model should NOT call any tool
                is_correct = len(predicted) == 0
            elif answers and i < len(answers):
                gt = answers[i].get("ground_truth", [])
                is_correct = evaluate(predicted, gt)
            else:
                is_correct = None

            if is_correct:
                correct += 1

            if not is_correct and is_correct is not None:
                gt = answers[i].get("ground_truth", []) if answers else []
                pred_str = json.dumps(predicted, default=str)[:140]
                gt_str = json.dumps(gt, default=str)[:140]
                print(f"  [{i+1:3d}/{total}] FAIL {test_id}")
                print(f"           Got:      {pred_str}")
                print(f"           Expected: {gt_str}")

            if (i + 1) % 20 == 0 or i == total - 1:
                pct = correct / (i + 1) * 100
                print(f"  [{i+1:3d}/{total}] {correct}/{i+1} correct ({pct:.1f}%) [{latency:.1f}s/req]")

        except Exception as e:
            errors += 1
            print(f"  [{i+1:3d}/{total}] ERROR {test_id}: {e}")

    evaluated = total - errors
    accuracy = correct / evaluated * 100 if evaluated > 0 else 0
    print(f"\n  {category}: {correct}/{evaluated} ({accuracy:.1f}%)")
    return {"category": category, "correct": correct, "evaluated": evaluated, "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", default="simple_python,multiple")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",")]

    # Check server
    try:
        requests.get(f"{API_BASE}/models/status", timeout=5)
    except requests.ConnectionError:
        print("ERROR: Server not running at localhost:8080")
        sys.exit(1)

    print(f"BFCL v4 Function Calling Benchmark")
    print(f"Categories: {', '.join(categories)}")

    results = []
    for cat in categories:
        if cat not in CATEGORIES:
            print(f"Unknown category: {cat}")
            continue
        results.append(run_test(cat, args.limit))

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    total_c = sum(r["correct"] for r in results)
    total_e = sum(r["evaluated"] for r in results)
    for r in results:
        print(f"  {r['category']:20s}: {r['correct']:3d}/{r['evaluated']:3d} ({r['accuracy']:.1f}%)")
    if total_e > 0:
        print(f"  {'OVERALL':20s}: {total_c:3d}/{total_e:3d} ({total_c/total_e*100:.1f}%)")


if __name__ == "__main__":
    main()
