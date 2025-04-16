import os
import sys
import torch
import argparse
import json
from typing import List, Dict, Any, Optional
import subprocess
from tempfile import NamedTemporaryFile

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from gpt2_local_loader import load_gpt_model, gpt_generate

PYTHON_TEST_PROMPTS = [
    {
        "name": "fibonacci",
        "prompt": "Write a Python function to generate the first n Fibonacci numbers.",
        "expected_elements": ["def", "fibonacci", "for", "yield", "return"],
        "test_case": "assert fibonacci(5) == [0, 1, 1, 2, 3] or fibonacci(5) == [1, 1, 2, 3, 5]"
    },
    {
        "name": "quicksort",
        "prompt": "Implement the quicksort algorithm in Python.",
        "expected_elements": ["def", "quicksort", "pivot", "if", "return"],
        "test_case": "assert quicksort([3, 1, 4, 1, 5, 9, 2, 6, 5]) == [1, 1, 2, 3, 4, 5, 5, 6, 9]"
    },
    {
        "name": "binary_search",
        "prompt": "Write a Python function to perform binary search on a sorted list.",
        "expected_elements": ["def", "binary_search", "mid", "if", "return"],
        "test_case": "assert binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9], 5) == 4"
    },
    {
        "name": "list_comprehension",
        "prompt": "Create a list of squares of even numbers from 1 to 20 using list comprehension.",
        "expected_elements": ["[", "for", "if", "**2", "]"],
        "test_case": "assert result == [4, 16, 36, 64, 100, 144, 196, 256, 324, 400]"
    },
    {
        "name": "class_definition",
        "prompt": "Create a Python class for a simple bank account with deposit and withdraw methods.",
        "expected_elements": ["class", "BankAccount", "def", "__init__", "self", "deposit", "withdraw"],
        "test_case": """
account = BankAccount(100)
account.deposit(50)
account.withdraw(30)
assert account.balance == 120
"""
    },
]

def evaluate_code_syntax(code: str) -> bool:
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False

def execute_code_with_test(code: str, test_case: str) -> Dict[str, Any]:
    result = {"success": False, "error": None}
    with NamedTemporaryFile(suffix='.py', delete=False) as f:
        temp_file = f.name
        full_code = code + "\n\ntry:\n    " + test_case + "\n    print('Test passed!')\nexcept Exception as e:\n    print(f'Test failed: {e}')"
        f.write(full_code.encode('utf-8'))
    try:
        process = subprocess.run([sys.executable, temp_file], capture_output=True, text=True, timeout=5)
        result["stdout"] = process.stdout
        result["stderr"] = process.stderr
        result["success"] = process.returncode == 0 and "Test passed!" in process.stdout
    except subprocess.TimeoutExpired:
        result["error"] = "Execution timed out (5 seconds)"
    except Exception as e:
        result["error"] = str(e)
    finally:
        os.unlink(temp_file)
    return result

def check_expected_elements(code: str, expected_elements: List[str]) -> Dict[str, Any]:
    result = {"has_all_elements": True, "missing_elements": []}
    for element in expected_elements:
        if element not in code:
            result["has_all_elements"] = False
            result["missing_elements"].append(element)
    return result

def generate_html(results: Dict[str, Any], output_path: str):
    html = """
    <html>
    <head>
        <style>
            body { monospace: Arial; margin: 20px; }
            .test { border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; }
            .passed { color: green; }
            .failed { color: red; }
            pre { background: #f8f8f8; padding: 10px; overflow-x: auto; }
            .side-by-side { display: flex; gap: 20px; }
            .block { flex: 1; }
        </style>
    </head>
    <body>
        <h1>Model Comparison Results</h1>
        <p><b>Baseline:</b> {baseline} | <b>Fine-tuned:</b> {finetuned}</p>
        <p><b>Tests Passed:</b> {ft_passed}/{total} (Fine-tuned)</p>
        <hr/>
    """.format(
        baseline=results["baseline"],
        finetuned=results["finetuned"],
        ft_passed=results["summary"]["finetuned_passed"],
        total=results["summary"]["total_tests"]
    )

    for test in results["tests"]:
        status = "passed" if test["finetuned"]["execution"]["success"] else "failed"
        html += f"<div class='test'>"
        html += f"<h2>{test['name'].title()}</h2>"
        html += f"<p><b>Prompt:</b> {test['prompt']}</p>"
        html += "<div class='side-by-side'>"
        html += "<div class='block'><h3>Baseline</h3><pre>{}</pre></div>".format(test["baseline"]["code"])
        html += "<div class='block'><h3>Fine-Tuned</h3><pre>{}</pre></div>".format(test["finetuned"]["code"])
        html += "</div>"
        html += f"<p class='{status}'><b>Test Result:</b> {status.upper()}</p>"
        html += "</div>"

    html += "</body></html>"

    with open(output_path, "w") as f:
        f.write(html)

    print(f"\n📄 HTML view saved to {output_path}")

def compare_models(
    baseline_model_size: str,
    baseline_weights_dir: Optional[str],
    finetuned_model_size: str,
    finetuned_weights_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    output_file: str = "comparison_results.json",
    max_tokens: int = 300,
    temperature: float = 0.7,
    device: Optional[str] = None
):
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load baseline model from weights_dir (raw GPT-2)
    print(f"\n🧠 Loading baseline model ({baseline_model_size})...")
    base_model, base_tokenizer = load_gpt_model(
        model_size=baseline_model_size,
        weights_dir=baseline_weights_dir,
        checkpoint_path=None,
        device=device
    )

    # ✅ Load finetuned model from .pth
    print(f"🎯 Loading fine-tuned model ({finetuned_model_size})...")
    ft_model, ft_tokenizer = load_gpt_model(
        model_size=finetuned_model_size,
        weights_dir=finetuned_weights_dir,
        checkpoint_path=checkpoint_path,
        device=device
    )

    results = {
        "baseline": baseline_model_size,
        "finetuned": finetuned_model_size,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tests": []
    }

    passed_baseline = 0
    passed_finetuned = 0

    for test_case in PYTHON_TEST_PROMPTS:
        prompt = f"# Python Code\n# Task: {test_case['prompt']}\n\n"
        print(f"\n🔍 Generating for: {test_case['name']}")
        print("🧠 Baseline model generating...")
        base_code = gpt_generate(
    model=base_model,
    tokenizer=base_tokenizer,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=temperature,
    device=device
)
        print(base_code)
        base_result = {
            "syntax_valid": evaluate_code_syntax(base_code),
            "element_check": check_expected_elements(base_code, test_case["expected_elements"]),
            "execution": execute_code_with_test(base_code, test_case["test_case"]),
            "code": base_code
        }
        if base_result["execution"]["success"]:
            passed_baseline += 1
        print("🎯 Fine-tuned model generating...")
        ft_code = gpt_generate(
    model=ft_model,
    tokenizer=ft_tokenizer,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=temperature,
    device=device
)
        
        print(ft_code)
        ft_result = {
            "syntax_valid": evaluate_code_syntax(ft_code),
            "element_check": check_expected_elements(ft_code, test_case["expected_elements"]),
            "execution": execute_code_with_test(ft_code, test_case["test_case"]),
            "code": ft_code
        }
        if ft_result["execution"]["success"]:
            passed_finetuned += 1

        results["tests"].append({
            "name": test_case["name"],
            "prompt": test_case["prompt"],
            "baseline": base_result,
            "finetuned": ft_result,
            "improved": not base_result["execution"]["success"] and ft_result["execution"]["success"]
        })

    results["summary"] = {
        "baseline_passed": passed_baseline,
        "finetuned_passed": passed_finetuned,
        "improvements": sum(t["improved"] for t in results["tests"]),
        "total_tests": len(PYTHON_TEST_PROMPTS)
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    generate_html(results, output_file.replace('.json', '.html'))

    print(f"\n📝 Comparison saved to {output_file}")
    print(f"✅ Baseline passed: {passed_baseline}")
    print(f"🚀 Finetuned passed: {passed_finetuned}")
    print(f"⬆️ Improvements: {results['summary']['improvements']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned GPT-2 on Python code generation")
    parser.add_argument("--baseline-model-size", type=str, default="355M")
    parser.add_argument("--baseline-weights-dir", type=str, default="gpt2_weights")
    parser.add_argument("--finetuned-model-size", type=str, default="355M")
    parser.add_argument("--finetuned-weights-dir", type=str, default="checkpoints")
    parser.add_argument("--finetuned-checkpoint", type=str, default=None,
                    help="Path to fine-tuned .pth checkpoint file")
    parser.add_argument("--output-file", type=str, default="comparison_results.json")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    compare_models(
        baseline_model_size=args.baseline_model_size,
        baseline_weights_dir=args.baseline_weights_dir,
        finetuned_model_size=args.finetuned_model_size,
        finetuned_weights_dir=args.finetuned_weights_dir,
        checkpoint_path=args.finetuned_checkpoint,
        output_file=args.output_file,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device
    )
