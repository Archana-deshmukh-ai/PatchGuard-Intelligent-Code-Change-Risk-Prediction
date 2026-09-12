import os
import sys
import shutil
import stat
import subprocess
import pandas as pd

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
sys.path.insert(0, src_dir)

from szz_labeler import SZZDefectLabeler

def remove_readonly(func, path, exc_info):
    """Clear the read-only attribute and retry removal on Windows."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def run_cmd(cmd: list, cwd: str):
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    if res.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed: {res.stderr}")
    return res.stdout

def setup_controlled_szz_test_repo(test_dir: str):
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, onerror=remove_readonly)
    os.makedirs(test_dir, exist_ok=True)

    run_cmd(["git", "init"], cwd=test_dir)
    run_cmd(["git", "config", "user.name", "SZZ Tester"], cwd=test_dir)
    run_cmd(["git", "config", "user.email", "szz@example.com"], cwd=test_dir)

    # 1. Bug-introducing commit 1
    app_py = os.path.join(test_dir, "app.py")
    with open(app_py, "w") as f:
        f.write("def calculate_total(price, tax):\n    return price - tax  # BUG: subtracts tax instead of adding\n")
    run_cmd(["git", "add", "."], cwd=test_dir)
    run_cmd(["git", "commit", "-m", "Initial implementation of calculation logic"], cwd=test_dir)
    intro_commit_1 = run_cmd(["git", "rev-parse", "HEAD"], cwd=test_dir).strip()

    # 2. Unrelated commit (documentation)
    readme = os.path.join(test_dir, "README.md")
    with open(readme, "w") as f:
        f.write("# SZZ Test Repository\n")
    run_cmd(["git", "add", "."], cwd=test_dir)
    run_cmd(["git", "commit", "-m", "Add project README documentation"], cwd=test_dir)

    # 3. Bug-introducing commit 2 (utils.py)
    utils_py = os.path.join(test_dir, "utils.py")
    with open(utils_py, "w") as f:
        f.write("def format_string(s):\n    return s.upper()  # BUG: should be title\n")
    run_cmd(["git", "add", "."], cwd=test_dir)
    run_cmd(["git", "commit", "-m", "Add utility string formatter"], cwd=test_dir)
    intro_commit_2 = run_cmd(["git", "rev-parse", "HEAD"], cwd=test_dir).strip()

    # 4. Pure rename commit
    run_cmd(["git", "mv", "README.md", "DOCUMENTATION.md"], cwd=test_dir)
    run_cmd(["git", "commit", "-m", "Rename README to DOCUMENTATION"], cwd=test_dir)

    # 5. Binary file commit
    bin_file = os.path.join(test_dir, "sample.bin")
    with open(bin_file, "wb") as f:
        f.write(bytes([0x00, 0xFF, 0xFE]))
    run_cmd(["git", "add", "."], cwd=test_dir)
    run_cmd(["git", "commit", "-m", "Add binary file asset"], cwd=test_dir)

    # 6. Formatting-only commit
    with open(app_py, "a") as f:
        f.write("\n# Cosmetic comment addition\n")
    run_cmd(["git", "add", "."], cwd=test_dir)
    run_cmd(["git", "commit", "-m", "Formatting: Add trailing comment"], cwd=test_dir)

    # 7. Test modification commit
    test_py = os.path.join(test_dir, "test_app.py")
    with open(test_py, "w") as f:
        f.write("def test_calc():\n    assert True\n")
    run_cmd(["git", "add", "."], cwd=test_dir)
    run_cmd(["git", "commit", "-m", "Add initial test suite"], cwd=test_dir)

    # 8. Bug-fixing commit 1 (fixes app.py, includes issue ID GH-42 and test update)
    with open(app_py, "w") as f:
        f.write("def calculate_total(price, tax):\n    return price + tax  # FIXED: add tax\n# Cosmetic comment addition\n")
    with open(test_py, "a") as f:
        f.write("    assert calculate_total(100, 10) == 110\n")
    run_cmd(["git", "add", "."], cwd=test_dir)
    run_cmd(["git", "commit", "-m", "Fix GH-42 calculation error when computing total tax"], cwd=test_dir)
    fix_commit_1 = run_cmd(["git", "rev-parse", "HEAD"], cwd=test_dir).strip()

    # 9. Multi-file bug-fixing commit 2 (fixes utils.py bug, references issue #88)
    with open(utils_py, "w") as f:
        f.write("def format_string(s):\n    return s.title()  # FIXED: title case\n")
    run_cmd(["git", "add", "."], cwd=test_dir)
    run_cmd(["git", "commit", "-m", "Fix #88 issue in string formatter"], cwd=test_dir)
    fix_commit_2 = run_cmd(["git", "rev-parse", "HEAD"], cwd=test_dir).strip()

    # 10. Pure Revert commit without fix keyword (e.g. Revert "Add utility string formatter")
    run_cmd(["git", "revert", "--no-edit", "HEAD"], cwd=test_dir)

    return intro_commit_1, intro_commit_2, fix_commit_1, fix_commit_2

def run_szz_controlled_test():
    test_dir = os.path.abspath(os.path.join(script_dir, "test_szz_repo"))
    intro_1, intro_2, fix_1, fix_2 = setup_controlled_szz_test_repo(test_dir)

    labeler = SZZDefectLabeler(test_dir)
    df_labels = labeler.generate_labeled_dataset()

    print("\n==========================================================================")
    print("         CONTROLLED SZZ DEFECT LABELING EXPERIMENT REPORT                 ")
    print("==========================================================================")
    print(f"Test Repository Path : {test_dir}")
    print(f"Total Commits Evaluated : {len(df_labels)}")
    print(f"SZZ-Identified Bug-Introducing Commits (label = 1) : {(df_labels['bug_introduced'] == 1).sum()}")
    print(f"No SZZ Evidence Commits (label = 0)                : {(df_labels['bug_introduced'] == 0).sum()}\n")

    cols_to_print = ['commit_hash', 'bug_introduced', 'label_confidence', 'label_source', 'fixing_commit_hash', 'fixing_issue_id', 'traced_lines_count']
    print("--- EXTRACTED SZZ PROVENANCE DATASET ---")
    print(df_labels[cols_to_print].to_string(index=False))

    print("\n--------------------------------------------------------------------------")
    print("SPECIFIC SZZ VERIFICATION CHECKS:")
    print("--------------------------------------------------------------------------")

    # Check 1: intro_1 (Initial calculation logic) receives bug_introduced = 1
    row_1 = df_labels[df_labels['full_hash'] == intro_1]
    is_intro_1_flagged = not row_1.empty and row_1['bug_introduced'].values[0] == 1
    print(f"1. Traced Bug-Introducing Commit 1 (app.py)   : bug_introduced = {row_1['bug_introduced'].values[0] if not row_1.empty else 'N/A'} -> PASSED={is_intro_1_flagged}")

    # Check 2: intro_2 (utils.py) receives bug_introduced = 1
    row_2 = df_labels[df_labels['full_hash'] == intro_2]
    is_intro_2_flagged = not row_2.empty and row_2['bug_introduced'].values[0] == 1
    print(f"2. Traced Bug-Introducing Commit 2 (utils.py) : bug_introduced = {row_2['bug_introduced'].values[0] if not row_2.empty else 'N/A'} -> PASSED={is_intro_2_flagged}")

    # Check 3: Issue ID preservation
    issue_id_captured = not row_1.empty and "GH-42" in row_1['fixing_issue_id'].values[0]
    print(f"3. Issue ID Provenance Capture (GH-42)        : fixing_issue_id = '{row_1['fixing_issue_id'].values[0] if not row_1.empty else ''}' -> PASSED={issue_id_captured}")

    # Check 4: Label Semantics
    semantics_val = df_labels['label_semantics'].unique()
    print(f"4. Label Semantics Validation                  : Registered labels = {list(semantics_val)}")

    # Check 5: Pure Revert Isolation
    revert_rows = df_labels[df_labels['label_source'] == 'revert_trace']
    print(f"5. Pure Revert Isolation Check                : Revert trace records = {len(revert_rows)}")
    print("==========================================================================\n")

if __name__ == '__main__':
    run_szz_controlled_test()
