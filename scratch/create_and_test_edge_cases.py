import os
import sys
import shutil
import subprocess
import pandas as pd

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
sys.path.insert(0, src_dir)

from git_extractor import GitRepositoryExtractor

def run_cmd(cmd: list, cwd: str):
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    if res.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed: {res.stderr}")
    return res.stdout

def create_isolated_test_repo(test_repo_dir: str):
    """Creates a temporary, isolated Git repository with controlled edge-case commits."""
    if os.path.exists(test_repo_dir):
        shutil.rmtree(test_repo_dir)
    os.makedirs(test_repo_dir, exist_ok=True)
    
    # Git init
    run_cmd(["git", "init"], cwd=test_repo_dir)
    run_cmd(["git", "config", "user.name", "Test Developer"], cwd=test_repo_dir)
    run_cmd(["git", "config", "user.email", "test@example.com"], cwd=test_repo_dir)
    
    # 1. Commit 1: Normal source-file modification
    src_dir = os.path.join(test_repo_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    main_py = os.path.join(src_dir, "main.py")
    with open(main_py, "w") as f:
        f.write("def main():\n    print('Hello World')\n")
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "1. Normal source-file creation"], cwd=test_repo_dir)
    
    with open(main_py, "a") as f:
        f.write("def helper():\n    return 42\n")
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "1b. Normal source-file modification"], cwd=test_repo_dir)
    
    # 2. Commit 2: New source file
    utils_py = os.path.join(src_dir, "utils.py")
    with open(utils_py, "w") as f:
        f.write("def util_func():\n    pass\n")
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "2. New source file"], cwd=test_repo_dir)
    
    # 3. Commit 3: Deleted source file
    os.remove(utils_py)
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "3. Deleted source file"], cwd=test_repo_dir)
    
    # 4. Commit 4: File rename with no content change
    core_py = os.path.join(src_dir, "core.py")
    run_cmd(["git", "mv", "src/main.py", "src/core.py"], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "4. File rename with no content change"], cwd=test_repo_dir)
    
    # 5. Commit 5: Binary file addition
    bin_file = os.path.join(test_repo_dir, "logo.bin")
    with open(bin_file, "wb") as f:
        f.write(bytes([0x00, 0xFF, 0xFE, 0xFD, 0x12, 0x34]))
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "5. Binary file addition"], cwd=test_repo_dir)
    
    # 6. Commit 6: Empty commit using --allow-empty
    run_cmd(["git", "commit", "--allow-empty", "-m", "6. Empty commit"], cwd=test_repo_dir)
    
    # 7. Commit 7: Branch merge commit
    run_cmd(["git", "checkout", "-b", "feature-branch"], cwd=test_repo_dir)
    feat_file = os.path.join(src_dir, "feature.py")
    with open(feat_file, "w") as f:
        f.write("# Feature code\n")
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "7a. Feature commit on branch"], cwd=test_repo_dir)
    
    # Checkout master / main and merge
    # Determine default branch name
    branch_res = run_cmd(["git", "branch", "--show-current"], cwd=test_repo_dir).strip()
    # Switch back to previous branch
    run_cmd(["git", "checkout", "HEAD~1"], cwd=test_repo_dir) # or main
    # Actually let's find current branch name before feature branch creation
    
    print("[TestRepo] Temporary repository created with controlled edge cases.")

def run_edge_case_experiment():
    test_repo_dir = os.path.abspath(os.path.join(script_dir, "test_repo"))
    
    if os.path.exists(test_repo_dir):
        shutil.rmtree(test_repo_dir)
    os.makedirs(test_repo_dir, exist_ok=True)
    
    # Initialize Git
    run_cmd(["git", "init"], cwd=test_repo_dir)
    run_cmd(["git", "config", "user.name", "EdgeCase Tester"], cwd=test_repo_dir)
    run_cmd(["git", "config", "user.email", "tester@example.com"], cwd=test_repo_dir)
    
    # Determine default branch name
    default_branch = run_cmd(["git", "branch", "--show-current"], cwd=test_repo_dir).strip() or "master"
    
    # 1. Normal source file modification
    src_dir = os.path.join(test_repo_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    main_py = os.path.join(src_dir, "main.py")
    with open(main_py, "w") as f:
        f.write("def main():\n    print('Hello World')\n")
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "Commit 1: Normal source file creation"], cwd=test_repo_dir)
    
    with open(main_py, "a") as f:
        f.write("def add(a, b):\n    return a + b\n")
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "Commit 2: Normal source file modification"], cwd=test_repo_dir)
    
    # 2. New source file
    utils_py = os.path.join(src_dir, "utils.py")
    with open(utils_py, "w") as f:
        f.write("def util():\n    pass\n")
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "Commit 3: New source file"], cwd=test_repo_dir)
    
    # 3. Deleted source file
    os.remove(utils_py)
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "Commit 4: Deleted source file"], cwd=test_repo_dir)
    
    # 4. File rename with no content change
    run_cmd(["git", "mv", "src/main.py", "src/core.py"], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "Commit 5: Pure file rename"], cwd=test_repo_dir)
    
    # 5. Binary file addition
    bin_file = os.path.join(test_repo_dir, "asset.bin")
    with open(bin_file, "wb") as f:
        f.write(bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]))
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "Commit 6: Binary file addition"], cwd=test_repo_dir)
    
    # 6. Empty commit using --allow-empty
    run_cmd(["git", "commit", "--allow-empty", "-m", "Commit 7: Empty commit"], cwd=test_repo_dir)
    
    # 7. Merge commit setup
    run_cmd(["git", "checkout", "-b", "feature-branch"], cwd=test_repo_dir)
    feat_file = os.path.join(src_dir, "feature.py")
    with open(feat_file, "w") as f:
        f.write("def feature():\n    return True\n")
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "Commit 8: Feature branch commit"], cwd=test_repo_dir)
    
    run_cmd(["git", "checkout", default_branch], cwd=test_repo_dir)
    run_cmd(["git", "merge", "--no-ff", "feature-branch", "-m", "Commit 9: Merge feature branch"], cwd=test_repo_dir)
    
    # 8. Test file modification
    tests_dir = os.path.join(test_repo_dir, "tests")
    os.makedirs(tests_dir, exist_ok=True)
    test_core = os.path.join(tests_dir, "test_core.py")
    with open(test_core, "w") as f:
        f.write("def test_core():\n    assert True\n")
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "Commit 10: Test file modification"], cwd=test_repo_dir)
    
    # 9. Multiple directories touched
    pkg_dir = os.path.join(test_repo_dir, "pkg")
    ui_dir = os.path.join(test_repo_dir, "ui")
    os.makedirs(pkg_dir, exist_ok=True)
    os.makedirs(ui_dir, exist_ok=True)
    with open(os.path.join(pkg_dir, "db.py"), "w") as f:
        f.write("# DB module\n")
    with open(os.path.join(ui_dir, "view.py"), "w") as f:
        f.write("# View module\n")
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "Commit 11: Multi-directory source changes"], cwd=test_repo_dir)
    
    # 10. Multiple files with different churn sizes
    file_small = os.path.join(src_dir, "small.py")
    file_large = os.path.join(src_dir, "large.py")
    with open(file_small, "w") as f:
        f.write("# 5 lines\n" * 5)
    with open(file_large, "w") as f:
        f.write("# 100 lines\n" * 100)
    run_cmd(["git", "add", "."], cwd=test_repo_dir)
    run_cmd(["git", "commit", "-m", "Commit 12: Multi-file different churn sizes"], cwd=test_repo_dir)
    
    # Run Extractor (Without Merges)
    extractor = GitRepositoryExtractor(test_repo_dir)
    df_no_merges = extractor.extract_commit_history(include_merges=False)
    
    # Run Extractor (With Merges)
    df_with_merges = extractor.extract_commit_history(include_merges=True)
    
    print("\n==========================================================================")
    print("        ISOLATED EDGE-CASE VALIDATION EXPERIMENT REPORT                   ")
    print("==========================================================================")
    print(f"Test Repository Location : {test_repo_dir}")
    print(f"Total Commits Extracted (include_merges=False) : {len(df_no_merges)}")
    print(f"Total Commits Extracted (include_merges=True)  : {len(df_with_merges)}\n")
    
    cols_to_show = [
        'commit_hash', 'lines_added', 'lines_deleted', 'code_churn',
        'files_changed', 'num_directories_touched', 'is_test_file_modified',
        'avg_lines_changed_per_file', 'max_lines_changed_in_single_file', 'num_source_files_changed'
    ]
    
    print("--- EXTRACTED COMMIT METRICS (Non-Merge History) ---")
    print(df_no_merges[cols_to_show].to_string(index=False))
    
    print("\n--------------------------------------------------------------------------")
    print("SPECIFIC EDGE CASE VERIFICATIONS:")
    print("--------------------------------------------------------------------------")
    
    # 1. Pure Rename Check
    # Find commit corresponding to "Pure file rename"
    rename_row = df_no_merges[df_no_merges['commit_hash'].isin(
        run_cmd(["git", "log", "--grep=Pure file rename", "--format=%h"], cwd=test_repo_dir).splitlines()
    )]
    if not rename_row.empty:
        rename_churn = rename_row['code_churn'].values[0]
        print(f"1. Pure Rename Churn Check         : code_churn = {rename_churn} (EXPECTED: 0) -> PASSED={rename_churn==0}")
    
    # 2. Binary File Check
    bin_row = df_no_merges[df_no_merges['commit_hash'].isin(
        run_cmd(["git", "log", "--grep=Binary file addition", "--format=%h"], cwd=test_repo_dir).splitlines()
    )]
    if not bin_row.empty:
        files_c = bin_row['files_changed'].values[0]
        churn_c = bin_row['code_churn'].values[0]
        print(f"2. Binary File Extraction Check    : files_changed = {files_c}, code_churn = {churn_c} -> PASSED (Didn't crash)")
        
    # 3. Empty Commit Check
    empty_row = df_no_merges[df_no_merges['commit_hash'].isin(
        run_cmd(["git", "log", "--grep=Empty commit", "--format=%h"], cwd=test_repo_dir).splitlines()
    )]
    if not empty_row.empty:
        empty_churn = empty_row['code_churn'].values[0]
        empty_files = empty_row['files_changed'].values[0]
        print(f"3. Empty Commit Safety Check       : code_churn = {empty_churn}, files_changed = {empty_files} -> PASSED={empty_churn==0 and empty_files==0}")
        
    # 4. Merge Commit Behavior Check
    skipped_merges = len(df_with_merges) - len(df_no_merges)
    print(f"4. Merge Commit Filtering Check    : Skipped {skipped_merges} merge commit(s) when include_merges=False -> PASSED={skipped_merges==1}")
    
    # 5. Test File Detection Check
    test_row = df_no_merges[df_no_merges['commit_hash'].isin(
        run_cmd(["git", "log", "--grep=Test file modification", "--format=%h"], cwd=test_repo_dir).splitlines()
    )]
    if not test_row.empty:
        is_test = test_row['is_test_file_modified'].values[0]
        print(f"5. Test File Detection Check       : is_test_file_modified = {is_test} -> PASSED={is_test==1}")
        
    # 6. Multi-Directory Count Check
    dir_row = df_no_merges[df_no_merges['commit_hash'].isin(
        run_cmd(["git", "log", "--grep=Multi-directory", "--format=%h"], cwd=test_repo_dir).splitlines()
    )]
    if not dir_row.empty:
        dirs_touched = dir_row['num_directories_touched'].values[0]
        print(f"6. Multi-Directory Count Check     : num_directories_touched = {dirs_touched} (EXPECTED: 2) -> PASSED={dirs_touched==2}")
        
    # 7. Avg & Max Lines Churn Check
    multi_file_row = df_no_merges[df_no_merges['commit_hash'].isin(
        run_cmd(["git", "log", "--grep=Multi-file different churn", "--format=%h"], cwd=test_repo_dir).splitlines()
    )]
    if not multi_file_row.empty:
        avg_lines = multi_file_row['avg_lines_changed_per_file'].values[0]
        max_lines = multi_file_row['max_lines_changed_in_single_file'].values[0]
        print(f"7. Avg & Max Lines Churn Check     : avg_lines = {avg_lines}, max_lines = {max_lines} (EXPECTED max: 100) -> PASSED={max_lines==100}")
        
    # 8. Chronological Order Check
    chronological = df_no_merges['commit_timestamp'].is_monotonic_increasing
    print(f"8. Chronological Order Verification : Timestamps strictly sorted = {chronological} -> PASSED={chronological}")
    print("==========================================================================\n")

if __name__ == '__main__':
    run_edge_case_experiment()
