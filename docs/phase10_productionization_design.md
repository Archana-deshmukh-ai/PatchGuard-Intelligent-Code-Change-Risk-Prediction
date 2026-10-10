# PatchGuard Phase 10 — Productionization Architecture and Design Proposal

**Project**: PatchGuard — Intelligent Code Change Risk Prediction
**Milestone**: Phase 10.1: Architecture Inspection & Productionization Design
**Status**: DESIGN AND INSPECTION ONLY (No Code or Dependency Modifications)
**Date**: October 10, 2026
**Repository Working Directory**: `C:\Projects\CODE_project\bug-prediction`
**GitHub Repository**: `https://github.com/Archana-deshmukh-ai/PatchGuard-Intelligent-Code-Change-Risk-Prediction.git`

---

## 1. Current-State Checkpoint

Prior to designing Phase 10, the repository state was independently inspected and verified:

* **Current Git Branch**: `main`
* **Commit HEAD SHA**: `0e142ca204196c07f7276501bfbf6bd03650eb51`
* **Remote Tracking (`origin/main`)**: `0e142ca204196c07f7276501bfbf6bd03650eb51` (0 ahead / 0 behind)
* **Working Tree Modifications**:
  * Modified tracked files: `PROJECT_LOG.md`, `README.md`, `requirements.txt`
  * Untracked directories/files: `src/ui/` (`__init__.py`, `helpers.py`, `benchmark_helpers.py`, `pr_helpers.py`, `app.py`), `tests/test_dashboard.py`
  * Staged files: `0` (clean index)
* **Working Tree Preservation**: All uncommitted Phase 9.1–9.5 changes are preserved intact without any resets, stashes, or reverts.
* **Production Model Artifact**: `models/logistic_regression_real.pkl`
  * SHA-256: `A32B30DFF63B3A85C3FD9D9983921D5F5C5C5629ACC9C206F77759C46F60F387` (Verified 100% byte-identical)
* **Production Scaler Artifact**: `models/real_data_scaler.pkl`
  * SHA-256: `C8E65F768DAB97ACF8DF573DA60F6B1C0E9232EBF2173677F88E3EFA5C479B49` (Verified 100% byte-identical)
* **Production Decision Threshold**: `0.35` (Locked)
* **Canonical Feature Schema**: `v1.0-10features` (Exact 10 features in mandatory ordering)
* **Active Runtime Environment**: Python 3.12.6 on Windows AMD64
* **Existing Installed Packages Relevant to API / Containerization**:
  * `pydantic` 2.13.5 (installed)
  * `starlette` 1.7.0 (installed)
  * `uvicorn` 0.54.0 (installed)
  * `requests` 2.32.5 (installed)
  * `scikit-learn` 1.8.0, `numpy` 1.26.4, `pandas` 2.3.3, `joblib` 1.5.3 (installed)
  * `cryptography` 50.0.1, `PyJWT` 2.15.0 (installed)

---

## 2. Existing Architecture & Public Reuse Boundaries

PatchGuard contains mature, thoroughly tested production components that must be reused directly by the API service layer without duplication:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PATCHGUARD REUSE ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────┘

       [HTTP Client / CI/CD]                    [Terminal Developer]
                 │                                        │
                 ▼                                        ▼
   ┌───────────────────────────┐            ┌───────────────────────────┐
   │  Proposed REST API Layer  │            │  Existing CLI (cli.py)    │
   │  (src/api/ - Phase 10.2)  │            │  & Streamlit UI (app.py)  │
   └─────────────┬─────────────┘            └─────────────┬─────────────┘
                 │                                        │
                 │  Reuses Service & Engine               │  Direct Import
                 ▼                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                       CORE BUSINESS LOGIC & ENGINES                     │
   │                                                                         │
   │  1. RiskPredictionEngine (src/engine/predictor.py)                      │
   │     - Model: Logistic Regression (real data)                            │
   │     - Scaler: Train-only StandardScaler                                 │
   │     - Threshold: Locked 0.35                                            │
   │     - Log-odds contributions: beta_i * z_i                              │
   │     - Canonical Schema: 10 features                                     │
   │                                                                         │
   │  2. GitRepositoryExtractor (src/git_extractor.py)                       │
   │     - Subprocess Git parsing, full SHA resolution, commit metrics       │
   │                                                                         │
   │  3. PRAnalyzer (src/analysis/pr_analyzer.py)                            │
   │     - Merge-base diff extraction, constituent commit predictions        │
   │                                                                         │
   │  4. LLMCodeAnalyzer (src/analysis/analyzer.py)                          │
   │     - Mock & OpenAI providers, grounded findings, evidence validator   │
   │                                                                         │
   │  5. GitHubPRWorkflow & Webhooks (src/github/)                           │
   │     - HMAC-SHA256 signature verification, acquisition, reporting        │
   └─────────────────────────────────────────────────────────────────────────┘
```

### Architectural Subsystem Analysis

1. **Inference Engine (`src/engine/predictor.py`)**:
   - `RiskPredictionEngine(models_dir, default_threshold=0.35)` loads and validates model and scaler artifacts.
   - `predict(repo_path, commit_hash="HEAD", threshold=None) -> PredictionResult` encapsulates end-to-end inference, log-odds contribution calculation, presentation labels, and provenance.
   - Reused as a long-lived singleton in the API process; loaded once during application startup.
2. **Domain Exception Hierarchy (`src/engine/exceptions.py`)**:
   - Clean, fine-grained exception hierarchy rooted in `PatchGuardError`.
   - Distinct classes: `RepositoryNotFoundError`, `InvalidGitRepositoryError`, `CommitNotFoundError`, `UnsupportedCommitTypeError` (e.g. merge commits), `EmptyCommitError`, `InvalidFeatureValueError`, `ModelArtifactNotFoundError`, `CorruptedArtifactError`, `IncompatibleSchemaError`.
   - The API error-handling middleware maps each exception directly to appropriate HTTP status codes (400, 403, 404, 422, 500) and structured, sanitized JSON error payloads.
3. **Commit & Repository Resolution**:
   - Handled securely by `GitRepositoryExtractor` using `subprocess.run(["git", "rev-parse", ...], shell=False)`.
   - Resolves any valid Git reference (`HEAD`, tag, branch, short SHA) to a canonical 40-character commit SHA.
4. **Pull Request Analysis (`src/analysis/pr_analyzer.py`)**:
   - Orchestrates multi-commit evaluation without computing a fabricated PR-wide probability score.
   - Preserves per-commit ML defect predictions and evaluates cumulative diffs via `merge-base`.
5. **GitHub Integration & Webhook Verification (`src/github/`)**:
   - `verify_webhook_signature(payload_bytes, signature_header, secret)` verifies HMAC-SHA256 signatures in constant time.
   - `WebhookEventHandler.parse_event()` extracts structured pull request events (`opened`, `synchronize`, `reopened`).
   - `RepositoryAcquisitionManager` acquires commits securely into isolated temporary workspaces.

---

## 3. Recommended API Framework & Technical Rationale

### Framework Comparison

| Evaluation Criterion | **FastAPI** | **Flask** | **Starlette (Pure)** |
| :--- | :--- | :--- | :--- |
| **Existing Dependency Footprint** | `pydantic` (2.13.5), `starlette` (1.7.0), and `uvicorn` (0.54.0) are **already installed**. Only `fastapi>=0.110.0` is required. | `flask` is **not installed**; requires adding `flask`, plus `pydantic` or `marshmallow` for schemas. | `starlette` is already installed, but lacks automated schema validation and OpenAPI generator. |
| **Request & Response Validation** | Native Pydantic v2 schemas; strict type checking, regex validation, and range bounds (`Field(ge=0.01, le=0.99)`). | Requires manual payload validation or third-party wrappers. | Requires manual dictionary parsing and error handling. |
| **Interactive OpenAPI Documentation** | Built-in Swagger UI (`/docs`) and ReDoc (`/redoc`) with zero external tooling. | Requires third-party extensions (e.g. `flasgger`, `flask-restx`). | None built-in; must be hand-authored or custom-served. |
| **ASGI & Concurrency Lifecycle** | Native ASGI lifespan support (`@asynccontextmanager`) for loading and caching the model singleton once at boot. | WSGI application; requires external WSGI server; lifecycle hooks are less flexible. | Native ASGI lifespan support, but lacks route-level schema binding. |
| **Integration Testability** | `fastapi.testclient.TestClient` allows 100% in-process, synchronous test execution without starting external network servers. | `flask.testing.FlaskClient` provides good testing, but lacks schema-driven test harnesses. | `starlette.testclient.TestClient` works well, but lacks Pydantic model integration. |
| **Containerization Standard** | Industry standard with `uvicorn main:app --host 0.0.0.0 --port 8000` or Gunicorn + Uvicorn worker. | Requires Gunicorn/uWSGI container setup. | Similar to FastAPI with Uvicorn. |

### Technical Recommendation
**FastAPI (with Pydantic v2, Starlette, and Uvicorn)** is recommended.
* **Justification**:
  1. The heavy lifting dependencies (`pydantic` 2.13.5, `starlette` 1.7.0, and `uvicorn` 0.54.0) are already verified present in the active Python environment.
  2. Pydantic v2 provides strict input validation (repository name allowlists, commit SHA regex, threshold range bounding), which is vital for securing an inference service.
  3. Automatic OpenAPI documentation generation ensures full contract transparency for CI/CD developers and frontend integrations.
  4. ASGI lifespan support enables clean, thread-safe model caching at startup with zero per-request disk reads or artifact reloads.

---

## 4. Proposed API Endpoint Contracts

### 4.1. `GET /health` (Process Liveness Probe)
* **Purpose**: Basic process health check for container runtimes, Kubernetes liveness probes, and load balancers.
* **Inclusion**: **Included in Phase 10.2**.
* **Request Schema**: No query or body parameters.
* **Response Schema (HTTP 200)**:
  ```json
  {
    "status": "pass",
    "service": "patchguard-api",
    "version": "1.0.0"
  }
  ```
* **Distinction**: Indicates solely that the HTTP worker process is responsive; it **does NOT** imply model artifacts are loaded or ready for inference.

---

### 4.2. `GET /ready` (Model & Dependency Readiness Probe)
* **Purpose**: Verifies that the production Logistic Regression model and StandardScaler artifacts are loaded in memory, SHA-256 hashes match verified constants, and the inference engine is ready to accept prediction requests.
* **Inclusion**: **Included in Phase 10.2**.
* **Request Schema**: No query or body parameters.
* **Validation Rules**:
  * Inspects the cached `RiskPredictionEngine` instance.
  * Verifies artifact SHA-256 hashes match production constants.
  * Does **NOT** retrain, refit, or touch model files on disk.
* **Response Schema (HTTP 200)**:
  ```json
  {
    "status": "ready",
    "model_id": "logistic_regression_real",
    "model_version": "v1.0.0-phase4",
    "feature_schema_version": "v1.0-10features",
    "decision_threshold": 0.35,
    "model_artifact_hash": "a32b30dff63b3a85c3fd9d9983921d5f5c5c5629acc9c206f77759c46f60f387",
    "scaler_artifact_hash": "c8e65f768dab97acf8df573da60f6b1c0e9232ebf2173677f88e3efa5c479b49"
  }
  ```
* **Error Response (HTTP 503 Service Unavailable)**:
  ```json
  {
    "status": "unready",
    "reason": "Model artifacts not loaded or hash mismatch",
    "details": "Model artifact hash mismatch or file missing"
  }
  ```

---

### 4.3. `POST /v1/predictions/commit` (Single Commit Risk Prediction)
* **Purpose**: Predicts defect risk for a specific commit in an allowlisted local Git repository using the production ML model.
* **Inclusion**: **Included in Phase 10.2**.
* **Request Schema**:
  ```json
  {
    "repository": "bottle",
    "commit_hash": "HEAD",
    "threshold": 0.35
  }
  ```
  *Validation Rules*:
  * `repository` (string, required): Repository identifier. Must match regex `^[a-zA-Z0-9_\-\.]+$` and resolve to a directory within the configured `ALLOWED_REPOS_ROOT`. Rejects path traversal tokens (`..`, `/`, `\`).
  * `commit_hash` (string, optional, default `"HEAD"`): Git reference or SHA. Must match regex `^[a-zA-Z0-9_\-\.\/]+$` with max length 100.
  * `threshold` (float, optional, default `0.35`): Decision threshold bounded in range `[0.01, 0.99]`.
* **Response Schema (HTTP 200)**:
  ```json
  {
    "commit": {
      "commit_hash": "588622c",
      "full_hash": "588622c8b5d706c2e02863b8c0b153abfbc5e3c6",
      "repository": "bottle"
    },
    "prediction": {
      "raw_probability": 0.2205,
      "is_above_threshold": false,
      "decision_threshold": 0.35,
      "prediction_label": "Standard Risk Change",
      "risk_level": "LOW",
      "presentation_disclaimer": "Model probabilities represent estimated risk scores and are not calibrated real-world defect probabilities. Risk levels (LOW/MEDIUM/HIGH) are presentation labels only for UI display."
    },
    "model_signals": {
      "top_positive_factors": {
        "max_lines_changed_in_single_file": 0.3682
      },
      "top_negative_factors": {
        "avg_lines_changed_per_file": -0.5545,
        "num_source_files_changed": -0.5346
      },
      "log_odds_contributions": {
        "lines_added": -0.0817,
        "lines_deleted": 0.1169,
        "code_churn": -0.0679,
        "files_changed": 0.0758,
        "functions_changed": -0.2215,
        "num_directories_touched": -0.1426,
        "is_test_file_modified": -0.0877,
        "avg_lines_changed_per_file": -0.5545,
        "max_lines_changed_in_single_file": 0.3682,
        "num_source_files_changed": -0.5346
      }
    },
    "features": {
      "lines_added": 1,
      "lines_deleted": 0,
      "code_churn": 1,
      "files_changed": 1,
      "functions_changed": 1,
      "num_directories_touched": 1,
      "is_test_file_modified": 0,
      "avg_lines_changed_per_file": 1.0,
      "max_lines_changed_in_single_file": 1,
      "num_source_files_changed": 0
    },
    "provenance": {
      "model_id": "logistic_regression_real",
      "model_version": "v1.0.0-phase4",
      "feature_schema_version": "v1.0-10features",
      "prediction_timestamp": "2026-10-10T14:27:06.023560+00:00",
      "model_artifact_hash": "a32b30dff63b3a85c3fd9d9983921d5f5c5c5629acc9c206f77759c46f60f387",
      "scaler_artifact_hash": "c8e65f768dab97acf8df573da60f6b1c0e9232ebf2173677f88e3efa5c479b49"
    }
  }
  ```
* **Standard Error Response Format**:
  ```json
  {
    "error": {
      "code": "COMMIT_NOT_FOUND",
      "message": "Commit 'xyz123' could not be found in repository 'bottle'.",
      "remediation": "Verify the commit hash or revision exists in the repository."
    }
  }
  ```
* **HTTP Status Code Mapping**:
  * `200 OK`: Successful inference.
  * `400 Bad Request`: `UnsupportedCommitTypeError` (e.g. merge commit), `EmptyCommitError` (0 files changed), or invalid numerical threshold.
  * `403 Forbidden`: Repository path outside configured allowlist root.
  * `404 Not Found`: `RepositoryNotFoundError`, `InvalidGitRepositoryError`, or `CommitNotFoundError`.
  * `422 Unprocessable Entity`: Request body validation error (missing required fields, malformed JSON).
  * `500 Internal Server Error`: Unhandled engine exception (sanitized; zero stack trace or internal filesystem path exposed).

---

### 4.4. Deferred Endpoints (Evaluated for Subsequent Releases)
1. **`POST /v1/analysis/commit` (ML + Qualitative LLM Analysis)**:
   - *Status*: **Deferred to Phase 10.5 / post-v1**.
   - *Rationale*: Involves external network egress (OpenAI API), high latency (5–15s), variable token costs, and API key management. Keeping the Phase 10.2 core API strictly focused on fast, deterministic ML inference is essential for predictable performance and local container deployment.
2. **`POST /v1/predictions/pr` (Local Multi-Commit PR Analysis)**:
   - *Status*: **Deferred to Phase 10.5 / post-v1**.
   - *Rationale*: Multi-commit PR analysis iterates over $N$ commits synchronously. Should be introduced only after the single-commit core endpoint is verified and hardened.
3. **`POST /v1/webhooks/github` (Automated GitHub Webhook Ingestion)**:
   - *Status*: **Deferred to Phase 10.5 / post-v1**.
   - *Rationale*: `GitHubPRWorkflow` and HMAC-SHA256 signature verification are already implemented in `src/github/`. Exposing them via REST requires configuring external webhook tunnels or reverse proxies, which belongs in a dedicated CI/CD integration milestone.

---

## 5. Security & Deployment Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SECURITY DEFENSE IN DEPTH                          │
└─────────────────────────────────────────────────────────────────────────────┘

  [Untrusted Client Request]
            │
            ▼
  1. Header Authentication   --> Validate X-API-Key (constant-time hmac.compare_digest)
            │                    (Local-only mode restricts listening to 127.0.0.1)
            ▼
  2. Request Size Guard      --> Limit request body size <= 64 KB
            │
            ▼
  3. Pydantic Schema Guard   --> Regex check on repository & commit_hash; threshold in [0.01, 0.99]
            │
            ▼
  4. Filesystem Boundary     --> os.path.realpath() verification against ALLOWED_REPOS_ROOT
            │                    Zero arbitrary filesystem traversal permitted (CWE-22 / CWE-73)
            ▼
  5. Subprocess Safety       --> Subprocess argument lists only; shell=False; 30s timeout
            │
            ▼
  6. Response Sanitization   --> Structured error payloads; 0 stack traces; 0 secret leakage
```

### Specific Security Controls

1. **Authentication & Authorization**:
   - Configurable API key via `X-API-Key` HTTP header (`PATCHGUARD_API_KEY`).
   - Constant-time verification using `hmac.compare_digest`.
   - If `PATCHGUARD_API_KEY` is not configured, the API logs a warning and binds strictly to `127.0.0.1` (local prototype mode). Public network interfaces (`0.0.0.0`) are permitted only when authentication is explicitly enabled or running inside an isolated container network.
2. **Repository Allowlisting & Path Traversal Prevention**:
   - The service will **NEVER** accept arbitrary absolute filesystem paths from untrusted HTTP clients.
   - All repository targets are resolved relative to a strictly configured root: `PATCHGUARD_REPOS_ROOT` (defaults to `./repositories` or a configurable path).
   - Verification logic:
     ```python
     canonical_root = os.path.realpath(config.allowed_repos_root)
     target_path = os.path.realpath(os.path.join(canonical_root, repository_param))
     if not target_path.startswith(canonical_root + os.sep) and target_path != canonical_root:
         raise ForbiddenRepositoryAccessError("Path traversal prohibited")
     ```
3. **Subprocess & Git Execution Safety**:
   - Zero shell execution (`shell=False` is strictly enforced across all `subprocess.run` calls).
   - All Git commands are invoked with an explicit timeout (e.g. 30 seconds) to prevent Denial of Service from hung locks or recursive operations.
4. **Secret Handling & Error Sanitization**:
   - Internal stack traces and server file paths are completely suppressed in production error responses.
   - Exceptions are mapped to domain-specific error objects with clear user remediation advice.
5. **Concurrency & Thread Safety**:
   - The `RiskPredictionEngine` instance is created once during ASGI lifespan startup.
   - Scikit-learn's `LogisticRegression.predict_proba` and `StandardScaler.transform` are thread-safe for concurrent read-only evaluation.
6. **Container Security & Privileges**:
   - Non-root user: `USER patchguard` (UID 10001).
   - Read-only root filesystem where feasible, with writable `/tmp`.
   - Production model and scaler files mounted or copied with read-only permissions (`0444`).

---

## 6. Service-Layer Structure

To ensure zero coupling between the HTTP transport layer and the core inference engine, the API will be housed in a clean, isolated package under `src/api/`:

```
src/
├── api/
│   ├── __init__.py           # Exports create_app()
│   ├── app.py                # FastAPI app initialization, routes, lifespan context
│   ├── config.py             # APIConfig (pydantic-settings / env vars)
│   ├── schemas.py            # Pydantic v2 request/response models
│   ├── service.py            # PredictionService wrapping RiskPredictionEngine
│   └── error_handlers.py     # Maps PatchGuardError -> HTTP status codes & JSON
├── engine/                   # Untouched, existing prediction engine
├── analysis/                 # Untouched, existing LLM analysis & PR analyzer
├── github/                   # Untouched, existing GitHub integration
├── ui/                       # Untouched, existing Streamlit dashboard
└── cli.py                    # Untouched, existing CLI
```

### Component Responsibilities:
* `config.py`: Loads environment configuration (`PATCHGUARD_API_KEY`, `PATCHGUARD_REPOS_ROOT`, `PATCHGUARD_MODELS_DIR`, `PATCHGUARD_DEFAULT_THRESHOLD`).
* `schemas.py`: Pydantic models for `CommitPredictionRequest`, `CommitPredictionResponse`, `HealthResponse`, `ReadyResponse`, and `ErrorResponse`.
* `service.py`: Provides `PredictionService` which handles repository resolution, allowlist validation, and calls `RiskPredictionEngine.predict()`.
* `error_handlers.py`: Registers FastAPI exception handlers for `PatchGuardError`, `RepositoryError`, `CommitError`, `ModelError`, and unexpected errors.
* `app.py`: Sets up FastAPI routes, mounts middleware, and manages ASGI startup lifespan.

---

## 7. Implementation Sequence (Phases 10.2–10.4)

### Phase 10.2: REST API & Automated Tests
* **Deliverables**:
  1. Add `fastapi>=0.110.0` to `requirements.txt`.
  2. Implement `src/api/` (`config.py`, `schemas.py`, `service.py`, `error_handlers.py`, `app.py`, `__init__.py`).
  3. Create comprehensive test suite `tests/test_api.py` using `fastapi.testclient.TestClient`.
  4. Ensure 100% offline test execution without external network calls.
  5. Update `README.md` with API startup instructions (`uvicorn src.api.app:app --host 0.0.0.0 --port 8000`).

### Phase 10.3: Docker Containerization & Verification
* **Deliverables**:
  1. Author production `Dockerfile` (multi-stage build, Python 3.12 slim, non-root user `patchguard`, Git CLI installed).
  2. Author `.dockerignore` (excluding `.git`, `scratch/`, `__pycache__`, virtual environments, test outputs).
  3. Define `HEALTHCHECK` directive invoking `GET /health`.
  4. Create `docker-compose.yml` for local container testing.
  5. Verify container build and endpoint invocation locally.

### Phase 10.4: CI/CD & Deployment Readiness
* **Deliverables**:
  1. Author GitHub Actions workflow `.github/workflows/ci.yml`.
  2. Define pipeline steps: Linting (`git diff --check`, flake8/ruff), Test discovery & execution (excluding artifact-regenerating tests), SHA-256 hash assertions for model/scaler files, and Docker build verification.
  3. Final synchronization of `README.md` and `PROJECT_LOG.md` marking Phase 10 complete.

---

## 8. Acceptance Criteria & Verification Commands

### Acceptance Criteria for Phase 10.2:
1. `GET /health` returns `200 OK` with JSON `{"status": "pass", "service": "patchguard-api", "version": "1.0.0"}`.
2. `GET /ready` returns `200 OK` and includes SHA-256 hashes matching production constants.
3. `POST /v1/predictions/commit` returns `200 OK` with identical probability and feature values as `src/engine/predictor.py`.
4. Invalid repository names or directory traversal attempts (`../`) return `403 Forbidden` or `400 Bad Request`.
5. Non-existent commits return `404 Not Found` with structured, sanitized JSON error objects.
6. Merge commits return `400 Bad Request` with `UnsupportedCommitTypeError` description.
7. Model artifact hashes remain 100% byte-identical.
8. Test suite passes with 0 failures and 0 errors.

### Verification Commands:
```powershell
# 1. Run API test suite
python -m unittest tests/test_api.py

# 2. Run complete regression suite (excluding benchmark artifact regenerator)
python -m unittest tests/test_dashboard.py tests/test_prediction_engine.py tests/test_analysis_schema.py tests/test_openai_provider.py tests/test_cli.py tests/test_pr_analyzer.py tests/test_diff_extractor.py tests/test_analysis_pipeline.py tests/test_github_integration.py tests/test_github_workflow.py tests/test_repository_acquisition.py

# 3. Verify model artifact SHA-256 hashes
Get-FileHash models/logistic_regression_real.pkl, models/real_data_scaler.pkl -Algorithm SHA256

# 4. Check git diff cleanliness
git diff --check
git status --short --branch
```

---

## 9. Risks, Assumptions & Open Decisions

| Category | Description | Mitigation Strategy |
| :--- | :--- | :--- |
| **Path Traversal Risk** | Malicious client requests arbitrary directories outside permitted repository workspace. | Enforce `ALLOWED_REPOS_ROOT` base path containment using `os.path.realpath` and common-path validation. |
| **Git Timeout Risk** | Hangs on large repositories or corrupt Git packs during subprocess execution. | Enforce 30-second bounded timeout on all subprocess calls; return HTTP 504 on timeout. |
| **Model Invalidation Risk** | Retraining or unintended artifact overwriting during testing. | Enforce read-only model filesystem permissions; assert SHA-256 hashes in CI test preconditions and postconditions. |
| **Class Prevalence Shift** | Temporal class prevalence shift in code changes (from 45% to 20%) over multi-year history. | Prominently display presentation disclaimer and audit metadata in API response; keep threshold locked at 0.35. |
| **Authentication Strategy** | Choice between local-only prototype vs mandatory API key in Phase 10.2. | Require approval from user before finalizing authentication default (see Open Decisions). |

---

## 10. Phase 9 Test Accounting Reconciliation

The Phase 9 final audit reported:
* **Discovered Tests**: 241 unit tests across 12 test files.
* **Executed & Passing Tests**: 231 tests (0 failures, 0 errors).
* **Excluded Tests**: 10 tests in `tests/test_phase8_benchmarking.py`.

### Exact Breakdown:
```
Module                                 Discovered  Executed  Status
-------------------------------------------------------------------
test_analysis_pipeline.py                      24        24  PASS
test_analysis_schema.py                        11        11  PASS
test_cli.py                                     9         9  PASS
test_dashboard.py                              77        77  PASS
test_diff_extractor.py                         17        17  PASS
test_github_integration.py                     23        23  PASS
test_github_workflow.py                        16        16  PASS
test_openai_provider.py                         8         8  PASS
test_pr_analyzer.py                            11        11  PASS
test_prediction_engine.py                      12        12  PASS
test_repository_acquisition.py                 23        23  PASS
-------------------------------------------------------------------
Subtotal (Non-Benchmark Test Suites)          231       231  100% PASS
-------------------------------------------------------------------
test_phase8_benchmarking.py                    10         0  EXCLUDED
  - test_1 to test_8: Safe unit tests
  - test_9: Rewrites reports/experiments/phase8/ on repo root
  - test_10: Safe unit test
-------------------------------------------------------------------
Total Repository Unit Tests                   241       231  PASS (10 EXCLUDED)
```

**Reconciliation Note**:
In previous audit reports, shorthand phrasing referred to "1 benchmark test intentionally excluded" because `test_9_existing_production_artifacts_unmodified` was the specific causal test that runs `runner = Phase8BenchmarkRunner(self.project_root); runner.run_benchmark()` on the live repository, which rewrites `reports/experiments/phase8/`.
Because running `python -m unittest tests/test_phase8_benchmarking.py` would execute `test_9`, the entire module of 10 tests was withheld as a single unit from automated regression runs.
Every single one of the remaining 231 tests in the repository executed and passed cleanly.

---

## 11. Protected Artifact Hashes & Governance Locks

The production artifacts and governance settings remain strictly locked:

* **Production Model**: `models/logistic_regression_real.pkl`
  * SHA-256: `A32B30DFF63B3A85C3FD9D9983921D5F5C5C5629ACC9C206F77759C46F60F387`
* **Production Scaler**: `models/real_data_scaler.pkl`
  * SHA-256: `C8E65F768DAB97ACF8DF573DA60F6B1C0E9232EBF2173677F88E3EFA5C479B49`
* **Production Decision Threshold**: `0.35`
* **Feature Schema**: `v1.0-10features` (Exact 10 features: `lines_added`, `lines_deleted`, `code_churn`, `files_changed`, `functions_changed`, `num_directories_touched`, `is_test_file_modified`, `avg_lines_changed_per_file`, `max_lines_changed_in_single_file`, `num_source_files_changed`)

---

## 12. Exact Next Implementation Step

Upon user approval of this architecture proposal:
* Proceed to **Phase 10.2: Implement REST API and Automated Tests**:
  1. Add `fastapi>=0.110.0` to `requirements.txt`.
  2. Implement `src/api/` (`config.py`, `schemas.py`, `service.py`, `error_handlers.py`, `app.py`, `__init__.py`).
  3. Implement `tests/test_api.py` with comprehensive unit and integration tests.
  4. Verify all tests pass 100% offline while preserving production artifact hashes.
