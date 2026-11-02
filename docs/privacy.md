# ONCOGUARD AI - Privacy & Security Documentation

## Privacy Principles

### Data Minimization
- **No Personal Data Collected:** The platform processes only the public benchmark dataset (WDBC) which contains no patient-identifying information
- **No User Tracking:** No analytics, telemetry, or user behavior tracking
- **No Account System:** No authentication, no user profiles, no persistent identities

### Local-First Processing
- **Default Mode:** All ML computation runs locally on user's machine
- **No Cloud Dependency:** Core pipeline works without internet (except initial dataset download)
- **Assistant Offline Mode:** Fully functional without external API calls

### Data Flow Summary

| Component | Data Processed | Storage | External Calls |
|-----------|---------------|---------|----------------|
| Data Pipeline | UCI WDBC (public) | Local parquet/joblib | UCI download (once) |
| Training | Features + targets | Local model artifacts | None |
| Evaluation | Predictions + metrics | Local reports | None |
| Assistant (Online) | User messages only | Session memory only | OpenAI API |
| Assistant (Offline) | User messages only | Session memory only | None |
| Dashboard | User interactions | Browser session only | None |

## Assistant Privacy

### Online Mode (LLM Provider)
- **What's Sent:** User messages + relevant project context (documentation snippets)
- **What's NOT Sent:**
  - API keys or secrets
  - Local file system paths
  - Model artifacts or raw data
  - Conversation history beyond context window (10 messages)
  - Any patient/health information (user responsibility not to share)

- **Provider:** OpenAI (configurable)
- **Data Retention:** Per provider's policy (OpenAI: 30 days for abuse monitoring, not training)
- **User Control:** Can switch to offline mode anytime

### Offline Mode
- **Zero External Calls:** All processing local
- **No Data Leaves Machine:** Complete privacy
- **Limited Capabilities:** Predefined responses only

### Conversation Storage
- **In-Memory Only:** Session state in Streamlit
- **No Persistence:** Cleared on browser refresh/session end
- **Export Option:** User can download JSON (explicit action)
- **No Server Logs:** Streamlit doesn't log chat content by default

## Security Practices

### Secrets Management
```bash
# Correct: Environment variables
OPENAI_API_KEY=sk-...

# Correct: Streamlit secrets (not committed)
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."

# WRONG: Never hardcode
# api_key = "sk-..."  # ❌

# WRONG: Never commit .env
# git add .env  # ❌
```

### Files in .gitignore
```
.env
.env.local
.streamlit/secrets.toml
*.joblib
data/processed/
models/
reports/
__pycache__/
.venv/
```

### Input Validation
- **File Uploads:** Not implemented (no upload vectors)
- **User Messages:** Length limited (4000 chars), sanitized for display
- **Prediction Inputs:** Validated against feature schema, range-checked
- **Tool Calls:** Read-only, predefined functions only

### Model Artifact Security
- **Format:** joblib (Python pickle-based)
- **Trust Model:** Only load artifacts created by THIS pipeline
- **Verification:** `verify_model_reload()` checks prediction consistency
- **No Arbitrary Code:** Model artifacts contain only fitted estimators

### Dependency Security
- **Pinned Versions:** All dependencies in requirements.txt/pyproject.toml
- **Standard Libraries:** numpy, pandas, scikit-learn, PyTorch, Streamlit
- **No Obscure Packages:** Well-maintained, widely-used dependencies
- **Regular Updates:** Update with `pip install --upgrade` and test

## Threat Model

### Assets to Protect
1. **API Keys** — OpenAI credentials
2. **Model IP** — Trained model artifacts (low sensitivity, research only)
3. **User Conversations** — Chat history (session-only)
4. **System Integrity** — Code execution environment

### Threats & Mitigations

| Threat | Likelihood | Impact | Mitigation |
|--------|------------|--------|------------|
| API key exposure | Low | Medium | Env vars only, .gitignore, no logging |
| Malicious model pickle | Low | High | Only load self-generated artifacts, verify reload |
| User shares PHI with assistant | Medium | High | Safety guardrails detect & refuse, disclaimer |
| Data leakage in CV | Low | Medium | Strict train-only fitting, test set isolation |
| Dependency vulnerability | Medium | Medium | Pinned versions, regular audits |
| XSS in dashboard | Low | Low | Streamlit auto-escapes, no raw HTML injection |

## Compliance Notes

### NOT Compliant With (By Design)
- ❌ HIPAA — Not a healthcare application, no PHI processed
- ❌ GDPR — No personal data collected, but GDPR rights respected
- ❌ FDA 21 CFR Part 11 — Research tool, not clinical software
- ❌ SOC 2 — Single-user research platform

### Good Practices Followed
- ✅ Data minimization
- ✅ Purpose limitation (research only)
- ✅ Storage limitation (session-only for chats)
- ✅ Integrity & confidentiality (local processing)
- ✅ Transparency (open source, documented)

## User Responsibilities

### When Using Online Assistant
1. **Do not share** personal health information, patient data, or identifiers
2. **Understand** messages go to LLM provider (OpenAI)
3. **Use offline mode** for sensitive contexts
4. **Review** provider's privacy policy

### When Running Locally
1. **Keep .env secure** — don't share, don't commit
2. **Verify artifacts** — only load models you trained
3. **Update dependencies** — run security audits periodically

### When Sharing Results
1. **Anonymize** — no issue (dataset already anonymous)
2. **Attribute** — cite UCI and original authors
3. **Disclaim** — include research-only notice

## Incident Response

### If API Key Exposed
1. Revoke key immediately at provider dashboard
2. Generate new key
3. Update .env / secrets
4. Check usage logs for abuse

### If Malicious Artifact Loaded
1. Delete models/ directory
2. Re-run training pipeline
3. Verify with `verify_model_reload()`

### If User Reports Privacy Issue
1. Document concern
2. Assess data flow
3. Implement fix (usually: improve guardrails, add validation)
4. Update documentation

## Security Checklist for Deployment

- [ ] `.env` in `.gitignore`
- [ ] `.streamlit/secrets.toml` in `.gitignore`
- [ ] No hardcoded secrets in code
- [ ] API keys only in environment
- [ ] Model artifacts not committed (in .gitignore)
- [ ] Dependencies pinned
- [ ] Safety guardrails tested
- [ ] Offline mode functional
- [ ] Disclaimer displayed on health queries
- [ ] No file upload endpoints
- [ ] No arbitrary code execution
- [ ] Input validation on all user inputs

## Contact

For security concerns related to this research platform, please refer to the project documentation. This is an educational project without formal security support channels.