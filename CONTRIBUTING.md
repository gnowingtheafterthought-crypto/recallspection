# 🤝 Contributing to Recallspection

Hey, thanks for stopping by! Whether you're fixing a typo, optimizing FAISS, or adding a new cryptographic backend, we appreciate you.

Recallspection is a research-backed project from Sciencedelic Metatech, but we welcome outsiders who share our obsession with **zero-hallucination memory**.

## 🚀 Quick Start (Dev Setup)
1. Fork the repo.
2. Create a virtual env: `python -m venv venv`
3. Install dev deps: `pip install -e .[dev]`
4. Run tests: `pytest tests/` (ensure they all pass!)
5. Format code: `ruff format .` (we use Ruff)

## 🐛 Found a Bug? Have an Idea?
- Open an **Issue** with the `bug` or `enhancement` label.
- Please include your Python version, OS, and a minimal code snippet to reproduce.

## ✍️ Pull Request Process
1. Open a draft PR early (even if not ready) so we can discuss.
2. Keep PRs focused — one feature/fix per PR.
3. Update the README if you change behavior.
4. Ensure all tests pass and coverage doesn't drop.

## 📜 Legal Stuff (DCO)
To protect the project and its users, we require a **Developer Certificate of Origin (DCO)**. 
By signing off your commits (`git commit -s -m "..."`), you certify you have the right to submit the code under the AGPLv3 license.

**That's it! Now go break something (and fix it). We're excited to see your PR.**