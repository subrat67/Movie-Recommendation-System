# TODO for Recommendation Improvements

## Current Status
- CSVs ready in archive/ & src/data/
- Basic app working (heuristic recs)
- venv/pip deps fixed (scikit-learn ready)

## Improvement Steps (Content-Based TF-IDF + Cosine Sim)
1. [x] Update requirements.txt (scipy)
2. [x] Edit src/preprocess.py (soup added)
3. [ ] Run preprocess [Skipped: compute soup in app]
4. [ ] Edit src/app.py (TF-IDF/cosine recs)
5. [ ] pip install -r requirements.txt (venv)
6. [ ] Test: streamlit run src/app.py
7. [ ] Optional: SVD hybrid
