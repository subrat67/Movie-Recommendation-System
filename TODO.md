# TODO for Recommendation Improvements

## Current Status
- CSVs ready in archive/ and src/data/
- Basic app working (heuristic recs)
- Virtual environment and dependencies configured
- Streamlit app runs at http://localhost:8501

## Improvement Steps (Content-Based TF-IDF + Cosine Sim)
1. [x] Update requirements.txt (scipy)
2. [x] Edit src/preprocess.py (soup added)
3. [ ] Run preprocess (or compute soup in app)
4. [ ] Edit src/app.py (TF-IDF/cosine recs)
5. [ ] Test with `streamlit run src/app.py`
6. [ ] Optional: SVD hybrid
7. [ ] Optional: Deploy to cloud server
