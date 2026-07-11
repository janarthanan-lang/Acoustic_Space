# Legacy CNN stand-in (superseded)

This folder holds the original lightweight CNN classifier used before real
HuggingFace AST access was available. It's kept only for reference / diff
comparison. The live API (`main.py`) now uses `model_ast.py` +
`train_ast_model.py` in the parent `backend/` folder instead.

Not imported by anything in the current pipeline. Safe to delete if you
don't need the history.
