import app
import waitress
from doc_ir_model import doc_ir_model
from line_ir_model import line_ir_model

waitress.serve(app.hexaf_fever(), host="0.0.0.0", port=5000)
