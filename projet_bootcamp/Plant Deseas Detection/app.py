# app.py
import os
import json
import numpy as np
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import logging
from pathlib import Path

# TensorFlow CPU only
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Télécharger depuis Google Drive
import gdown

# ReportLab pour PDF (optionnel)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# ==============================
# Configuration
# ==============================
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Nettoyer les uploads au démarrage
# ==============================
def cleanup_uploads():
    if os.path.exists(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            fp = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(fp):
                os.remove(fp)
        logger.info("✓ Uploads nettoyés")

cleanup_uploads()

# ==============================
# Initialiser Flask
# ==============================
app = Flask(__name__, static_folder=UPLOAD_FOLDER, static_url_path='/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['JSON_SORT_KEYS'] = False
app.secret_key = 'change-this-secret-key'

# ==============================
# Charger le modèle CNN depuis Google Drive
# ==============================
GDRIVE_URL = "https://drive.google.com/uc?id=1Q-QNfWETkSJaGTFjjscIv04FYFuru4tf"
MODEL_PATH = "best_model.h5"

if not os.path.exists(MODEL_PATH):
    logger.info("Téléchargement du modèle depuis Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

try:
    model = load_model(MODEL_PATH)
    logger.info("✓ Modèle CNN chargé avec succès")
except Exception as e:
    logger.error(f"✗ Erreur lors du chargement du modèle: {e}")
    model = None

# ==============================
# Charger les labels
# ==============================
try:
    with open("class_indices.json", "r") as f:
        class_labels = {int(k): v for k, v in json.load(f).items()}
    logger.info(f"✓ {len(class_labels)} classes chargées")
except Exception as e:
    logger.error(f"✗ Erreur chargement labels: {e}")
    class_labels = {}

# ==============================
# Charger solutions maladies
# ==============================
try:
    from solutions import solutions
    logger.info(f"✓ {len(solutions)} solutions chargées")
except Exception as e:
    logger.error(f"✗ Erreur solutions: {e}")
    solutions = {}

# ==============================
# Fonctions utilitaires
# ==============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_disease_severity(label):
    severity_keywords = {
        'high': ['blight', 'rot', 'wilt', 'canker'],
        'medium': ['mildew', 'rust', 'spot', 'mosaic'],
        'low': ['healthy', 'normal']
    }
    label_lower = label.lower()
    for severity, keywords in severity_keywords.items():
        if any(kw in label_lower for kw in keywords):
            return severity
    return 'medium'

def get_treatment_steps(label, severity):
    base_steps = []
    if severity == 'high':
        base_steps = ["Isoler immédiatement la plante", "Consulter un phytopathologiste", "Utiliser traitements appropriés", "Inspecter quotidiennement"]
    elif severity == 'medium':
        base_steps = ["Augmenter fréquence inspection", "Appliquer traitements préventifs", "Améliorer ventilation", "Ajuster arrosage"]
    else:
        base_steps = ["Maintenir hygiène", "Continuer soins normaux", "Inspecter hebdomadairement", "Optimiser conditions croissance"]
    return base_steps

def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        label = class_labels.get(predicted_class, "Classe inconnue")
        
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_predictions = [{"class": class_labels.get(i, "Unknown"), "confidence": float(prediction[0][i])*100} for i in top_3_indices]
        return label, confidence, prediction[0], top_predictions
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise

def generate_pdf_report(prediction, confidence, disease_info, severity, img_path):
    if not HAS_REPORTLAB:
        return None
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        elements.append(Paragraph("🌿 Rapport d'Analyse Phytopathologique", ParagraphStyle('title', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#1b5e20'), spaceAfter=30, alignment=1)))
        
        info_data = [
            ['Diagnostic', prediction],
            ['Sévérité', severity.upper()],
            ['Confiance', f'{confidence*100:.1f}%'],
            ['Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        table = Table(info_data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([('BACKGROUND', (0,0),(0,-1),colors.HexColor('#1b5e20')), ('TEXTCOLOR',(0,0),(0,-1),colors.whitesmoke), ('ALIGN',(0,0),(-1,-1),'LEFT'), ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,-1),11), ('GRID',(0,0),(-1,-1),1,colors.black)]))
        elements.append(table)
        
        for section in ['description', 'causes', 'symptoms', 'management']:
            elements.append(Spacer(1,0.2*inch))
            elements.append(Paragraph(section.capitalize(), styles['Heading2']))
            elements.append(Paragraph(disease_info.get(section,'N/A'), styles['Normal']))
        
        doc.build(elements)
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(f"Erreur PDF: {e}")
        return None

# ==============================
# Routes Flask
# ==============================
@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("index.html", error="Aucun fichier sélectionné")
        file = request.files['file']
        if file.filename == "" or not allowed_file(file.filename):
            return render_template("index.html", error="Format non autorisé")
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        label, confidence, _, top_predictions = predict_image(filepath)
        severity = get_disease_severity(label)
        disease_info = solutions.get(label, {"description":"N/A","causes":"N/A","symptoms":"N/A","management":"N/A"})
        treatment_steps = get_treatment_steps(label, severity)
        image_url = f"/uploads/{filename}"
        
        return render_template("index.html", prediction=label, confidence=round(confidence*100,2), disease_info=disease_info, image_path=image_url, severity=severity, top_predictions=top_predictions, treatment_steps=treatment_steps)
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)

@app.route("/api/predict", methods=["POST"])
def predict_api():
    if 'file' not in request.files:
        return jsonify({"error":"No file provided"}), 400
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({"error":"File format not allowed"}), 400
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
    filename = timestamp + filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    label, confidence, _, top_predictions = predict_image(filepath)
    severity = get_disease_severity(label)
    disease_info = solutions.get(label, {"description":"N/A","causes":"N/A","symptoms":"N/A","management":"N/A"})
    
    return jsonify({"prediction":label, "confidence":float(confidence), "severity":severity, "disease_info":disease_info, "top_predictions":top_predictions})

@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    data = request.json
    disease_info = {
        "description": data.get("description",""),
        "causes": data.get("causes",""),
        "symptoms": data.get("symptoms",""),
        "management": data.get("management","")
    }
    pdf_buffer = generate_pdf_report(data.get("prediction","Unknown"), float(data.get("confidence",0))/100, disease_info, data.get("severity","unknown"), None)
    if pdf_buffer:
        return send_file(pdf_buffer, mimetype='application/pdf', as_attachment=True, download_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    return jsonify({"error":"PDF generation not available"}),503

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status":"ok","model_loaded":model is not None,"classes_loaded":len(class_labels)>0,"solutions_loaded":len(solutions)>0,"pdf_export":HAS_REPORTLAB})

# ==============================
# Gestion erreurs
# ==============================
@app.errorhandler(413)
def request_entity_too_large(error):
    return "Fichier >10MB", 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error":"Not found"}),404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error":"Internal server error"}),500

# ==============================
# Lancer Flask
# ==============================
if __name__ == "__main__":
    if model is None:
        logger.error("⚠ Le modèle n'a pas pu être chargé.")
    if not HAS_REPORTLAB:
        logger.warning("⚠ ReportLab non installé - Export PDF désactivé")
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port, debug=True)