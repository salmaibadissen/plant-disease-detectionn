# app.py
import os
import json
import numpy as np
import base64
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging
from pathlib import Path
import shutil

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
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

# ==============================
# Nettoyer les uploads au démarrage
# ==============================
def cleanup_uploads():
    """Supprimer tous les fichiers uploadés au démarrage"""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    logger.info(f"Supprimé: {filepath}")
        logger.info("✓ Nettoyage des uploads terminé")
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage: {e}")

# ==============================
# Configuration Logging
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nettoyer au démarrage
cleanup_uploads()

# ==============================
# Initialiser Flask
# ==============================
app = Flask(__name__, static_folder=UPLOAD_FOLDER, static_url_path='/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['JSON_SORT_KEYS'] = False
app.secret_key = 'your-secret-key-here-change-this'

# ==============================
# Charger le modèle CNN
# ==============================
try:
    model = load_model("best_model.h5")
    logger.info("✓ Modèle CNN chargé avec succès")
except Exception as e:
    logger.error(f"✗ Erreur lors du chargement du modèle: {e}")
    model = None

# ==============================
# Charger les labels
# ==============================
try:
    with open("class_indices.json", "r") as f:
        class_labels = json.load(f)
    class_labels = {int(k): v for k, v in class_labels.items()}
    logger.info(f"✓ {len(class_labels)} classes chargées")
except Exception as e:
    logger.error(f"✗ Erreur lors du chargement des labels: {e}")
    class_labels = {}

# ==============================
# Charger les solutions
# ==============================
try:
    from solutions import solutions
    logger.info(f"✓ {len(solutions)} solutions de maladies chargées")
except Exception as e:
    logger.error(f"✗ Erreur lors du chargement des solutions: {e}")
    solutions = {}

# ==============================
# Fonctions utilitaires
# ==============================
def allowed_file(filename):
    """Vérifier si le fichier a une extension autorisée"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_disease_severity(label):
    """Déterminer la sévérité de la maladie"""
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
    """Obtenir les étapes de traitement basées sur la maladie et la sévérité"""
    base_steps = []
    
    if severity == 'high':
        base_steps = [
            "Isoler immédiatement la plante",
            "Consulter un phytopathologiste",
            "Utiliser des traitements fongicides/bactéricides appropriés",
            "Inspecter quotidiennement"
        ]
    elif severity == 'medium':
        base_steps = [
            "Augmenter la fréquence d'inspection",
            "Appliquer des traitements de prévention",
            "Améliorer la ventilation autour de la plante",
            "Ajuster l'arrosage"
        ]
    else:
        base_steps = [
            "Maintenir une hygiène régulière",
            "Continuer les soins normaux",
            "Inspecter hebdomadairement",
            "Optimiser les conditions de croissance"
        ]
    
    return base_steps

# ==============================
# Fonction de prédiction image
# ==============================
def predict_image(img_path):
    """Prédire la maladie à partir d'une image"""
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Fichier non trouvé: {img_path}")
        
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        label = class_labels.get(predicted_class, "Classe inconnue")
        
        # Obtenir les top 3 prédictions
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_predictions = []
        for idx in top_3_indices:
            if idx < len(class_labels):
                top_predictions.append({
                    'class': class_labels.get(idx, "Unknown"),
                    'confidence': float(prediction[0][idx]) * 100
                })
        
        return label, confidence, prediction[0], top_predictions
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise

def generate_pdf_report(prediction, confidence, disease_info, severity, img_path):
    """Générer un rapport PDF"""
    if not HAS_REPORTLAB:
        return None
    
    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Titre
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1b5e20'),
            spaceAfter=30,
            alignment=1
        )
        elements.append(Paragraph("🌿 Rapport d'Analyse Phytopathologique", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Infos générales
        info_data = [
            ['Diagnostic', prediction],
            ['Sévérité', severity.upper()],
            ['Confiance', f'{confidence*100:.1f}%'],
            ['Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#1b5e20')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Description
        elements.append(Paragraph("Description", styles['Heading2']))
        elements.append(Paragraph(disease_info.get('description', 'N/A'), styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Causes
        elements.append(Paragraph("Causes", styles['Heading2']))
        elements.append(Paragraph(disease_info.get('causes', 'N/A'), styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Symptômes
        elements.append(Paragraph("Symptômes", styles['Heading2']))
        elements.append(Paragraph(disease_info.get('symptoms', 'N/A'), styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Traitement
        elements.append(Paragraph("Gestion et Traitement", styles['Heading2']))
        elements.append(Paragraph(disease_info.get('management', 'N/A'), styles['Normal']))
        
        # Générer le PDF
        doc.build(elements)
        pdf_buffer.seek(0)
        return pdf_buffer
    except Exception as e:
        logger.error(f"Erreur lors de la génération du PDF: {e}")
        return None

# ==============================
# Routes
# ==============================

@app.route("/", methods=["GET", "POST"])
def index():
    """Route principale"""
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("index.html", error="Aucun fichier sélectionné")
        
        file = request.files["file"]
        
        if file.filename == "":
            return render_template("index.html", error="Aucun fichier sélectionné")
        
        if not allowed_file(file.filename):
            return render_template(
                "index.html", 
                error="Format non autorisé. PNG, JPG, JPEG, GIF ou WebP"
            )
        
        try:
            filename = secure_filename(file.filename)
            # Ajouter un timestamp pour éviter les conflits
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"Fichier uploadé: {filepath}")
            
            # Prédiction
            label, confidence, all_predictions, top_predictions = predict_image(filepath)
            confidence_percent = confidence * 100
            
            # Sévérité
            severity = get_disease_severity(label)
            
            # Informations sur la maladie
            if "healthy" in label.lower():
                disease_info = {
                    "description": "La plante semble en bonne santé.",
                    "causes": "Aucune maladie détectée.",
                    "symptoms": "Aucun symptôme anormal.",
                    "management": "Maintenir les soins réguliers."
                }
            else:
                disease_info = solutions.get(label)
                if disease_info is None:
                    disease_info = {
                        "description": "Pas de description disponible.",
                        "causes": "Information non disponible.",
                        "symptoms": "Information non disponible.",
                        "management": "Consulter un expert."
                    }
            
            logger.info(f"Prédiction: {label} ({confidence_percent:.2f}%)")
            
            # URL correcte pour l'image
            image_url = f"/uploads/{filename}"
            
            # Étapes de traitement
            treatment_steps = get_treatment_steps(label, severity)
            
            return render_template(
                "index.html",
                prediction=label,
                confidence=round(confidence_percent, 2),
                disease_info=disease_info,
                image_path=image_url,
                severity=severity,
                top_predictions=top_predictions,
                treatment_steps=treatment_steps
            )
            
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return render_template(
                "index.html", 
                error=f"Erreur: {str(e)}"
            )
    
    # Afficher l'interface vierge au chargement initial
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Servir les fichiers uploadés"""
    return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)

@app.route("/api/health", methods=["GET"])
def health_check():
    """Vérifier l'état de l'application"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "classes_loaded": len(class_labels) > 0,
        "solutions_loaded": len(solutions) > 0,
        "pdf_export": HAS_REPORTLAB
    })

@app.route("/api/diseases", methods=["GET"])
def get_diseases():
    """Obtenir la liste des maladies supportées"""
    return jsonify({
        "count": len(solutions),
        "diseases": list(solutions.keys())
    })

@app.route("/api/predict", methods=["POST"])
def predict_api():
    """API de prédiction (JSON)"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File format not allowed"}), 400
    
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        label, confidence, all_predictions, top_predictions = predict_image(filepath)
        severity = get_disease_severity(label)
        disease_info = solutions.get(label, {})
        
        return jsonify({
            "prediction": label,
            "confidence": float(confidence),
            "severity": severity,
            "disease_info": disease_info,
            "top_predictions": top_predictions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    """Exporter le rapport en PDF"""
    data = request.json
    
    disease_info = {
        "description": data.get("description", ""),
        "causes": data.get("causes", ""),
        "symptoms": data.get("symptoms", ""),
        "management": data.get("management", "")
    }
    
    pdf_buffer = generate_pdf_report(
        data.get("prediction", "Unknown"),
        float(data.get("confidence", 0)) / 100,
        disease_info,
        data.get("severity", "unknown"),
        None
    )
    
    if pdf_buffer:
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
    return jsonify({"error": "PDF generation not available"}), 503

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template(
        "index.html", 
        error="Le fichier dépasse 10MB"
    ), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erreur serveur: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    if model is None:
        logger.error("⚠ Le modèle n'a pas pu être chargé.")
    if not HAS_REPORTLAB:
        logger.warning("⚠ ReportLab non installé - Export PDF désactivé")
    
    app.run(debug=True, host="127.0.0.1", port=5000)