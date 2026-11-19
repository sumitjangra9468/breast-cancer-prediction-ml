
import streamlit as st
import pandas as pd
import joblib
import re
from pathlib import Path
from io import BytesIO
from pypdf import PdfReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🎗️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# Load CSS and HTML safely
css_path = BASE_DIR / "style.css"
html_path = BASE_DIR / "index.html"

if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True
    )
# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "cancer_model.pkl")

try:
    saved = load_model()
    model = saved["model"]
    encoder = saved["label_encoder"]
    features = saved["feature_columns"]
    medians = saved.get("feature_medians", {})
    best_name = saved.get("best_model_name", "Trained Model")
    results = saved.get("results", [])

except Exception as e:
    st.error(f"Unable to load cancer_model.pkl: {e}")
    st.stop()

# Check for unwanted columns
if any(str(f).strip().lower().startswith("unnamed") for f in features):
    st.error(
        "Your saved model contains an unwanted Unnamed column. "
        "Please clean the dataset and retrain the model."
    )
    st.stop()

# ---------------- NAVIGATION ----------------
st.sidebar.title("🎗️ Breast Cancer")
st.sidebar.caption("Detection System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Single Prediction",
        "PDF Prediction",
        "Model Performance",
        "About"
    ]
)

# ---------------- PDF REPORT ----------------
def create_report(prediction, inputs):
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("Breast Cancer Prediction Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Predicted class: {prediction}", styles["Heading2"]),
        Paragraph(f"Selected model: {best_name}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Input Parameters", styles["Heading2"])
    ]

    rows = [["Feature", "Value"]]
    rows += [[str(k), str(v)] for k, v in inputs.items()]

    table = Table(rows, repeatRows=1)
    table.setStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
    ])

    content.extend([
        table,
        Spacer(1, 12),
        Paragraph(
            "This result is for educational purposes only and is not intended for medical diagnosis.",
            styles["Normal"]
        )
    ])

    pdf.build(content)
    return buffer.getvalue()

# ---------------- SHARED PREDICTION FUNCTION ----------------
def predict(input_df):
    try:
        # Always use the exact feature order from training
        input_df = input_df[features].copy()

        prediction = model.predict(input_df)
        label = encoder.inverse_transform(prediction)[0]

        st.subheader("Prediction Result")

        if str(label).upper() in ["M", "MALIGNANT"]:
            st.success(f"🟢 Predicted class: {label}")
        elif str(label).upper() in ["B", "BENIGN"]:
            st.error(f"🔴 Predicted class: {label}")
        else:
            st.info(f"Predicted class: {label}")

        # Show confidence only when supported by the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            confidence = max(probabilities) * 100
            st.metric("Model confidence estimate", f"{confidence:.2f}%")

        report = create_report(label, input_df.iloc[0].to_dict())

        st.download_button(
            "Download PDF Report",
            data=report,
            file_name="prediction_report.pdf",
            mime="application/pdf"
        )

     
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------- HOME ----------------
if page == "Home":
    st.title("Breast Cancer Detection Dashboard")

    if html_path.exists():
        st.markdown(
            html_path.read_text(encoding="utf-8"),
            unsafe_allow_html=True
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Input Features", len(features))
    c2.metric("Prediction Classes", len(encoder.classes_))
    c3.metric("Selected Model", best_name)

    st.subheader("Project Overview")

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="metric-card">
            <h3>Machine Learning</h3>
            <p>Classification using the trained model.</p>
            <p>Preprocessing and feature scaling.</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="metric-card">
            <h3>Prediction Tools</h3>
            <p>Manual input and PDF-based prediction.</p>
            <p>Downloadable prediction reports.</p>
        </div>
        """, unsafe_allow_html=True)

    if results:
        st.subheader("Model Accuracy")
        result_df = pd.DataFrame(results)
        st.dataframe(result_df, use_container_width=True)

        # if {"Model", "Accuracy (%)"}.issubset(result_df.columns):
        #     st.bar_chart(result_df.set_index("Model")["Accuracy (%)"])

# ---------------- SINGLE PREDICTION ----------------
elif page == "Single Prediction":
    st.title("Single Prediction")
    st.write("Enter the required feature measurements.")
    values = {}
    with st.form("prediction_form"):
        for i in range(0, len(features), 4):
            columns = st.columns(4)
            for col, feature in zip(columns, features[i:i + 4]):
                with col:
                    try:
                        default_value = float(medians.get(feature, 0.0))
                    except (TypeError, ValueError):
                        default_value = 0.0

                    values[feature] = st.number_input(
                        feature,
                        value=default_value,
                        format="%.5f"
                    )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([values], columns=features)
        predict(input_df)

# ---------------- PDF PREDICTION ----------------

elif page == "PDF Prediction":
    st.title("PDF Prediction")
    st.write(
        "Upload a PDF containing data for multiple people. "
        "Each person's data should have all required feature names and values."
    )

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="prediction_pdf"
    )

    if uploaded_pdf is not None:
        try:
            reader = PdfReader(uploaded_pdf)

            pdf_text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )

            # Split the PDF into individual patient records.
            # Use headings such as Patient 1, Patient 2, etc.
            sections = re.split(
                r"(?im)(?=^\s*(?:(?:sample)\s+)?"
                r"(?:patient|person|record)\s*"
                r"(?:id|no\.?|number)?\s*[:#-]?\s*\d+\b)",
                pdf_text
            )

            # Remove empty sections
            sections = [s.strip() for s in sections if s.strip()]

            # If there are no patient headings, try treating the
            # entire PDF as one record.
            if len(sections) == 1:
                sections = [pdf_text]

            all_records = []
            errors = []

            # Extract each patient's feature values
            for index, section in enumerate(sections, start=1):
                values = {}

                for feature in features:
                    pattern = (
                        r"(?<![A-Za-z0-9_])"
                        + re.escape(str(feature))
                        + r"(?![A-Za-z0-9_])"
                        + r"\s*(?::|=)?\s*"
                        + r"([-+]?(?:\d+\.?\d*|\.\d+)"
                        r"(?:[eE][-+]?\d+)?)"
                    )

                    match = re.search(
                        pattern, section, re.IGNORECASE
                    )

                    if match:
                        values[feature] = float(match.group(1))

                missing = [f for f in features if f not in values]

                if missing:
                    errors.append({
                        "Record": index,
                        "Missing features": ", ".join(missing)
                    })
                else:
                    all_records.append(values)

            # Show extraction errors if any records are incomplete
            if errors:
                st.error(
                    "Some records could not be extracted completely. "
                    "Please check the PDF and ensure each person has "
                    "all required feature names and values."
                )
                st.dataframe(
                    pd.DataFrame(errors),
                    use_container_width=True
                )

            if all_records:
                input_df = pd.DataFrame(
                    all_records,
                    columns=features
                )

                input_df.insert(
                    0,
                    "Person",
                    [f"Person {i}" for i in range(1, len(input_df) + 1)]
                )

                st.subheader("Extracted Patient Data")
                st.write(f"Complete records found: {len(all_records)}")
                st.dataframe(input_df, use_container_width=True)

                if st.button("Predict All People"):
                    # Keep only the feature columns for the model
                    model_input = input_df[features].copy()

                    # Predict every person using the trained model
                    predictions = model.predict(model_input)
                    labels = encoder.inverse_transform(predictions)

                    results_df = pd.DataFrame({
                        "Person": input_df["Person"],
                        "Predicted Diagnosis": labels
                    })

                    # Add confidence when supported by the model
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(model_input)
                        results_df["Confidence (%)"] = (
                            probabilities.max(axis=1) * 100
                        ).round(2)

                    st.subheader("Prediction Results")
                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )

                    # Download results for all people
                    csv_data = results_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download All Predictions (CSV)",
                        data=csv_data,
                        file_name="all_patient_predictions.csv",
                        mime="text/csv"
                    )

                    st.warning(
                        "This is an educational prediction, not a "
                        "medical diagnosis."
                    )

        except Exception as e:
            st.error(f"Unable to read this PDF: {e}")


# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":
    st.title("Model Performance")

    if results:
        result_df = pd.DataFrame(results)
        st.dataframe(result_df, use_container_width=True)

        if {"Model", "Accuracy (%)"}.issubset(result_df.columns):
            st.bar_chart(result_df.set_index("Model")["Accuracy (%)"])
    else:
        st.info("No saved model comparison results found.")

    st.write("Selected model:", best_name)

# ---------------- ABOUT ----------------
elif page == "About":
    st.title("About the Project")
    st.markdown("""
    <div class="hero-card">
        <h2>Breast Cancer Prediction Using Machine Learning</h2>
        <p>
            An academic project using supervised machine learning
            and Streamlit to classify breast cancer feature measurements.
        </p>
    </div>

    <p style="
        color: #000000;
        font-size: 16px;
        font-weight: 600;
        margin-top: 20px;
    ">
        This application is for educational purposes and does not
        replace assessment by a qualified healthcare professional.
    </p>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
    
st.markdown(
    '<div class="footer">Breast Cancer Detection | College Project</div>',
    unsafe_allow_html=True
)
