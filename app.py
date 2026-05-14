import streamlit as st
from workflow import workflow, AgentState, PatientProfile, RiskAssessment, PrescriptionPlan, LifestylePlan, MedicalSearchQuery, MedicalEvidence, ClinicalAlert

st.set_page_config(page_title="Healthcare AI Clinical Support", layout="wide")

st.title("🩺 Healthcare AI Clinical Decision Support System")
st.markdown("Interactive patient monitoring, risk assessment, and lifestyle guidance using AI.")

# --- Patient Input Form ---
with st.form("patient_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=55)
    gender = st.selectbox("Gender", ["male", "female"])
    height = st.number_input("Height (meters)", min_value=0.5, max_value=2.5, value=1.75)
    weight = st.number_input("Weight (kg)", min_value=1, max_value=300, value=90)
    sbp = st.number_input("Systolic BP", min_value=50, max_value=250, value=145)
    dbp = st.number_input("Diastolic BP", min_value=30, max_value=150, value=92)
    chol = st.number_input("Total Cholesterol", min_value=0, max_value=500, value=250)
    ldl = st.number_input("LDL Cholesterol", min_value=0, max_value=500, value=170)
    hdl = st.number_input("HDL Cholesterol", min_value=0, max_value=500, value=35)
    tg = st.number_input("Triglycerides", min_value=0, max_value=500, value=220)
    hr = st.number_input("Heartbeat Rate", min_value=30, max_value=200, value=72)
    sugar = st.number_input("Fasting Sugar Level (mmol/L)", min_value=0.0, max_value=20.0, value=8.0)
    sleep_hours = st.number_input("Average Sleep Hours", min_value=0.0, max_value=24.0, value=5.5)
    steps = st.number_input("Average Daily Steps", min_value=0, max_value=20000, value=3000)
    
    st.markdown("### 🧪 Additional Clinical Markers")
    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
        resp = st.number_input("Respiratory Rate", min_value=0, max_value=60, value=16)
    with col2:
        wbc = st.number_input("WBC Count (x10^9/L)", min_value=0.0, max_value=100.0, value=7.0, step=0.1)
        plt = st.number_input("Platelets (x10^9/L)", min_value=0.0, max_value=1000.0, value=250.0, step=1.0)
    with col3:
        spo2 = st.number_input("Oxygen Saturation (%)", min_value=0.0, max_value=100.0, value=98.0, step=0.1)
    
    symptoms = st.text_input("Symptoms (comma-separated)", "chest_pain,fatigue")
    medical_history = st.text_input("Medical History (comma-separated)", "diabetes")
    allergies = st.text_input("Allergies (comma-separated)", "Penicillin")
    immunizations = st.text_input("Immunizations (comma-separated)", "Influenza")

    submitted = st.form_submit_button("🚀 Run Clinical Workflow")

if submitted:
    with st.spinner(text="Running clinical workflow...", show_time=True):
        # Prepare state
        raw_patient_data = {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "systolic_bp": sbp,
            "diastolic_bp": dbp,
            "cholesterol": chol,
            "ldl_cholesterol": ldl,
            "hdl_cholesterol": hdl,
            "triglycerides": tg,
            "heartbeat_rate": hr,
            "sugar_level": sugar,
            "avg_sleep_hours": sleep_hours,
            "avg_daily_steps": steps,
            "temperature": temp,
            "respiratory_rate": resp,
            "wbc_count": wbc,
            "platelets": plt,
            "oxygen_saturation": spo2,
            "symptoms": [s.strip() for s in symptoms.split(",")],
            "medical_history": [s.strip() for s in medical_history.split(",")],
            "allergies": [s.strip() for s in allergies.split(",")],
            "immunizations": [s.strip() for s in immunizations.split(",")]
        }

        state = AgentState(
            patient_profile=PatientProfile(
                age=age,
                gender=gender,
                height=height,
                weight=weight,
                systolic_bp=sbp,
                diastolic_bp=dbp,
                cholesterol=chol,
                ldl_cholesterol=ldl,
                hdl_cholesterol=hdl,
                triglycerides=tg,
                heartbeat_rate=hr,
                sugar_level=sugar,
                avg_sleep_hours=sleep_hours,
                avg_daily_steps=steps,
                temperature=temp,
                respiratory_rate=resp,
                wbc_count=wbc,
                platelets=plt,
                oxygen_saturation=spo2,
                symptoms=raw_patient_data["symptoms"],
                medical_history=raw_patient_data["medical_history"],
                allergies=raw_patient_data["allergies"],
                immunizations=raw_patient_data["immunizations"]
            ),
            risk_assessment=RiskAssessment(disease_risks=[], risk_flags=[], risk_summary=""),
            prescription_plan=PrescriptionPlan(medications=[], recommendations=[], instructions=[]),
            lifestyle_plan=LifestylePlan(exercises=[], diet=[], sleep=[], metabolic_advice=[]),
            raw_patient_data=raw_patient_data,
            medical_search_query=MedicalSearchQuery(query=""),
            medical_evidence=MedicalEvidence(query="", retrieved_chunks_count=0, refined_context="", clinical_summary="", sources_used=[]),
            clinical_alert=ClinicalAlert(
                urgency="LOW",
                message="No alert",
                conditions_flagged=[],
                interaction_flags=[],
                recommended_action="Continue monitoring"
            ),
            treatment_road_map=""
        )

        # Execute workflow
        result = workflow.invoke(state)

        st.success("Clinical workflow completed successfully!")

        # Display Clinical Road Map (Synthesis)
        st.subheader("🗺️ Clinical Road Map")
        st.info(result.get("treatment_road_map", "Synthesis pending..."))

        # Display Risk Assessment
        st.subheader("📊 Risk Assessment")
        ra = result.get("risk_assessment")
        st.markdown(f"**Risk Summary:** {ra.risk_summary}")
        st.markdown("**Disease Risks:**")
        for risk in ra.disease_risks:
            st.markdown(f"- {risk.disease_name}: {risk.risk_score*100:.1f}%")
        st.markdown("**Risk Flags:**")
        for flag in ra.risk_flags:
            st.markdown(f"- {flag}")

        # Display Prescription Plan
        st.subheader("💊 Prescription Plan")
        pp = result.get("prescription_plan")
        st.markdown("**Medications:**")
        for med in pp.medications:
            st.markdown(f"- **Name:** {med.name}, Dose: {med.dose}, Notes: {med.notes}")
        st.markdown("**Recommendations:**")
        for rec in pp.recommendations:
            st.markdown(f"- {rec}")
        st.markdown("**Instructions:**")
        for instr in pp.instructions:
            st.markdown(f"- {instr}")

        # Display Lifestyle Plan
        st.subheader("🥗 Lifestyle Recommendations")
        lp = result.get("lifestyle_plan")
        st.markdown("**Exercises:**")
        for ex in lp.exercises:
            st.markdown(f"- {ex}")
        st.markdown("**Diet:**")
        for d in lp.diet:
            st.markdown(f"- {d}")
        st.markdown("**Sleep:**")
        for s in lp.sleep:
            st.markdown(f"- {s}")
        st.markdown("**Metabolic Advice:**")
        for m in lp.metabolic_advice:
            st.markdown(f"- {m}")

        # Display Clinical Escalation / Alert
        st.subheader("⚠️ Clinical Alert / Escalation")
        ca = result.get("clinical_alert")
        st.markdown(f"**Urgency:** {ca.urgency}")
        st.markdown(f"**Message:** {ca.message}")
        st.markdown(f"**Conditions Flagged:** {', '.join(ca.conditions_flagged)}")
        st.markdown(f"**Interaction Risks:** {', '.join(ca.interaction_flags)}")
        st.markdown(f"**Recommended Action:** {ca.recommended_action}")