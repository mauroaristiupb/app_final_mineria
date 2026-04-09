# ==============================
# Librerías
# ==============================

import pandas as pd
import pickle
import streamlit as st

# ==============================
# Cargar modelo
# ==============================

filename = "modelo-reg.pkl"
modelo, min_max_scaler, variables = pickle.load(open(filename, "rb"))


# ==============================
# Interfaz
# ==============================

st.title("Predicción número de intentos suicidas")

# ------------------------------
# Variables principales
# ------------------------------

edad_paciente = st.slider(
    "Edad del paciente",
    min_value=5,
    max_value=100,
    value=20
)

sexo = st.selectbox(
    "Sexo",
    ["H", "M"]
)

tipo_caso = "Intento"

reg_seg_social = st.selectbox(
    "Régimen seguridad social (Contributivo, Subsidiado, Especial, Excepcion, No asegurado, Independiente)",
    ["C", "S", "E", "P", "N", "I"]
)

estado_civil = st.slider(
    "Estado civil (1. Soltero, 2. Casado, 3. Unión libre, 4. Viudo, 5. Separado - Divorciado)",
    min_value=1,
    max_value=5,
    value=1
)

escolaridad = st.slider(
    "Escolaridad (1. Preescolar, 2. Básica primaria, 3. Básica secundaria, 5. Media técnica, 7. Técnica profesional, 8. Tecnológica o técnica, 9. Profesional, 10. Especialización, 11. Maestría, 12. Doctorado, 13. Ninguno, 14. Sin información)",
    min_value=1,
    max_value=14,
    value=1
)

# ------------------------------
# Variables binarias automáticas
# ------------------------------

binary_variables = [
    "conflict_pareja_ex",
    "enf_cron_dol_disc",
    "probl_econ",
    "muerte_fam",
    "probl_educ",
    "probl_jurid",
    "suic_fam_amig",
    "maltrato_fis_psic_sex",
    "probl_lab",
    "probl_fam",
    "fact_des_no_ident",
    "consum_spa",
    "ant_fam_cond_suic",
    "ideac_suic_persist",
    "plan_org_suic",
    "ant_trast_psiq",
    "trast_depr",
    "trast_person",
    "trast_bip",
    "esquizof",
    "ant_viol_abus",
    "abuso_alcoh",
    "fact_riesg_no_ident",
    "ahorc_asfx",
    "elemt_cortop",
    "arm_fuego",
    "inmolac",
    "lanz_vacio",
    "intoxicacion",
    "lanz_vehic",
    "lanz_cuerp_agua",
    "mec_no_ident"
]

st.subheader("Factores de riesgo")

binary_inputs = {}

for var in binary_variables:
    binary_inputs[var] = 1 if st.selectbox(
        var.replace("_", " ").capitalize(),
        ["No", "Si"]
    ) == "Si" else 0

# ------------------------------
# Otras variables categóricas
# ------------------------------

tipo_sust = st.slider(
    "Tipo de sustancia (1.Medicamentos, 2.Plaguicidas, 3.Metanol, 4.Metales, 5. Solventes, 6.Otras sustancias químicas, 7.Gases, 8.Sustancias psicoactivas)",
    min_value=1,
    max_value=8,
    value=1
)

# ==============================
# Crear dataframe
# ==============================

data_dict = {
    "tipo_caso": tipo_caso,
    "sexo": sexo,
    "edad_paciente": edad_paciente,
    "reg_seg_social": reg_seg_social,
    "estado_civil": estado_civil,
    "escolaridad": escolaridad,
    "tipo_sust": tipo_sust
}

# añadir variables binarias automáticamente
data_dict.update(binary_inputs)

data = pd.DataFrame([data_dict])

# ==============================
# Preparación de datos
# ==============================

data_preparada = data.copy()

data_preparada = pd.get_dummies(
    data_preparada,
    drop_first=False,
    dtype=int
)

data_preparada = data_preparada.reindex(
    columns=variables,
    fill_value=0
)

# normalizar edad
data_preparada[["edad_paciente"]] = min_max_scaler.transform(
    data_preparada[["edad_paciente"]]
)

# ==============================
# Predicción
# ==============================

if st.button("Predecir"):

    pred = modelo.predict(data_preparada)[0]
    st.subheader("Resultado")
    st.success(f"Número estimado de intentos: {round(pred,2)}")