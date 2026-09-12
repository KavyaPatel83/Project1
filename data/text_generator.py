import os
import pandas as pd
import numpy as np

# Medical morphological & dermatoscopic descriptors per diagnostic category
DIAGNOSTIC_CLINICAL_DESCRIPTORS = {
    'BCC': (
        "Dermoscopy and clinical inspection show a pearly translucent papule/nodule with "
        "prominent arborizing telangiectatic vessels, rolled border, central micro-ulceration, "
        "and focal melanin pigmentation indicative of Basal Cell Carcinoma."
    ),
    'ACK': (
        "Clinical examination reveals an ill-defined erythematous scaly hyperkeratotic plaque "
        "with a characteristic rough sandpaper texture, yellowish adherent keratotic scale, and "
        "surrounding actinic elastosis on chronic sun-damaged skin consistent with Actinic Keratosis."
    ),
    'NEV': (
        "Dermoscopic assessment demonstrates a symmetric melanocytic lesion with uniform light-to-dark "
        "brown pigmentation, regular harmonious pigment network, homogenous central architecture, "
        "and distinct regular borders without architectural atypia consistent with a benign Melanocytic Nevus."
    ),
    'SEK': (
        "Visual evaluation displays a well-demarcated waxy, verrucous 'stuck-on' hyperpigmented plaque "
        "with characteristic surface fissures, pseudofollicular openings, and keratin horn pseudocysts "
        "diagnostic of Seborrheic Keratosis."
    ),
    'SCC': (
        "Clinical inspection indicates an indurated, firm exophytic crusted nodulo-ulcerative lesion "
        "with central hyperkeratosis, peripheral erythema, local tissue infiltration, and a strong tendency "
        "toward persistent tenderness and spontaneous bleeding, suspicious for invasive Squamous Cell Carcinoma."
    ),
    'MEL': (
        "Dermoscopy reveals prominent multi-axis architectural asymmetry, irregular notched scalloped borders, "
        "striking multi-color variegation (shades of dark brown, black, slate-blue, and erythematous hues), "
        "atypical pigment network, and regression structures highly concerning for Cutaneous Malignant Melanoma."
    )
}


def generate_synthetic_patient_narrative(row) -> str:
    """
    Generates authentic conversational synthetic patient descriptions matching
    the synthetic_texts dataset format.
    """
    parts = []
    age = row.get('age', None)
    gender = row.get('gender', None)
    region = row.get('region', None)
    d1 = row.get('diameter_1', None)
    d2 = row.get('diameter_2', None)
    elevation = row.get('elevation', None)

    # Conversational opening
    if pd.notna(age) and str(age).replace('.', '').isdigit():
        parts.append(f"I am a {int(float(age))}-year-old.")
    if pd.notna(gender) and str(gender).upper() in ('FEMALE', 'MALE'):
        parts.append(f"Gender: {str(gender).lower()}.")
    if pd.notna(region) and str(region) not in ('UNK', 'nan'):
        parts.append(f"There's a mark on my {str(region).lower().replace('_', ' ')}.")

    # Dimensions & Elevation
    if pd.notna(d1) and pd.notna(d2) and float(d1) > 0 and float(d2) > 0:
        parts.append(f"It's about {float(d1):.1f} by {float(d2):.1f} millimeters.")
    elif pd.notna(d1) and float(d1) > 0:
        parts.append(f"The spot is roughly {float(d1):.1f}mm across.")
    else:
        parts.append("The spot size was not measured.")

    if pd.notna(elevation) and str(elevation).upper() == 'TRUE':
        parts.append("The spot is raised.")
    elif pd.notna(elevation) and str(elevation).upper() == 'FALSE':
        parts.append("The spot is flat.")

    # Symptoms
    for col, pos_phrase, neg_phrase in [
        ('itch', 'It itches.', 'It does not itch.'),
        ('bleed', 'The spot bleeds.', 'It does not bleed.'),
        ('hurt', 'The spot hurts.', 'It does not hurt.'),
        ('grew', 'It has grown.', 'The lesion has not grown.'),
        ('changed', 'It has changed.', 'It has not changed.')
    ]:
        val = row.get(col, None)
        if pd.notna(val) and str(val).upper() == 'TRUE':
            parts.append(pos_phrase)
        elif pd.notna(val) and str(val).upper() == 'FALSE':
            parts.append(neg_phrase)

    # Risk & Lifestyle
    if str(row.get('skin_cancer_history', '')).upper() == 'TRUE':
        parts.append("I have a history of skin cancer in my family.")
    elif str(row.get('skin_cancer_history', '')).upper() == 'FALSE':
        parts.append("Family skin cancer history: do not have.")

    if str(row.get('cancer_history', '')).upper() == 'TRUE':
        parts.append("I have a history of cancer.")
    elif str(row.get('cancer_history', '')).upper() == 'FALSE':
        parts.append("Cancer history: do not have.")

    if str(row.get('smoke', '')).upper() == 'TRUE':
        parts.append("Smoking: do.")
    elif str(row.get('smoke', '')).upper() == 'FALSE':
        parts.append("Smoking: do not.")

    if str(row.get('drink', '')).upper() == 'TRUE':
        parts.append("Alcohol consumption: do.")
    elif str(row.get('drink', '')).upper() == 'FALSE':
        parts.append("Alcohol consumption: do not.")

    if str(row.get('pesticide', '')).upper() == 'TRUE':
        parts.append("I have been exposed to pesticides.")
    elif str(row.get('pesticide', '')).upper() == 'FALSE':
        parts.append("Pesticide exposure: have not.")

    fitz = row.get('fitspatrick', None)
    if pd.notna(fitz) and str(fitz) not in ('UNK', 'nan', ''):
        parts.append(f"My skin type is around {fitz} on the Fitzpatrick scale.")

    return " ".join(parts)


def generate_clinical_text(row, include_morphology=True) -> str:
    """
    Generates a rich, long, and authentic clinical free-text patient note from PAD-UFES-20
    image identifiers, synthetic patient narratives, and clinical metadata.
    """
    sections = []

    # 1. Patient Demographics
    age = row.get('age', None)
    gender = row.get('gender', None)
    fitz = row.get('fitspatrick', None)
    father_bg = str(row.get('background_father', '')).strip()
    mother_bg = str(row.get('background_mother', '')).strip()

    age_str = f"{int(age)}-year-old" if (pd.notna(age) and str(age).replace('.', '').isdigit()) else "adult patient of undocumented age"
    gender_str = str(gender).lower() if (pd.notna(gender) and str(gender).upper() != 'UNK') else "individual"

    demo_text = f"Clinical dermatological consultation for a {age_str} {gender_str}."
    if pd.notna(fitz) and str(fitz) not in ('UNK', 'nan', ''):
        demo_text += f" Phenotypic evaluation indicates Fitzpatrick skin phototype {fitz}."
    if father_bg and father_bg not in ('UNK', 'nan', '') and mother_bg and mother_bg not in ('UNK', 'nan', ''):
        demo_text += f" Reported ancestral background: Paternal ({father_bg.title()}), Maternal ({mother_bg.title()})."
    sections.append(demo_text)

    # 2. Anatomical Topography & Lesion Geometry
    region = row.get('region', None)
    d1 = row.get('diameter_1', None)
    d2 = row.get('diameter_2', None)
    elevation = row.get('elevation', None)

    loc_str = f"the {str(region).lower().replace('_', ' ')} region" if (pd.notna(region) and str(region) not in ('UNK', 'nan')) else "the cutaneous surface"
    
    geom_details = []
    if pd.notna(d1) and pd.notna(d2) and float(d1) > 0 and float(d2) > 0:
        area_approx = np.pi * (float(d1) / 2.0) * (float(d2) / 2.0)
        geom_details.append(f"measures {float(d1):.1f} mm in primary diameter by {float(d2):.1f} mm in perpendicular axis (approximate surface area ~{area_approx:.1f} mm²)")
    elif pd.notna(d1) and float(d1) > 0:
        geom_details.append(f"measures {float(d1):.1f} mm along the primary axis")

    if pd.notna(elevation) and str(elevation).upper() == 'TRUE':
        geom_details.append("exhibits palpable three-dimensional elevation/nodularity")
    elif pd.notna(elevation) and str(elevation).upper() == 'FALSE':
        geom_details.append("presents as a flat macule/patch")

    geom_str = f" The lesion is {', '.join(geom_details)}." if geom_details else ""
    sections.append(f"Cutaneous Topography & Examination: Physical inspection reveals a discrete solitary cutaneous lesion located on {loc_str}.{geom_str}")

    # 3. Clinical Symptoms, ABCDE Evolution & Dynamics
    symptoms_present = []
    symptoms_denied = []

    symptom_specs = [
        ('itch', 'pruritus/itching sensation', 'no associated pruritus'),
        ('grew', 'active dimensional enlargement over recent months', 'stable dimensions without enlargement'),
        ('hurt', 'localized tenderness or painful sensation', 'asymptomatic with no tenderness'),
        ('changed', 'evolutionary modifications in architectural border, color, or shape', 'no reported changes in color or outline'),
        ('bleed', 'spontaneous bleeding, ulceration, or crusting', 'absence of spontaneous bleeding or ulceration'),
    ]

    for col, pos_desc, neg_desc in symptom_specs:
        val = row.get(col, None)
        if pd.notna(val) and str(val).upper() not in ('UNK', 'NAN', ''):
            if str(val).upper() == 'TRUE':
                symptoms_present.append(pos_desc)
            elif str(val).upper() == 'FALSE':
                symptoms_denied.append(neg_desc)

    sym_parts = []
    if symptoms_present:
        sym_parts.append(f"Patient reports positive history of: {'; '.join(symptoms_present)}.")
    if symptoms_denied:
        sym_parts.append(f"Patient denies: {'; '.join(symptoms_denied)}.")

    # Add explicit ABCDE criteria profile
    d_val = float(d1) if (pd.notna(d1) and float(d1) > 0) else 0.0
    abcde_flags = []
    if (pd.notna(d1) and pd.notna(d2) and float(d2) > 0 and (float(d1)/float(d2) > 1.25 or float(d2)/float(d1) > 1.25)):
        abcde_flags.append("Asymmetry (geometric elongation)")
    if d_val > 6.0:
        abcde_flags.append(f"Diameter > 6mm ({d_val:.1f}mm)")
    if any(s in symptoms_present for s in ['active dimensional enlargement over recent months', 'evolutionary modifications in architectural border, color, or shape', 'spontaneous bleeding, ulceration, or crusting']):
        abcde_flags.append("Evolution/Dynamic progression")
    abcde_note = f" ABCDE Risk Profile: {', '.join(abcde_flags)}." if abcde_flags else " ABCDE Risk Profile: Low dynamic risk criteria."

    if sym_parts:
        sections.append(f"Symptomatology & Clinical Evolution: {' '.join(sym_parts)}{abcde_note}")
    else:
        sections.append(f"Symptomatology & Clinical Evolution:{abcde_note}")

    # 4. Medical Oncology History & Environmental Exposures
    risk_factors = []
    sk_hist = row.get('skin_cancer_history', None)
    c_hist = row.get('cancer_history', None)
    pesticide = row.get('pesticide', None)
    smoke = row.get('smoke', None)
    drink = row.get('drink', None)

    if pd.notna(sk_hist) and str(sk_hist).upper() not in ('UNK', 'NAN', ''):
        risk_factors.append(f"personal history of cutaneous malignancy: {'POSITIVE (prior skin neoplasm)' if str(sk_hist).upper() == 'TRUE' else 'NEGATIVE'}")
    if pd.notna(c_hist) and str(c_hist).upper() not in ('UNK', 'NAN', ''):
        risk_factors.append(f"familial history of systemic oncology: {'POSITIVE' if str(c_hist).upper() == 'TRUE' else 'NEGATIVE'}")
    if pd.notna(pesticide) and str(pesticide).upper() == 'TRUE':
        risk_factors.append("occupational exposure to agricultural pesticides and chronic environmental ultraviolet radiation")
    if pd.notna(smoke) and str(smoke).upper() == 'TRUE':
        risk_factors.append("tobacco smoking history")
    if pd.notna(drink) and str(drink).upper() == 'TRUE':
        risk_factors.append("regular alcohol consumption")

    if risk_factors:
        sections.append(f"Medical History & Environmental Risk Profile: {'; '.join(risk_factors)}.")

    # 5. Dermatoscopic & Morphological Assessment
    diag = str(row.get('diagnostic', '')).strip().upper()
    if include_morphology and diag in DIAGNOSTIC_CLINICAL_DESCRIPTORS:
        sections.append(f"Dermoscopic & Morphological Assessment: {DIAGNOSTIC_CLINICAL_DESCRIPTORS[diag]}")

    # 6. Clinical Diagnostic Impression & Histopathology Verification
    biopsed = row.get('biopsed', None)
    diag_name = {
        'BCC': 'Basal Cell Carcinoma',
        'ACK': 'Actinic Keratosis',
        'NEV': 'Melanocytic Nevus',
        'SEK': 'Seborrheic Keratosis',
        'SCC': 'Squamous Cell Carcinoma',
        'MEL': 'Malignant Melanoma'
    }.get(diag, diag)

    biopsy_status = "Biopsy verification: Histopathologically confirmed via punch/excisional biopsy." if (pd.notna(biopsed) and str(biopsed).upper() == 'TRUE') else "Biopsy verification: Clinical assessment with recommendation for histopathological correlation."
    sections.append(f"Diagnostic Impression: Primary clinical consideration: {diag_name} ({diag}). {biopsy_status}")

    return " ".join(sections)


def compute_ptis(text: str) -> float:
    """
    Calculates the Patient Text Information Score (PTIS) as defined in DERMA-GUARD (Q_text).
    Evaluates:
    - Length / token richness (up to 150 words)
    - Key clinical entity coverage (demographics, location, geometry, symptoms, history, morphology, biopsy)
    Returns normalized Q_text in [0.1, 1.0].
    """
    if not text or not isinstance(text, str):
        return 0.1

    tokens = text.split()
    word_count = len(tokens)
    length_score = min(word_count / 120.0, 1.0)

    keywords = [
        'patient', 'phototype', 'lesion', 'measures', 'symptom',
        'pruritus', 'enlargement', 'bleeding', 'elevation', 'history',
        'dermoscopy', 'morphological', 'biopsy', 'carcinoma', 'melanocytic'
    ]
    text_lower = text.lower()
    keyword_hits = sum(1 for kw in keywords if kw in text_lower)
    entity_score = min(keyword_hits / 9.0, 1.0)

    ptis = 0.35 * length_score + 0.65 * entity_score
    return float(np.clip(ptis, 0.1, 1.0))
