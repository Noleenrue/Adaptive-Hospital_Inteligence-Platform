CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,

    patient_reference VARCHAR(100),

    raw_probability DOUBLE PRECISION NOT NULL,
    calibrated_probability DOUBLE PRECISION NOT NULL,

    risk_tier VARCHAR(20) NOT NULL,

    model_name VARCHAR(100) NOT NULL DEFAULT 'readmission_xgb',
    model_version VARCHAR(20) NOT NULL DEFAULT '1.0',

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)