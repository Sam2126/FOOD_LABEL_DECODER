CREATE TABLE scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT,
    ingredients_raw TEXT,
    flagged_ingredients TEXT,
    scan_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE drift_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT,
    previous_ingredients TEXT,
    new_ingredients TEXT,
    diff TEXT,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
