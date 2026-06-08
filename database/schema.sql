-- Execute this SQL script in MySQL Workbench to create the database table manually.

CREATE DATABASE IF NOT EXISTS housing_db;
USE housing_db;

CREATE TABLE IF NOT EXISTS house_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    area INT,
    bedrooms INT,
    bathrooms INT,
    stories INT,
    mainroad VARCHAR(5),
    guestroom VARCHAR(5),
    basement VARCHAR(5),
    hotwaterheating VARCHAR(5),
    airconditioning VARCHAR(5),
    parking INT,
    prefarea VARCHAR(5),
    furnishingstatus VARCHAR(20),
    predicted_price FLOAT,
    source VARCHAR(20),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
