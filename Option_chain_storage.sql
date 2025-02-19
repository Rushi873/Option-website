CREATE DATABASE IF NOT EXISTS option_data;
USE option_data;

-- Table to store asset details
CREATE TABLE IF NOT EXISTS assets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    asset_name VARCHAR(50) UNIQUE NOT NULL
);

-- Table to store expiry dates for each asset
CREATE TABLE IF NOT EXISTS expiries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    asset_id INT NOT NULL,
    expiry_date DATE NOT NULL,
    FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

-- Table to store option data (latest)
CREATE TABLE IF NOT EXISTS option_chain (
    id INT AUTO_INCREMENT PRIMARY KEY,
    asset_id INT NOT NULL,
    expiry_id INT NOT NULL,
    strike_price INT NOT NULL,
    option_type ENUM('CE', 'PE') NOT NULL,
    identifier VARCHAR(100) UNIQUE NOT NULL,
    open_interest INT NOT NULL,
    change_in_oi INT NOT NULL,
    total_traded_volume INT NOT NULL,
    implied_volatility FLOAT NOT NULL,
    last_price FLOAT NOT NULL,
    bid_qty INT NOT NULL,
    bid_price FLOAT NOT NULL,
    ask_qty INT NOT NULL,
    ask_price FLOAT NOT NULL,
    total_buy_qty INT NOT NULL,
    total_sell_qty INT NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE,
    FOREIGN KEY (expiry_id) REFERENCES expiries(id) ON DELETE CASCADE
);

-- Indexing for faster queries
CREATE INDEX idx_asset_expiry ON option_chain(asset_id, expiry_id);
CREATE INDEX idx_identifier ON option_chain(identifier);

-- Create 6 buffer tables with the same structure as option_chain
CREATE TABLE IF NOT EXISTS buffer_1 LIKE option_chain;
CREATE TABLE IF NOT EXISTS buffer_2 LIKE option_chain;
CREATE TABLE IF NOT EXISTS buffer_3 LIKE option_chain;
CREATE TABLE IF NOT EXISTS buffer_4 LIKE option_chain;
CREATE TABLE IF NOT EXISTS buffer_5 LIKE option_chain;
CREATE TABLE IF NOT EXISTS buffer_6 LIKE option_chain;


CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(100) NOT NULL
);

CREATE TABLE strategies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    strategy_name VARCHAR(100),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE strategy_positions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    strategy_id INT,
    strike_price DECIMAL(10,2),
    expiry_date DATE,
    option_type ENUM('CE', 'PE'),
    lots INT,
    entry_price DECIMAL(10,2),
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

ALTER TABLE users MODIFY password VARCHAR(255);



