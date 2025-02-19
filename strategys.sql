INSERT INTO option_data.strategies (user_id, strategy_name) VALUES
(NULL, 'Long Call'),
(NULL, 'Long Put'),
(NULL, 'Covered Call'),
(NULL, 'Protective Put'),
(NULL, 'Bull Call Spread'),
(NULL, 'Bear Put Spread'),
(NULL, 'Iron Condor'),
(NULL, 'Straddle'),
(NULL, 'Strangle'),
(NULL, 'Butterfly Spread');




-- 🔷 Long Call
INSERT INTO option_data.strategy_positions (strategy_id, strike_price, expiry_date, option_type, lots, entry_price)
VALUES ((SELECT id FROM option_data.strategies WHERE strategy_name = 'Long Call'), 100, NULL, 'CE', 1, 5.00);

-- 🔷 Long Put
INSERT INTO option_data.strategy_positions (strategy_id, strike_price, expiry_date, option_type, lots, entry_price)
VALUES ((SELECT id FROM option_data.strategies WHERE strategy_name = 'Long Put'), 100, NULL, 'PE', 1, 4.00);

-- 🔷 Covered Call (Long Stock + Short Call)
INSERT INTO option_data.strategy_positions (strategy_id, strike_price, expiry_date, option_type, lots, entry_price)
VALUES
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Covered Call'), 100, NULL, 'CE', -1, 3.00);

-- 🔷 Protective Put (Long Stock + Long Put)
INSERT INTO option_data.strategy_positions (strategy_id, strike_price, expiry_date, option_type, lots, entry_price)
VALUES
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Protective Put'), 100, NULL, 'PE', 1, 3.50);

-- 🔷 Bull Call Spread (Long Call + Short Call)
INSERT INTO option_data.strategy_positions (strategy_id, strike_price, expiry_date, option_type, lots, entry_price)
VALUES
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Bull Call Spread'), 100, NULL, 'CE', 1, 4.00),
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Bull Call Spread'), 110, NULL, 'CE', -1, 2.00);

-- 🔷 Bear Put Spread (Long Put + Short Put)
INSERT INTO option_data.strategy_positions (strategy_id, strike_price, expiry_date, option_type, lots, entry_price)
VALUES
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Bear Put Spread'), 110, NULL, 'PE', 1, 4.50),
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Bear Put Spread'), 100, NULL, 'PE', -1, 2.50);

-- 🔷 Iron Condor (4 legs: Short Put + Long Put + Short Call + Long Call)
INSERT INTO option_data.strategy_positions (strategy_id, strike_price, expiry_date, option_type, lots, entry_price)
VALUES
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Iron Condor'), 90, NULL, 'PE', 1, 1.50),
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Iron Condor'), 95, NULL, 'PE', -1, 2.00),
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Iron Condor'), 105, NULL, 'CE', -1, 2.00),
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Iron Condor'), 110, NULL, 'CE', 1, 1.50);

-- 🔷 Straddle (Long Call + Long Put, same strike)
INSERT INTO option_data.strategy_positions (strategy_id, strike_price, expiry_date, option_type, lots, entry_price)
VALUES
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Straddle'), 100, NULL, 'CE', 1, 5.00),
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Straddle'), 100, NULL, 'PE', 1, 4.50);

-- 🔷 Strangle (Long Call + Long Put, different strikes)
INSERT INTO option_data.strategy_positions (strategy_id, strike_price, expiry_date, option_type, lots, entry_price)
VALUES
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Strangle'), 105, NULL, 'CE', 1, 4.00),
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Strangle'), 95, NULL, 'PE', 1, 3.50);

-- 🔷 Butterfly Spread (Long Call + 2 Short Calls + Long Call)
INSERT INTO option_data.strategy_positions (strategy_id, strike_price, expiry_date, option_type, lots, entry_price)
VALUES
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Butterfly Spread'), 95, NULL, 'CE', 1, 3.00),
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Butterfly Spread'), 100, NULL, 'CE', -2, 2.00),
((SELECT id FROM option_data.strategies WHERE strategy_name = 'Butterfly Spread'), 105, NULL, 'CE', 1, 1.50);
