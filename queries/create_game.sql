CREATE TABLE IF NOT EXISTS game (
    id VARCHAR(8),
    year SMALLINT,
    week SMALLINT,
    home_team VARCHAR(3),
    away_team VARCHAR(3),
    home_score SMALLINT,
    away_score SMALLINT,
    PRIMARY KEY (id),
    FOREIGN KEY (home_team) REFERENCES team(id),
    FOREIGN KEY (away_team) REFERENCES team(id)
);
