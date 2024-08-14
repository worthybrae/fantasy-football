CREATE TABLE IF NOT EXISTS game (
    id MEDIUMINT,
    year SMALLINT,
    week SMALLINT,
    home_team MEDIUMINT,
    away_team MEDIUMINT,
    home_score SMALLINT,
    away_score SMALLINT,
    PRIMARY KEY (id),
    FOREIGN KEY (home_team) REFERENCES team(id),
    FOREIGN KEY (away_team) REFERENCES team(id)
);
