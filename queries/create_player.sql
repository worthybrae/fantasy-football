CREATE TABLE IF NOT EXISTS player (
    id MEDIUMINT,
    name VARCHAR(255),
    position VARCHAR(10),
    years_played SMALLINT,
    team MEDIUMINT,
    PRIMARY KEY (id),
    FOREIGN KEY (team) REFERENCES team(id)
);
