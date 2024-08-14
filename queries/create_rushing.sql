CREATE TABLE IF NOT EXISTS rushing (
    player MEDIUMINT,
    game MEDIUMINT,
    team MEDIUMINT,
    attempts SMALLINT,
    yards SMALLINT,
    touchdowns SMALLINT,
    ybc SMALLINT,
    yac SMALLINT,
    broken_tackles SMALLINT,
    FOREIGN KEY (player) REFERENCES player(id),
    FOREIGN KEY (game) REFERENCES game(id),
    FOREIGN KEY (team) REFERENCES team(id)
);
