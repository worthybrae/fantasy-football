CREATE TABLE IF NOT EXISTS snaps (
    player MEDIUMINT,
    game MEDIUMINT,
    team MEDIUMINT,
    snaps SMALLINT,
    FOREIGN KEY (player) REFERENCES player(id),
    FOREIGN KEY (game) REFERENCES game(id),
    FOREIGN KEY (team) REFERENCES team(id)
);
