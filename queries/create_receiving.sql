CREATE TABLE IF NOT EXISTS receiving (
    player MEDIUMINT,
    game MEDIUMINT,
    team MEDIUMINT,
    targets SMALLINT,
    receptions SMALLINT,
    yards SMALLINT,
    touchdowns SMALLINT,
    yac SMALLINT,
    ybc SMALLINT,
    broken_tackles SMALLINT,
    drops SMALLINT,
    FOREIGN KEY (player) REFERENCES player(id),
    FOREIGN KEY (game) REFERENCES game(id),
    FOREIGN KEY (team) REFERENCES team(id)
);
