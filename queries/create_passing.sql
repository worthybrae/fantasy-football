CREATE TABLE IF NOT EXISTS passing (
    player MEDIUMINT,
    game MEDIUMINT,
    team MEDIUMINT,
    completions SMALLINT,
    attempts SMALLINT,
    yards SMALLINT,
    cay SMALLINT,
    yac SMALLINT,
    drops SMALLINT,
    bad_throws SMALLINT,
    pressured SMALLINT,
    blitzed SMALLINT,
    scrambles SMALLINT,
    FOREIGN KEY (player) REFERENCES player(id),
    FOREIGN KEY (game) REFERENCES game(id),
    FOREIGN KEY (team) REFERENCES team(id)
);
