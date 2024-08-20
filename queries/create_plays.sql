CREATE TABLE IF NOT EXISTS plays (
    game VARCHAR(8),
    quarter SMALLINT,
    time VARCHAR(5),
    down SMALLINT,
    to_go SMALLINT,
    location VARCHAR(6),
    type VARCHAR(10),
    action varchar(10),
    passer varchar(255),
    rusher varchar(255),
    receiver varchar(255),
    details VARCHAR(255),
    FOREIGN KEY (game) REFERENCES game(id),
    FOREIGN KEY (passer) REFERENCES player(id),
    FOREIGN KEY (rusher) REFERENCES player(id),
    FOREIGN KEY (receiver) REFERENCES player(id)
);