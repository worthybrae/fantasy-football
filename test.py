from models.team import Team
from helpers.database import create_connection

if __name__ == '__main__':
    con = create_connection()
    t = Team('steelers', 'pit')
    r = t.check_if_exists(con)
    print(r)