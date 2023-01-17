import math
import random as rn

spielfeld = [" ",
             "1","2","3",
             "4","5","6",
             "7","8","9"]


def spielfeld_ausgeben():
    print(spielfeld[1] + "|" + spielfeld[2] + "|" + spielfeld[3] )
    print(spielfeld[4] + "|" + spielfeld[5] + "|" + spielfeld[6] )
    print(spielfeld[7] + "|" + spielfeld[8] + "|" + spielfeld[9] )

player1 = "X"
bot_player = "O"

def spielzug_player():
    spielzug = int(input(">"))
    if spielzug > 9 or spielzug < 1:
        print("Error: limitiertes Feld")
    else:
        return spielzug

def spielfeld_update(spielzug, player):
    if spielfeld[spielzug] is player1 or spielfeld[spielzug] is bot_player:
        #print("schon besetzt du idiot!")
        if player == player1:
            spielfeld_update(spielzug_player(), player1)
        else:
            spielfeld_update(spielzug_bot(), bot_player)
    else:
        spielfeld[spielzug] = player
        print(f"{player} wählt: {spielzug}")


def spielzug_bot():
    spielzug = rn.randint(1,9)
    if spielzug > 9 or spielzug < 1:
        print("Error: limitiertes Feld")
    else:
        return spielzug

def check_game():
    if spielfeld[1] == spielfeld[2] == spielfeld[3]:
        return "GEWONNEN"
    if spielfeld[4] == spielfeld[5] == spielfeld[6]:
        return "GEWONNEN"
    if spielfeld[7] == spielfeld[8] == spielfeld[9]:
        return "GEWONNEN"
    # Kontrolle auf Spalten
    if spielfeld[1] == spielfeld[4] == spielfeld[7]:
        return "GEWONNEN"
    if spielfeld[2] == spielfeld[5] == spielfeld[8]:
        return "GEWONNEN"
    if spielfeld[3] == spielfeld[6] == spielfeld[9]:
        return "GEWONNEN"
    # Kontrolle auf Diagonalen
    if spielfeld[1] == spielfeld[5] == spielfeld[9]:
        return "GEWONNEN"
    if spielfeld[7] == spielfeld[5] == spielfeld[3]:
        return "GEWONNEN"
    else:
        return ""

spiel = True
print("Du bist: ", player1)
spielfeld_ausgeben()
while spiel == True:
    print("-----DU BIST AM ZUG-----")
    spielfeld_update(spielzug_player(), player1)
    spielfeld_ausgeben()
    print(check_game())
    if check_game() == "GEWONNEN":
        break
    print("-----DEIN ZUG IST VORBEI-----")
    print("-----BOT IST AM ZUG-----")
    bot_chose = spielzug_bot()
    spielfeld_update(bot_chose, bot_player)
    spielfeld_ausgeben()
    print(check_game())
    if check_game() == "GEWONNEN":
        break
    print("-----BOTS ZUG IST VORBEI-----")