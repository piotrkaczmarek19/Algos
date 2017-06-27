from random import *
reds = []
blacks = []
player_bet = []
spin_result = []


for i in range(1,49):
	if i%2 == 0:
		reds.append(i)
	else:
		blacks.append(i)

def welcome_function():
	print("*********************** Welcome to the casino! ***********************")
	return True

def is_valid_number(bet):
	try:
		bet_number = int(bet)
		return True
	except:
		return False

def setup_bet():
	bet =  input("Where do you want to place your bet? Red[R], Black[B] or Number[0-49]?")	
	if bet.upper() in ["R", "B"]:
		player_bet.append(bet.upper())
		player_bet.append("color")
		print(player_bet)
	elif is_valid_number(bet) == True:
		player_bet.append(bet)
		player_bet.append("number")
		print(player_bet)
	else:
		print("Please enter a valid value")
		setup_bet()

def bet_amount():
	try:
		amount = int(input("How much will you bet? [0-1000]"))
		return amount
	except:
		print("Jeez. Numerical value")
		bet_amount()

def spin_wheel():
	result = randrange(49)
	if result in reds:
		spin_result.append(result)
		spin_result.append("R")
	elif result in blacks:
		spin_result.append(result)
		spin_result.append("B")

def check_winnings(bet):
	print("The ball landed on ", spin_result[0]," ! It is a ", spin_result[1])
	if player_bet[0] in spin_result:
		if player_bet[1] == "color":
			print("Win!")
			return bet * 2
		elif player_bet[1] == "number":
			print("Win!")
			return bet *10
	else:
		print("Lose!")
		return 0

def update_wallet(win, wallet):
	wallet+=win

#New turn and reset result list as well as player's choice
def continue_game(player_wallet, spin_results):
	print(player_wallet)
	del spin_result[:]
	del player_bet[:]
	decision = input("Continue? [Y] or [N]")
	if decision.upper() == "N":
		print("Bye!")
		return False
	else:
		return True

def game_turn(player_wallet):
	setup_bet()
	bet = bet_amount()
	player_wallet -= bet
	spin_wheel()
	win = check_winnings(bet)
	player_wallet += win
	game_on = continue_game(player_wallet, spin_result)

def main():
	player_wallet = 1000
	game_on = welcome_function()
	while game_on == True:
		game_turn(player_wallet)
	

