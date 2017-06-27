from random import randrange

guessed_letters = []
def choose_word():
	with open('words.txt', 'r') as words_list:
		num_lines = sum(1 for line in words_list)
		random_line = randrange(1,num_lines)
				
	words = open('words.txt')
	lines = words.readlines()
	word = lines[random_line]
	
	return word[0:-1]

def initialize_game(word, guessed_letters):
	word_length = len(word)
	for i in range(0, word_length):
		guessed_letters.append("_")
		i = i+1
	update_view(guessed_letters)

def player_turn(guessed_letters):
	global game_counter
	print "You have ", game_counter, " chances left\n" 
	letter_proposed = raw_input("What letter?\n")
	word_array = list(word)
	if str(letter_proposed) in word_array:
		guessed_letters[word_array.index(letter_proposed)] = letter_proposed
	else:
		game_counter -= 1		
	update_view(guessed_letters)
	if "_" not in guessed_letters:
		print("You won!")
		exit()

def update_view(guessed_letters):
	print(" ".join(guessed_letters))

def new_game():
	global game_counter
	play_again = raw_input("Play again? [Y] or [N]")
	print(play_again.strip())
	if play_again.strip() == "y":
		game_counter = 7
		main()
	else:
		exit()


def main():
	initialize_game(word, guessed_letters)
	while game_counter>0:
		player_turn(guessed_letters)
	print("you are out of luck!")
	new_game()

word = choose_word()
game_counter = 7

