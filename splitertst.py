import random

# Class definition
class Pet:
    def __init__(self, name):
        self.name = name
        self.energy = 50

    # Function inside a class (method)
    def update_energy(self, amount):
        self.energy += amount
        if self.energy > 100:
            self.energy = 100
        elif self.energy < 0:
            self.energy = 0

# Standalone function
DUMMY_BONUS = 5
def get_random_boost():
    return random.randint(1, 15)

def main():
    my_pet = Pet("Boba")
    print(f"Starting game with {my_pet.name}!")
    
    # For loop to run a set number of startup checks
    print("Running initial checks:")
    for check_num in range(1, 4):
        print(f"-> Check {check_num} complete.")

    # While loop for action turns
    turns = 3
    while turns > 0:
        action = random.choice(["feed", "sleep"])
        
        # If statements for decision-making
        if action == "feed":
            boost = get_random_boost()
            my_pet.update_energy(boost)
            print(f"{my_pet.name} ate food. Energy up by {boost}!")
        else:
            my_pet.update_energy(-10)
            print(f"{my_pet.name} rested. Energy went down a bit.")
            
        turns -= 1

    print(f"Final energy for {my_pet.name}: {my_pet.energy}")

if __name__ == "__main__":
    main()
