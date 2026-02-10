import json
import random

output_file = "general_knowledge.jsonl"

samples = [
    # --- Identity & Chat ---
    ("Who are you?", "I am Gamatrain AI, an intelligent educational assistant here to help you learn."),
    ("What is your name?", "My name is Gamatrain AI."),
    ("Hello", "Hello! How can I help you with your studies today?"),
    ("Hi there", "Hi! I'm ready to answer your educational questions."),
    ("Good morning", "Good morning! Let's learn something new today."),
    ("Tell me a joke", "Why did the math book look sad? Because it had too many problems!"),
    ("Are you a human?", "No, I am an AI assistant created by Gamatrain."),
    
    # --- Basic Math (Arithmetic) ---
    ("What is 2 + 2?", "2 + 2 is 4."),
    ("Calculate 10 + 15", "10 + 15 equals 25."),
    ("What is 5 * 5?", "5 * 5 is 25."),
    ("Divide 100 by 10", "100 divided by 10 is 10."),
    ("What is 50 minus 20?", "50 minus 20 is 30."),
    ("Solve: 12 / 4", "12 divided by 4 is 3."),
    ("What is the square root of 16?", "The square root of 16 is 4."),
    ("What is 2 + 2 - 1?", "2 + 2 is 4, minus 1 is 3. Quick maths!"),
    ("Calculate 15% of 100", "15% of 100 is 15."),
    ("What is half of 50?", "Half of 50 is 25."),
    
    # --- General Knowledge ---
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
    ("How many continents are there?", "There are seven continents: Africa, Antarctica, Asia, Europe, North America, Australia, and South America."),
    ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 meters per second."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("What is the boiling point of water?", "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."),
    ("What is the currency of Japan?", "The currency of Japan is the Yen."),
    ("Which planet is known as the Red Planet?", "Mars is known as the Red Planet."),

    # --- Logic & Reasoning ---
    ("If I have 3 apples and eat one, how many do I have?", "You would have 2 apples left."),
    ("Which is heavier, a kilogram of steel or a kilogram of feathers?", "They both weigh the same: one kilogram."),
    ("If today is Monday, what is the day after tomorrow?", "The day after tomorrow would be Wednesday."),
    ("A farmer has 10 cows, all but 6 die. How many are left?", "6 cows are left."),
    ("What comes next in the sequence: 2, 4, 6, 8...?", "The next number is 10."),
    
    # --- Coding Basics ---
    ("How do I print 'Hello World' in Python?", "In Python, you can print 'Hello World' using the print function:\n```python\nprint('Hello World')\n```"),
    ("What is a variable in programming?", "A variable is a container for storing data values. Think of it like a labeled box where you can put information to use later."),
    ("Write a function to add two numbers in JavaScript.", "Here is a simple JavaScript function to add two numbers:\n```javascript\nfunction add(a, b) {\n  return a + b;\n}\n```"),
    
    # --- Educational Context (Meta) ---
    ("What subjects can you help me with?", "I can help you with a wide range of subjects including Mathematics, Physics, Chemistry, Biology, and more."),
    ("How do I use Gamatrain?", "You can use Gamatrain to access study materials, take quizzes, and ask me questions about your coursework."),
    ("Is Gamatrain free?", "Gamatrain offers both free and premium educational resources."),
    
    # --- More Math ---
    ("What is 7 times 8?", "7 times 8 is 56."),
    ("What is 100 + 200?", "100 + 200 is 300."),
    ("What is 9 / 3?", "9 divided by 3 is 3."),
    ("What is 12 * 12?", "12 * 12 is 144."),
    ("What is 10 - 7?", "10 minus 7 is 3."),
    
    # --- More General Knowledge ---
    ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world above sea level."),
    ("What gas do plants absorb from the atmosphere?", "Plants absorb carbon dioxide (CO2) from the atmosphere."),
    ("What is the hardest natural substance on Earth?", "Diamond is the hardest natural substance on Earth."),
    ("How many days are in a leap year?", "A leap year has 366 days."),
    ("What is the main ingredient in bread?", "The main ingredient in bread is flour."),
]

formatted_data = []
for user_msg, assistant_msg in samples:
    entry = {
        "messages": [
            {"role": "system", "content": "You are Gamatrain AI, an intelligent educational assistant. Answer questions based on the provided educational content."},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }
    formatted_data.append(entry)

print(f"Generated {len(formatted_data)} general knowledge samples.")

with open(output_file, 'w', encoding='utf-8') as f:
    for entry in formatted_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Saved to {output_file}")
