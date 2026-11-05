import pandas as pd
import random

grammar = [
("what is a noun","a noun is a word that names a person place thing or idea"),
("what is a verb","a verb is a word that expresses an action or state"),
("what is an adjective","an adjective describes a noun"),
("what is an adverb","an adverb modifies a verb adjective or another adverb"),
("what is a sentence","a sentence is a group of words that expresses a complete thought"),
("what is grammar","grammar is the system of rules for using a language"),
]

vocab = [
("what does happy mean","feeling pleasure or joy"),
("what does intelligent mean","able to learn understand and think logically"),
("what does curious mean","eager to learn or know something"),
("what does confident mean","feeling sure about yourself"),
("what does brave mean","showing courage"),
]

sentences = [
("make a sentence using happy","i am happy today"),
("make a sentence using brave","the firefighter is brave"),
("make a sentence using curious","the child is curious about the world"),
("make a sentence using intelligent","she is an intelligent student"),
]

corrections = [
("correct this sentence she dont like apples","she does not like apples"),
("correct this sentence he go to school yesterday","he went to school yesterday"),
("correct this sentence they was playing","they were playing"),
]

gk = [
("who discovered gravity","isaac newton"),
("who invented the telephone","alexander graham bell"),
("who invented the light bulb","thomas edison"),
("what is the capital of france","paris"),
("what planet is known as the red planet","mars"),
]

shortcuts = [
("what does lol mean","laughing out loud"),
("what does brb mean","be right back"),
("what does btw mean","by the way"),
("what does idk mean","i do not know"),
]

base = grammar + vocab + sentences + corrections + gk + shortcuts

prefixes = [
"tell me",
"can you explain",
"do you know",
"please tell me",
"i want to know",
"help me understand",
]

rows = []

TARGET = 200000

while len(rows) < TARGET:
    q,a = random.choice(base)

    if random.random() < 0.6:
        q = random.choice(prefixes) + " " + q

    rows.append({
        "input": q.lower(),
        "output": a.lower()
    })

df = pd.DataFrame(rows)

df.to_csv("complete_dataset_200k.csv",index=False)

print("Dataset generated!")
print("Total rows:",len(df))