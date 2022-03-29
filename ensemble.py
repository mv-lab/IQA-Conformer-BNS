import os

with open("output.txt", "w") as f:

    f1 = open(os.path.join("callbacks", "PIPAL", "IQA_Conformer", "eval", "output.txt"), "r")
    f2 = open(os.path.join("callbacks", "PIPAL", "IQA_Transformer", "eval", "output.txt"), "r")

    for line1, line2 in zip(f1.readlines(), f2.readlines()):

        path = line1.split(",")[0]
        score1 = float(line1.split(",")[1])
        score2 = float(line2.split(",")[1])
        score = (score1 + score2) / 2.0

        f.write("{},{:.6f}".format(path, score))
        f.write("\n")